import inspect
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embed, 4 * config.n_embed
        )  # the 4x factor is from the original attention paper.  4 x d_model = d_ff
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # kvq projections for all heads, in a batch
        # 3 is for the kvq
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                )
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, seq_len, n_embed)

        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # each tensor = (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, n_head, T, head_size)

        # att = (q @ k.transpose(-2, -1)) * (
        #     1.0 / math.sqrt(k.shape[-1])
        # )  # (B, n_heads, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # # att @ v is a weighted sum of the values
        # y = (
        #     att @ v
        # )  # (B, n_heads, T, T) * (B, n_heads, T, head_size) --> (B, n_heads, T, head_size)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # this transpose and view are concat-ing 6 head matrices side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing
        self.transformer["wte"].weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        # scaling for the residual paths to get the std back to unit-std.  Scale by 1/sqrt(n_residuals  )
        # 2 comes from the fact that each block has two residual paths
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx shape (B, T)
        _, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # (T, n_embed)
        pos_emb = self.transformer["wpe"](pos)
        # (B, T, n_embed)
        tok_emb = self.transformer["wte"](idx)
        x = pos_emb + tok_emb
        for block in self.transformer["h"]:  # type: ignore
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print(f"loading weights from pretrained gpt: {model_type}")

        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embed": 768},
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embed": 1024},
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embed": 1280},
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embed": 1600},
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith(".attn.bias")]  # discard the tril

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k
            for k in sd_hf
            if not k.endswith(".attn.masked_bias") or not k.endswith("attn.bias")
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = sorted([shard for shard in shards if split in shard])
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(shards) > 0
        if master_process:
            print(f"Found {len(shards)} for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = B * T * self.process_rank

    def next_batch(self):
        buf = self.tokens[
            self.current_position : self.current_position + self.B * self.T + 1
        ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(
            self.tokens
        ):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y


ddp = int(os.environ.get("RANK", "-1")) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)


total_batch_size = 524288  # in tokens, 2 ** 19 (a "nice" number)
B = 2  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(
        f"total desired batch_size: {total_batch_size}, gradient accumulation steps: {grad_accum_steps}"
    )

train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="train")
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="val")

torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if "cuda" in device:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # type: ignore

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # 375e6 warmup tokens (from gpt paper) / 2e19 = 715
max_steps = 19073  # 10e9 (10B) tokens, roughly, 2 ** 19 tokens/step (batch size). 10e9 / 2 ** 19 = 19073


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(  # type: ignore
    weight_decay=0.1, learning_rate=6e-4, device=device
)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w"):
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    if step % 100 == 0:
        model.eval()  # type: ignore
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                model_checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                model_checkpoint = {
                    "model": raw_model.state_dict(),  # type: ignore
                    "config": raw_model.config,  # type: ignore
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(model_checkpoint, model_checkpoint_path)

                optimizer_checkpoint_path = os.path.join(log_dir, f"opt_{step:05d}.pt")
                opt_checkpoint = {"optimizer": optimizer.state_dict(), "step": step}
                torch.save(opt_checkpoint, optimizer_checkpoint_path)

    model.train()  # type: ignore
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # to account for the lack of mean reduction
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # type: ignore
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type:ignore
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / (t1 - t0)
    if master_process:
        print(
            f"step: {step} | loss: {loss_accum:.6f} | lr: {lr} |  norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )

if ddp:
    destroy_process_group()
