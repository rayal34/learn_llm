from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.attn = nn.Linear(config.n_embed, 3 * config.n_embed)  # qkv matrix
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.proj.NANOGPT_SCALE_INIT = 1  # type: ignore
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones((config.context_length, config.context_length)).view(
                    1, 1, config.context_length, config.context_length
                )
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, context_length, n_embed)
        qkv = self.attn(x)  # (batch_size, context_length, 3 * n_embed)
        q, k, v = qkv.split(
            self.n_embed, -1
        )  # each matrix = (batch_size, context_length, n_embed)

        # each matrix = (batch_size, n_head, context_length, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attn = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        # attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, n_heads, T, T) @ (B, n_heads, T, head_size) --> (B, n_heads, T, head_size)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (B, n_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)  # (B, n_embed, n_embed)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(
            config.n_embed,
            4 * config.n_embed,  # 4 * d_model = d_ff (from original paper)
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.linear2.NANOGPT_SCALE_INIT = 1  # type: ignore

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.mha = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.context_length, config.n_embed),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "lnf": nn.LayerNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, T = idx.shape
        assert T <= self.config.context_length
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer["wpe"](pos)
        tok_emb = self.transformer["wte"](idx)
        x = pos_emb + tok_emb
        for block in self.transformer["h"]:  # type: ignore
            x = block(x)
        x = self.transformer["lnf"](x)
        logits = self.lm_head(x)

        return logits

    def configure_optimizers(self, weight_decay, lr, device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params, no_decay_params = [], []
        for p in param_dict.values():
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"Number of weight-decayed tensors: {len(decay_params)} with {num_decay_params:,} parameters"
        )
        print(
            f"Number of non-weight-decayed tensors: {len(no_decay_params)} with {num_no_decay_params:,} parameters"
        )
        use_fused = "cuda" in device
        optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, fused=use_fused)
        return optimizer
