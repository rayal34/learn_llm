import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.configs import GPTConfig
from gpt2.models import GPT


def init_log_file(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, "w"):
        pass


def compute_gradient_accumulation_steps(
    total_batch_size, micro_batch_size, context_length, ddp_world_size
):
    grad_accumulation_steps = total_batch_size // (
        micro_batch_size * context_length * ddp_world_size
    )

    assert grad_accumulation_steps > 0
    return grad_accumulation_steps


def get_model(
    config: GPTConfig,
    use_ddp: bool,
    ddp_local_rank: int,
    device: str,
):

    model = GPT(config)

    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model


def train_loop(
    train_config,
    model_config,
    model,
    optimizer,
    train_loader,
    val_loader,
    use_ddp,
    ddp_world_size,
    device,
    master_process,
    log_dir: str,
    log_path: str,
    step: int = 0,
    wandb=None,
):

    device_type = device.split(":")[0]
    grad_accumulation_steps = compute_gradient_accumulation_steps(
        train_config.total_batch_size,
        train_config.batch_size,
        model_config.context_length,
        ddp_world_size,
    )
    if master_process:
        print(
            f"Total batch size: {train_config.total_batch_size}, gradient accumulation steps: {grad_accumulation_steps}"
        )

    max_steps = train_config.max_steps
    eval_interval = train_config.eval_interval

    while step < max_steps:
        lr = get_cosine_decay_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.max_lr,
            train_config.max_lr * train_config.min_lr_factor,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t0 = time.time()
        last_step = step == max_steps - 1

        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        # train
        for micro_step in range(grad_accumulation_steps):
            x, y = train_loader.next_batch()
            if device_type == "cuda":
                x, y = (
                    x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True),
                )
            else:
                x, y = x.to(device), y.to(device)
            if use_ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accumulation_steps - 1
                )
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss = loss / grad_accumulation_steps
            train_loss += loss.detach()

            loss.backward()
        if use_ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0

        # eval
        if step % eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(train_config.val_loss_steps):
                    x, y = val_loader.next_batch()
                    if device_type == "cuda":
                        x, y = (
                            x.pin_memory().to(device, non_blocking=True),
                            y.pin_memory().to(device, non_blocking=True),
                        )
                    else:
                        x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.shape[-1]), y.view(-1)
                        )

                    val_loss += loss.detach()

                val_loss /= train_config.val_loss_steps

            if use_ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "train/loss": train_loss.item(),
                        "val/loss": val_loss.item(),
                        "lr": lr,
                    }
                )
            if master_process:
                print(f"Validation loss: {val_loss.item():.4f}")
                with open(log_path, "a") as f:
                    f.write(
                        f"step: {step}, train loss: {train_loss.item():.4f}, val_loss = {val_loss.item():.4f}\n"
                    )
        if (
            master_process
            and step > 0
            and (step % train_config.ckpt_interval == 0 or last_step)
        ):
            ckpt_path = os.path.join(
                log_dir, train_config.ckpt_filename_format.format(step=step)
            )
            save_checkpoint(ckpt_path, model, optimizer, train_loader, step)

        tokens_processed = (
            train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step: {step} | train loss: {train_loss.item():.4f} | lr: {lr} |  norm: {norm.item():.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

        step += 1


def determine_ddp():
    use_ddp = int(os.environ.get("RANK", "-1")) != -1
    if use_ddp:
        assert torch.cuda.is_available()
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        print(f"Using DDP with world size {ddp_world_size}")
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

    return use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_cosine_decay_lr(step, warmup_steps, max_steps, max_lr, min_lr):

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


def save_checkpoint(filepath, model, optimizer, train_loader, step):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dataloader": {
            "current_shard": train_loader.current_shard,
            "base_position": train_loader.base_position,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        "step": step,
    }
    torch.save(ckpt, filepath)


def load_checkpoint(filepath, model, optimizer, train_loader, device):
    ckpt = torch.load(filepath, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    train_loader.current_shard = ckpt["dataloader"]["current_shard"]
    train_loader.base_position = ckpt["dataloader"]["base_position"]

    rng = ckpt["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])

    step = ckpt["step"]
    return step


class DataLoaderLite:
    def __init__(self, data_root, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = sorted([shard for shard in shards if split in shard])
        shards = [os.path.join(data_root, shard) for shard in shards]
        assert len(shards) > 0
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.base_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        pos = self.base_position + B * T * self.process_rank
        window = self.tokens[pos : pos + B * T + 1]
        x = window[:-1].view(B, T)
        y = window[1:].view(B, T)
        self.base_position += B * T * self.num_processes
        if self.base_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.base_position = 0
        return x, y
