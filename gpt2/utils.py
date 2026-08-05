import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist


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


def get_cosine_decay_lr(iteration, warmup_steps, max_steps, max_lr, min_lr):

    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps
    if iteration > max_steps:
        return min_lr

    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
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
            "current_position": train_loader.current_position,
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
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        window = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = window[:-1].view(B, T)
        y = window[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
