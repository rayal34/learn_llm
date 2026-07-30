import os
import time
from typing import cast

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2 import constants, utils
from gpt2.models import GPT, GPTConfig

use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = (
    utils.determine_ddp()
)

utils.seed_everything(1337)

assert constants.TOTAL_BATCH_SIZE % (constants.B * constants.T * ddp_world_size) == 0
grad_accumulation_steps = constants.TOTAL_BATCH_SIZE // (
    constants.B * constants.T * ddp_world_size
)
if master_process:
    print(
        f"Total batch size: {constants.TOTAL_BATCH_SIZE}, gradient accumulation steps: {grad_accumulation_steps}"
    )

train_loader = utils.DataLoaderLite(
    constants.B, constants.T, ddp_rank, ddp_world_size, "train"
)
val_loader = utils.DataLoaderLite(
    constants.B, constants.T, ddp_rank, ddp_world_size, "valid"
)

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if "cuda" in device:
    model = torch.compile(model)

if use_ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if use_ddp else model
raw_model = cast(GPT, raw_model)

optimizer = raw_model.configure_optimizers(
    constants.WEIGHT_DECAY, constants.MAX_LR, device
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w"):
    pass


for step in range(constants.MAX_STEPS):
    t0 = time.time()
    last_step = step == constants.MAX_STEPS - 1

    if step == constants.EVAL_INTERVAL or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                val_loss += loss.detach()

            val_loss /= val_loss_steps

        if use_ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"Validation loss: {val_loss.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step: {step}, val_loss = {val_loss.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                ckpt_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                utils.save_checkpoint(ckpt_path, model, optimizer, train_loader, step)

    model.train()
    optimizer.zero_grad()
    train_loss = 0.0
    for micro_step in range(grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if use_ddp:
            model.require_backward_grad_sync = micro_step == grad_accumulation_steps - 1
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss = loss / grad_accumulation_steps
        train_loss += loss.detach()

        loss.backward()
    if use_ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    lr = utils.get_cosine_decay_lr(
        step,
        constants.WARMUP_STEPS,
        constants.MAX_STEPS,
        constants.MAX_LR,
        constants.MIN_LR,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step: {step} | train loss: {train_loss:.6f} | lr: {lr} |  norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"step: {step}, train loss: {train_loss:.6f}\n")

if use_ddp:
    dist.destroy_process_group()
