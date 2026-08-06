import argparse
import os
from datetime import datetime
from typing import cast

import torch
import yaml
from torch import distributed as dist

from gpt2 import utils
from gpt2.models import GPT


def main(config_file):
    use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = (
        utils.determine_ddp()
    )

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["log_dir"]
    log_file = config["log_file"]
    log_path = os.path.join(log_dir, log_file)

    wandb_log = config["wandb_log"]
    wandb_project = config["wandb_project"]
    wandb_run_name = (
        config["wandb_run_name"]
        if config["wandb_run_name"] is not None
        else f"gpt2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # noqa: DTZ005
    )

    if wandb_log and master_process:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    else:
        wandb = None

    model = utils.get_model(config, use_ddp, ddp_local_rank, device)
    raw_model = model.module if use_ddp else model
    raw_model = cast(GPT, raw_model)

    optimizer = raw_model.configure_optimizers(
        config["weight_decay"], config["max_lr"], device
    )
    train_loader = utils.DataLoaderLite(
        config["data_root"],
        config["batch_size"],
        config["context_length"],
        ddp_rank,
        ddp_world_size,
        "train",
    )
    val_loader = utils.DataLoaderLite(
        config["data_root"],
        config["batch_size"],
        config["context_length"],
        ddp_rank,
        ddp_world_size,
        "val",
    )

    assert config["init_from"] in [
        "scratch",
        "resume_from_latest",
        "resume_from_specific",
    ]
    if config["init_from"] == "scratch":
        utils.seed_everything(1337)
        step = 0
        utils.init_log_file(log_dir, log_file)

    else:
        if config["init_from"] == "resume_from_latest":
            available_ckpts = sorted(
                [file for file in os.listdir(config["log_dir"]) if file.endswith(".pt")]
            )
            ckpt_filename = available_ckpts[-1]
        elif config["init_from"] == "resume_from_specific":
            ckpt_filename = config["ckpt_filename_format"].format(
                step=config["resume_ckpt_step"]
            )
        ckpt_path = os.path.join(config["log_dir"], ckpt_filename)
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        step = utils.load_checkpoint(ckpt_path, model, optimizer, train_loader, device)

    if device.startswith("cuda"):
        print("Compiling model")
        model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
    utils.train_loop(
        config,
        model,
        optimizer,
        train_loader,
        val_loader,
        use_ddp,
        ddp_world_size,
        device.split(":")[0],
        master_process,
        log_dir,
        log_path,
        step=step,
        wandb=wandb,
    )

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--config", type=str, default="./config.yml")
    args = arg_parse.parse_args()
    main(args.config)
