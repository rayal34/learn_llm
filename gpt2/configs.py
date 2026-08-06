from dataclasses import dataclass
from typing import Literal


@dataclass
class BaseConfig:
    @classmethod
    def to_dict(cls):
        return {
            field.name: getattr(cls, field.name)
            for field in cls.__dataclass_fields__.values()
        }


@dataclass
class TrainConfig(BaseConfig):
    init_from: Literal["scratch", "resume_from_latest", "resume_from_specific"] = (
        "scratch"
    )
    resume_ckpt_step: str | None = None
    ckpt_filename_format: str = "model_{step:05d}.pt"
    total_batch_size: int = 524288  # in tokens, 2 ** 19 (a "nice" number)
    batch_size: int = 2  # micro batch size
    max_lr: float = 6e-4
    min_lr_factor: float = 0.1
    warmup_steps: int = (
        715  # gpt3 used 375e6 warmup tokens.  375e6 / TOTAL_BATCH_SIZE = 715
    )
    max_steps: int = 19073  # TOTAL_TOKENS (10e9) / TOTAL_BATCH_SIZE = 19073
    ckpt_interval: int = 100
    weight_decay: float = 0.1
    eval_interval: int = 100
    val_loss_steps: int = 20
    log_dir: str = "log"
    log_file: str = "log.txt"
    wandb_log: bool = True
    wandb_project: str = "gpt2"
    wandb_run_name: str | None = None


@dataclass
class GPTConfig(BaseConfig):
    context_length: int = 1024
    vocab_size: int = (
        50304  # gpt vocab size = 50257, padded to 50304 to make it a multiple of 64
    )
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = False


@dataclass
class DataConfig(BaseConfig):
    data_root: str = "gpt2/edu_fineweb10B"
