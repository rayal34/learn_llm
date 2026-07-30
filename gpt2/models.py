import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = (
        50304  # gpt vocab size = 50257, padded to 50304 to make it a multiple of 64
    )
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embed, 3 * config.n_embed, bias=config.bias
        )  # qkv matrix
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, context_length, n_embed)
        qkv = self.c_attn(x)  # (batch_size, context_length, 3 * n_embed)
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
            q, k, v, dropout_p=self.dropout, is_causal=True
        )  # (B, n_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))  # (B, n_embed, n_embed)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embed,
            4 * config.n_embed,  # 4 * d_model = d_ff (from original paper)
            bias=config.bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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
                "ln_f": LayerNorm(config.n_embed, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(f"Number of parameters:  {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        x = self.transformer["ln_f"](x)
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
