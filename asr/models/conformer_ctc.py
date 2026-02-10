from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = self.ln(x).transpose(1, 2)  # (B,D,T)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)  # GLU
        y = self.dw(y)
        y = self.bn(y)
        y = swish(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2)


class FiLM(nn.Module):
    def __init__(self, d_model: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        # embed: (B,E)
        gamma_beta = self.proj(embed)  # (B,2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class NoiseConditioner(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) early encoder states
        B, T, D = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < x_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (B,T,1)
        xm = x.masked_fill(~mask, 0.0)
        denom = x_lens.clamp(min=1).unsqueeze(1).to(x.dtype)
        mean = xm.sum(dim=1) / denom
        var = ((xm - mean.unsqueeze(1)) ** 2).masked_fill(~mask, 0.0).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-5)
        stats = torch.cat([mean, std], dim=-1)
        return self.net(stats)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_mult=ff_mult, dropout=dropout)
        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, ff_mult=ff_mult, dropout=dropout)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        y = self.mha_ln(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        y = self.mha_drop(y)
        x = x + y
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_ln(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,T,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.dtype)


@dataclass(frozen=True)
class NoiseConditioningConfig:
    enabled: bool
    embed_dim: int


class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_mult: int,
        conv_kernel: int,
        dropout: float,
        noise_conditioning: NoiseConditioningConfig,
    ):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(n_mels), nn.Linear(n_mels, d_model))
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads=n_heads, ff_mult=ff_mult, conv_kernel=conv_kernel, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.noise_conditioning = noise_conditioning
        if noise_conditioning.enabled:
            self.conditioner = NoiseConditioner(in_dim=d_model, embed_dim=noise_conditioning.embed_dim)
            self.films = nn.ModuleList([FiLM(d_model=d_model, embed_dim=noise_conditioning.embed_dim) for _ in range(num_layers)])
        else:
            self.conditioner = None
            self.films = None

        self.ctc = nn.Linear(d_model, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: (B,T,F)
        B, T, _ = feats.shape
        key_padding_mask = torch.arange(T, device=feats.device).unsqueeze(0) >= feat_lens.unsqueeze(1)

        x = self.inp(feats)
        x = self.pos(x)

        cond = None
        if self.noise_conditioning.enabled and self.conditioner is not None and self.films is not None:
            cond = self.conditioner(x, feat_lens)  # (B,E)

        for i, blk in enumerate(self.blocks):
            if cond is not None:
                x = self.films[i](x, cond)
            x = blk(x, key_padding_mask=key_padding_mask)

        logits = self.ctc(x)  # (B,T,V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, feat_lens
