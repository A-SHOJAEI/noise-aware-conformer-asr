from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

import torchaudio


@dataclass(frozen=True)
class W2V2Config:
    pretrained: bool
    variant: str  # "base"


class Wav2Vec2CTC(nn.Module):
    def __init__(self, vocab_size: int, cfg: W2V2Config):
        super().__init__()
        if cfg.variant != "base":
            raise ValueError(f"Only variant=base is supported, got {cfg.variant}")

        if cfg.pretrained:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.w2v2 = bundle.get_model()
        else:
            # Same architecture as base, randomly initialized (useful for smoke tests).
            # torchaudio doesn't expose a direct "base config" object, so we build it from documented defaults.
            self.w2v2 = torchaudio.models.wav2vec2_model(
                extractor_mode="group_norm",
                extractor_conv_layer_config=[
                    (512, 10, 5),
                    (512, 3, 2),
                    (512, 3, 2),
                    (512, 3, 2),
                    (512, 3, 2),
                    (512, 2, 2),
                    (512, 2, 2),
                ],
                extractor_conv_bias=False,
                encoder_embed_dim=768,
                encoder_projection_dropout=0.1,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=12,
                encoder_num_heads=12,
                encoder_attention_dropout=0.1,
                encoder_ff_interm_features=3072,
                encoder_ff_interm_dropout=0.1,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.05,
            )

        self.ctc = nn.Linear(768, vocab_size)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # wav: (B,1,T)
        x = wav.squeeze(1)
        # torchaudio wav2vec2 forward signature differs across versions.
        out = None
        try:
            out = self.w2v2(x, wav_lens)
        except TypeError:
            out = self.w2v2(x)

        if isinstance(out, tuple) and len(out) == 2:
            feats, feat_lens = out
        else:
            feats = out
            # Approximate output lengths by proportional downsampling (conservative fallback).
            feat_lens = (wav_lens.to(torch.float32) / 320.0).to(torch.long).clamp(min=1)
        logits = self.ctc(feats)  # (B,TT,V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, feat_lens
