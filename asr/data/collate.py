"""Collate function for batching variable-length audio."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torchaudio.functional as F_audio

from asr.text.tokenizer import CharTokenizer


@dataclass
class WavBatch:
    wav: torch.Tensor        # (B, max_samples)
    wav_lens: torch.Tensor   # (B,)
    targets: torch.Tensor    # (B, max_target_len)
    target_lens: torch.Tensor  # (B,)
    texts: List[str]
    utt_ids: List[str]


def collate_wav(
    batch: List[Dict[str, Any]],
    tokenizer: CharTokenizer,
    target_sr: int = 16000,
) -> WavBatch:
    """Pad waveforms and encode text targets."""
    wavs = []
    texts = []
    utt_ids = []

    for item in batch:
        wav = item["wav"]
        sr = item["sample_rate"]
        # Mono
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        # Resample if needed
        if sr != target_sr:
            wav = F_audio.resample(wav, sr, target_sr)
        wavs.append(wav)
        texts.append(item["text"])
        utt_ids.append(item["utt_id"])

    # Pad waveforms
    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(wav_lens.max().item())
    padded = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        padded[i, :w.shape[0]] = w

    # Encode targets
    encoded = [tokenizer.encode(t.lower()) for t in texts]
    target_lens = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    max_tgt = int(target_lens.max().item()) if encoded else 0
    targets = torch.zeros(len(encoded), max(max_tgt, 1), dtype=torch.long)
    for i, e in enumerate(encoded):
        if e:
            targets[i, :len(e)] = torch.tensor(e, dtype=torch.long)

    return WavBatch(
        wav=padded,
        wav_lens=wav_lens,
        targets=targets,
        target_lens=target_lens,
        texts=texts,
        utt_ids=utt_ids,
    )
