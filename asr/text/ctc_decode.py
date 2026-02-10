from __future__ import annotations

from typing import List, Sequence

import torch

from asr.text.tokenizer import CharTokenizer


def greedy_ctc_decode(emissions: torch.Tensor, tokenizer: CharTokenizer) -> List[str]:
    """
    emissions: (B, T, V) log-probs or logits.
    """
    if emissions.dim() != 3:
        raise ValueError(f"Expected (B,T,V), got {tuple(emissions.shape)}")
    ids = torch.argmax(emissions, dim=-1)  # (B,T)
    hyps: List[str] = []
    blank_id = 0
    for seq in ids.tolist():
        out: List[int] = []
        prev = None
        for t in seq:
            if t == blank_id:
                prev = t
                continue
            if prev is not None and t == prev:
                continue
            out.append(t)
            prev = t
        hyps.append(tokenizer.decode_ctc(out))
    return hyps

