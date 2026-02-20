"""Dataset that reads audio manifests in JSONL format."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class ManifestRow:
    utt_id: str
    audio_path: str
    text: str
    duration: float


class ManifestDataset(Dataset):
    """Reads a JSONL manifest with fields: utt_id, audio_path, text, duration."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows: List[ManifestRow] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.rows.append(ManifestRow(
                    utt_id=str(r["utt_id"]),
                    audio_path=str(r["audio_path"]),
                    text=str(r["text"]),
                    duration=float(r.get("duration", 0.0)),
                ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        wav, sr = torchaudio.load(r.audio_path)
        return {
            "utt_id": r.utt_id,
            "wav": wav,
            "sample_rate": sr,
            "text": r.text,
        }
