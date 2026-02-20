"""Generate synthetic smoke ASR data (short sine-wave utterances with text labels)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio


_WORDS = [
    "hello", "world", "the", "cat", "sat", "on", "mat",
    "a", "dog", "ran", "in", "park", "yes", "no", "stop",
    "go", "left", "right", "up", "down", "one", "two", "three",
]

SAMPLE_RATE = 16000


def _generate_utterance(rng: random.Random, min_words: int = 2, max_words: int = 5) -> str:
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _text_to_sine(text: str, sr: int = SAMPLE_RATE, dur_per_char: float = 0.02) -> np.ndarray:
    """Generate a deterministic sine waveform whose length depends on text length."""
    n_chars = max(len(text), 1)
    duration = n_chars * dur_per_char + 0.1  # minimum 0.1s
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # Use character values as frequency modulation for variety
    freq = 200 + sum(ord(c) for c in text[:10]) % 300
    wav = 0.3 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    return wav


def generate_smoke_data(out_dir: str | Path, seed: int = 1337, n_train: int = 20, n_dev: int = 8, n_test: int = 8) -> None:
    """Generate synthetic audio + manifests for smoke testing."""
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    manifest_dir = out_dir / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    splits = {"train": n_train, "dev": n_dev, "test": n_test}
    for split_name, n in splits.items():
        manifest_rows: List[dict] = []
        for i in range(n):
            utt_id = f"{split_name}_{i:04d}"
            text = _generate_utterance(rng)
            wav_np = _text_to_sine(text)
            wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)

            wav_path = audio_dir / f"{utt_id}.wav"
            torchaudio.save(str(wav_path), wav_tensor, SAMPLE_RATE)

            manifest_rows.append({
                "utt_id": utt_id,
                "audio_path": str(wav_path),
                "text": text,
                "duration": len(wav_np) / SAMPLE_RATE,
            })

        manifest_path = manifest_dir / f"{split_name}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

        print(f"  {split_name}: {n} utterances -> {manifest_path}")

    print(f"Smoke data generated in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/smoke")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    generate_smoke_data(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
