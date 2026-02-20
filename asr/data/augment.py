"""Audio augmentation for noise-aware training."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class AugmentConfig:
    enabled: bool = False
    additive_noise: bool = False
    snr_db_choices: List[float] = field(default_factory=list)
    reverb: bool = False
    musan_root: Optional[str] = None
    rirs_root: Optional[str] = None


class Augmenter:
    """Apply audio augmentations (noise, reverb) for training.

    In smoke mode with no external noise corpora, falls back to
    synthetic Gaussian noise at specified SNR levels.
    """

    def __init__(self, cfg: AugmentConfig, sample_rate: int = 16000, mode: str = "train") -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.mode = mode

    def apply(
        self,
        wav: torch.Tensor,
        utt_id: str,
        corruption: str = "clean",
        snr_db: Optional[float] = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Apply augmentation to a single waveform.

        Args:
            wav: (T,) waveform tensor
            utt_id: utterance ID for deterministic augmentation
            corruption: type of corruption ("clean", "noise_snr", "noise_plus_reverb")
            snr_db: specific SNR; if None, randomly sample from cfg.snr_db_choices
            seed: random seed for deterministic augmentation
        """
        if not self.cfg.enabled or corruption == "clean":
            return wav

        rng = random.Random(seed + hash(utt_id))

        if self.cfg.additive_noise and corruption in ("noise_snr", "noise_plus_reverb"):
            if snr_db is None and self.cfg.snr_db_choices:
                snr_db = rng.choice(self.cfg.snr_db_choices)
            if snr_db is not None:
                wav = self._add_gaussian_noise(wav, snr_db, rng)

        return wav

    @staticmethod
    def _add_gaussian_noise(wav: torch.Tensor, snr_db: float, rng: random.Random) -> torch.Tensor:
        """Add Gaussian noise at specified SNR."""
        signal_power = wav.pow(2).mean()
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-9)
        noise_std = noise_power.sqrt()

        # Use torch generator for reproducibility
        gen = torch.Generator(device=wav.device)
        gen.manual_seed(rng.randint(0, 2**31))
        noise = torch.randn(wav.shape, generator=gen, device=wav.device, dtype=wav.dtype) * noise_std
        return wav + noise

    def get_snr_db(self, wav: torch.Tensor, utt_id: str, seed: int = 0) -> Optional[float]:
        """Report the SNR that would be applied (for noise-conditioning)."""
        if not self.cfg.enabled or not self.cfg.snr_db_choices:
            return None
        rng = random.Random(seed + hash(utt_id))
        return rng.choice(self.cfg.snr_db_choices)
