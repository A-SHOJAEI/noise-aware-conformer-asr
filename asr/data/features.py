"""Audio feature extraction (log-mel spectrograms)."""
from __future__ import annotations

from typing import Tuple

import torch
import torchaudio


class LogMelExtractor(torch.nn.Module):
    """Extract log-mel filterbank features from raw waveforms."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160) -> None:
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.hop_length = hop_length

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wav: (B, T) raw waveform
            wav_lens: (B,) number of samples per utterance

        Returns:
            feats: (B, T', n_mels) log-mel features
            feat_lens: (B,) number of frames per utterance
        """
        # mel_spec expects (B, T) -> (B, n_mels, T')
        mel = self.mel_spec(wav)
        # Log with stability
        log_mel = torch.log(mel.clamp(min=1e-9))
        # Transpose to (B, T', n_mels)
        feats = log_mel.transpose(1, 2)

        # Compute feature lengths
        feat_lens = (wav_lens.float() / self.hop_length).ceil().long()
        feat_lens = torch.clamp(feat_lens, max=feats.shape[1])

        return feats, feat_lens
