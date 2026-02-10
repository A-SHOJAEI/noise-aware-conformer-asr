from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from asr.config import get, load_yaml_config, save_yaml_config
from asr.data.augment import AugmentConfig, Augmenter
from asr.data.collate import collate_wav
from asr.data.features import LogMelExtractor
from asr.data.manifest_dataset import ManifestDataset
from asr.models.conformer_ctc import ConformerCTC, NoiseConditioningConfig
from asr.models.wav2vec2_ctc import W2V2Config, Wav2Vec2CTC
from asr.text.ctc_decode import greedy_ctc_decode
from asr.text.tokenizer import CharTokenizer
from asr.utils.device import resolve_device
from asr.utils.git import git_commit_hash
from asr.utils.io import write_json
from asr.utils.repro import setup_repro


def _specaugment(cfg: Dict[str, Any]) -> nn.Module:
    if not get(cfg, "specaugment.enabled", False):
        return nn.Identity()
    time_mask_param = int(get(cfg, "specaugment.time_mask_param", 80))
    freq_mask_param = int(get(cfg, "specaugment.freq_mask_param", 27))
    num_time_masks = int(get(cfg, "specaugment.num_time_masks", 2))
    num_freq_masks = int(get(cfg, "specaugment.num_freq_masks", 2))
    tm = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    class _Aug(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B,T,F)
            y = x.transpose(1, 2)  # (B,F,T)
            for _ in range(num_time_masks):
                y = tm(y)
            for _ in range(num_freq_masks):
                y = fm(y)
            return y.transpose(1, 2)

    return _Aug()


@dataclass
class TrainState:
    step: int = 0
    best_dev_wer: float = 1e9


def _ctc_loss(log_probs: torch.Tensor, out_lens: torch.Tensor, targets: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
    # log_probs: (B,T,V) -> (T,B,V)
    lp = log_probs.transpose(0, 1).contiguous()
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    return ctc(lp, targets, out_lens, target_lens)


@torch.no_grad()
def _eval_wer(
    model: nn.Module,
    cfg: Dict[str, Any],
    ds: ManifestDataset,
    tokenizer: CharTokenizer,
    device: torch.device,
    *,
    model_kind: str,
    augment: Optional[Augmenter],
    extract_mels: Optional[LogMelExtractor],
) -> float:
    model.eval()
    dl = DataLoader(
        ds,
        batch_size=int(get(cfg, "train.batch_size", 8)),
        shuffle=False,
        num_workers=int(get(cfg, "train.num_workers", 0)),
        collate_fn=lambda b: collate_wav(b, tokenizer, target_sr=int(get(cfg, "data.sample_rate", 16000))),
    )

    refs, hyps = [], []
    for batch in dl:
        wav = batch.wav.to(device)
        wav_lens = batch.wav_lens.to(device)
        if augment is not None and get(cfg, "augment.enabled", False):
            # Evaluate clean (no corruption) here; robustness is handled in asr.eval
            out_wavs = []
            for i in range(wav.size(0)):
                out_wavs.append(augment.apply(wav[i], batch.utt_ids[i], corruption="clean", snr_db=None, seed=int(get(cfg, "seed", 0))))
            wav = torch.stack(out_wavs, dim=0)

        if model_kind == "conformer_ctc":
            assert extract_mels is not None
            feats, feat_lens = extract_mels(wav, wav_lens)
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            log_probs, out_lens = model(feats, feat_lens)
        else:
            log_probs, out_lens = model(wav, wav_lens)

        texts = batch.texts
        pred = greedy_ctc_decode(log_probs, tokenizer)
        refs.extend(texts)
        hyps.extend(pred)

    # jiwer import is lightweight but keep it local to training.
    import jiwer

    return float(jiwer.wer(refs, hyps))


def _worker_init_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        import random
        import numpy as np

        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _init


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ASR models (Conformer-CTC + wav2vec2-CTC baseline).")
    ap.add_argument("--config", type=str, required=True, help="YAML config path.")
    ap.add_argument("--set", action="append", default=None, help="Override config keys, e.g. model.d_model=128")
    ap.add_argument("--run-dir", type=str, required=True, help="Run directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory contents.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config, overrides=args.set)
    run_dir = Path(args.run_dir)
    if run_dir.exists() and args.overwrite:
        # Only delete files we create.
        for p in (run_dir / "checkpoints").glob("*.pt"):
            p.unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    setup_repro(int(get(cfg, "seed", 0)), bool(get(cfg, "deterministic", False)))
    base_seed = int(get(cfg, "seed", 0))

    device = resolve_device(str(get(cfg, "device", "auto")))
    tokenizer = CharTokenizer()

    model_kind = str(get(cfg, "model.kind"))
    if model_kind == "conformer_ctc":
        noise_cfg = NoiseConditioningConfig(
            enabled=bool(get(cfg, "model.noise_conditioning.enabled", False)),
            embed_dim=int(get(cfg, "model.noise_conditioning.embed_dim", 64)),
        )
        model = ConformerCTC(
            n_mels=80,
            vocab_size=tokenizer.vocab_size,
            d_model=int(get(cfg, "model.d_model", 256)),
            n_heads=int(get(cfg, "model.n_heads", 4)),
            num_layers=int(get(cfg, "model.num_layers", 16)),
            ff_mult=int(get(cfg, "model.ff_mult", 4)),
            conv_kernel=int(get(cfg, "model.conv_kernel", 31)),
            dropout=float(get(cfg, "model.dropout", 0.1)),
            noise_conditioning=noise_cfg,
        )
        extractor = LogMelExtractor(sample_rate=int(get(cfg, "data.sample_rate", 16000)), n_mels=80)
        specaug = _specaugment(cfg)
    elif model_kind == "wav2vec2_ctc":
        wcfg = W2V2Config(
            pretrained=bool(get(cfg, "model.pretrained", True)),
            variant=str(get(cfg, "model.variant", "base")),
        )
        model = Wav2Vec2CTC(vocab_size=tokenizer.vocab_size, cfg=wcfg)
        extractor = None
        specaug = nn.Identity()
    else:
        raise ValueError(f"Unknown model.kind: {model_kind}")

    model.to(device)
    if extractor is not None:
        extractor.to(device)

    aug_cfg = AugmentConfig(
        enabled=bool(get(cfg, "augment.enabled", False)),
        additive_noise=bool(get(cfg, "augment.additive_noise", False)),
        snr_db_choices=[float(x) for x in (get(cfg, "augment.snr_db_choices", []) or [])],
        reverb=bool(get(cfg, "augment.reverb", False)),
        musan_root=get(cfg, "augment.musan_root", None),
        rirs_root=get(cfg, "augment.rirs_root", None),
    )
    augmenter = Augmenter(aug_cfg, sample_rate=int(get(cfg, "data.sample_rate", 16000)), mode="train")

    train_ds = ManifestDataset(str(get(cfg, "data.train_manifest")))
    dev_ds = ManifestDataset(str(get(cfg, "data.dev_manifest")))

    dl = DataLoader(
        train_ds,
        batch_size=int(get(cfg, "train.batch_size", 8)),
        shuffle=True,
        num_workers=int(get(cfg, "train.num_workers", 0)),
        collate_fn=lambda b: collate_wav(b, tokenizer, target_sr=int(get(cfg, "data.sample_rate", 16000))),
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(base_seed),
        worker_init_fn=_worker_init_fn(base_seed),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(get(cfg, "train.lr", 1e-3)),
        weight_decay=float(get(cfg, "train.weight_decay", 1e-4)),
    )
    amp_enabled = bool(get(cfg, "train.amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    state = TrainState()
    epochs = int(get(cfg, "train.epochs", 1))
    log_every = int(get(cfg, "train.log_every_steps", 50))
    grad_clip = float(get(cfg, "train.grad_clip_norm", 1.0))

    save_yaml_config(run_dir / "config_resolved.yaml", cfg)
    write_json(
        run_dir / "meta.json",
        {
            "created_at_unix": time.time(),
            "git_commit": git_commit_hash("."),
            "seed": int(get(cfg, "seed", 0)),
            "deterministic": bool(get(cfg, "deterministic", False)),
            "device": str(device),
            "model_kind": model_kind,
            "tokenizer": {"kind": "char", "vocab": tokenizer.vocab},
        },
    )

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in dl:
            state.step += 1
            wav = batch.wav.to(device)
            wav_lens = batch.wav_lens.to(device)

            # Train-time augmentation: apply per-example deterministically based on step+utt_id.
            if aug_cfg.enabled:
                out_wavs = []
                for i in range(wav.size(0)):
                    # noise_snr is used for training; reverb enabled is handled inside augmenter.
                    out_wavs.append(
                        augmenter.apply(
                            wav[i],
                            batch.utt_ids[i],
                            corruption="noise_plus_reverb" if aug_cfg.reverb else "noise_snr",
                            snr_db=None,
                            seed=int(get(cfg, "seed", 0)) + state.step,
                        )
                    )
                wav = torch.stack(out_wavs, dim=0)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                if model_kind == "conformer_ctc":
                    assert extractor is not None
                    feats, feat_lens = extractor(wav, wav_lens)
                    feats = specaug(feats)
                    log_probs, out_lens = model(feats, feat_lens)
                else:
                    log_probs, out_lens = model(wav, wav_lens)
                loss = _ctc_loss(log_probs, out_lens, batch.targets.to(device), batch.target_lens.to(device))

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()

            if state.step % log_every == 0:
                dt = time.time() - t0
                print(f"epoch={ep} step={state.step} loss={loss.item():.4f} dt={dt:.1f}s")

        # Dev WER for checkpointing.
        dev_wer = _eval_wer(
            model,
            cfg,
            dev_ds,
            tokenizer,
            device,
            model_kind=model_kind,
            augment=Augmenter(aug_cfg, sample_rate=int(get(cfg, "data.sample_rate", 16000)), mode="eval"),
            extract_mels=extractor,
        )
        print(f"epoch={ep} dev_wer={dev_wer:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": state.step,
            "epoch": ep,
            "dev_wer": dev_wer,
            "config": cfg,
            "tokenizer_vocab": tokenizer.vocab,
        }
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        if dev_wer < state.best_dev_wer:
            state.best_dev_wer = dev_wer
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()
