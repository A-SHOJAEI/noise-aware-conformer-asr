from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from asr.config import get, load_yaml_config
from asr.data.augment import AugmentConfig, Augmenter
from asr.data.collate import collate_wav
from asr.data.features import LogMelExtractor
from asr.data.manifest_dataset import ManifestDataset
from asr.models.conformer_ctc import ConformerCTC, NoiseConditioningConfig
from asr.models.wav2vec2_ctc import W2V2Config, Wav2Vec2CTC
from asr.text.ctc_decode import greedy_ctc_decode
from asr.text.tokenizer import CharTokenizer
from asr.utils.device import resolve_device
from asr.utils.io import read_json, write_json


def _load_checkpoint(run_dir: Path) -> Dict[str, Any]:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}/checkpoints")
    # We store metadata/config alongside weights; this requires full unpickling.
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


@torch.no_grad()
def eval_manifest(
    model: torch.nn.Module,
    model_kind: str,
    cfg: Dict[str, Any],
    manifest_path: Path,
    tokenizer: CharTokenizer,
    device: torch.device,
    corruptions: List[str],
    snr_sweep: List[float],
    *,
    seed: int,
) -> Dict[str, Any]:
    ds = ManifestDataset(str(manifest_path))
    dl = DataLoader(
        ds,
        batch_size=int(get(cfg, "train.batch_size", 8)),
        shuffle=False,
        num_workers=int(get(cfg, "train.num_workers", 0)),
        collate_fn=lambda b: collate_wav(b, tokenizer, target_sr=int(get(cfg, "data.sample_rate", 16000))),
        pin_memory=(device.type == "cuda"),
    )

    aug_cfg = AugmentConfig(
        enabled=bool(get(cfg, "augment.enabled", False)),
        additive_noise=bool(get(cfg, "augment.additive_noise", False)),
        snr_db_choices=[float(x) for x in (get(cfg, "augment.snr_db_choices", []) or [])],
        reverb=bool(get(cfg, "augment.reverb", False)),
        musan_root=get(cfg, "augment.musan_root", None),
        rirs_root=get(cfg, "augment.rirs_root", None),
    )
    augmenter = Augmenter(aug_cfg, sample_rate=int(get(cfg, "data.sample_rate", 16000)), mode="eval")
    extractor = LogMelExtractor(sample_rate=int(get(cfg, "data.sample_rate", 16000))) if model_kind == "conformer_ctc" else None
    if extractor is not None:
        extractor.to(device)

    import jiwer

    results: Dict[str, Any] = {"manifest": str(manifest_path), "corruptions": {}}
    sample_rate = int(get(cfg, "data.sample_rate", 16000))
    for corr in corruptions:
        if corr == "noise_snr_sweep":
            for snr in snr_sweep:
                key = f"noise_snr_{snr:g}db"
                results["corruptions"][key] = _eval_once(
                    dl,
                    model,
                    model_kind,
                    extractor,
                    tokenizer,
                    device,
                    augmenter,
                    seed,
                    corr="noise_snr",
                    snr_db=snr,
                    sample_rate=sample_rate,
                    jiwer_mod=jiwer,
                )
        else:
            results["corruptions"][corr] = _eval_once(
                dl,
                model,
                model_kind,
                extractor,
                tokenizer,
                device,
                augmenter,
                seed,
                corr=corr,
                snr_db=None,
                sample_rate=sample_rate,
                jiwer_mod=jiwer,
            )
    return results


@torch.no_grad()
def _eval_once(
    dl: DataLoader,
    model: torch.nn.Module,
    model_kind: str,
    extractor: Optional[LogMelExtractor],
    tokenizer: CharTokenizer,
    device: torch.device,
    augmenter: Augmenter,
    seed: int,
    *,
    corr: str,
    snr_db: Optional[float],
    sample_rate: int,
    jiwer_mod,
) -> Dict[str, Any]:
    refs: List[str] = []
    hyps: List[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_sec = 0.0
    for batch in dl:
        wav = batch.wav.to(device)
        wav_lens = batch.wav_lens.to(device)

        out_wavs = []
        for i in range(wav.size(0)):
            out_wavs.append(augmenter.apply(wav[i], batch.utt_ids[i], corruption=corr, snr_db=snr_db, seed=seed))
        wav = torch.stack(out_wavs, dim=0)

        if model_kind == "conformer_ctc":
            assert extractor is not None
            feats, feat_lens = extractor(wav, wav_lens)
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            log_probs, _ = model(feats, feat_lens)
        else:
            log_probs, _ = model(wav, wav_lens)

        pred = greedy_ctc_decode(log_probs, tokenizer)
        refs.extend(batch.texts)
        hyps.extend(pred)

        n_sec += float(wav_lens.sum().item()) / float(sample_rate)

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    wer = float(jiwer_mod.wer(refs, hyps))
    rtf = (dt / n_sec) if n_sec > 0 else None
    return {"wer": wer, "rtf": rtf, "num_utts": len(refs), "wall_time_s": dt}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained run and write artifacts/results.json.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-manifest", type=str, default=None, help="Override a single test manifest path.")
    ap.add_argument(
        "--test-manifests",
        type=str,
        nargs="*",
        default=None,
        help="Evaluate multiple manifests; values are paths. If set, overrides config test_manifest.",
    )
    ap.add_argument(
        "--corruptions",
        nargs="*",
        default=None,
        help="Corruptions: clean, noise_snr_sweep, reverb_only, noise_plus_reverb",
    )
    ap.add_argument("--snr-sweep", type=float, nargs="*", default=[0, 5, 10, 15, 20])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml_config(run_dir / "config_resolved.yaml")
    meta = read_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}
    ckpt = _load_checkpoint(run_dir)

    tokenizer = CharTokenizer()
    device = resolve_device(args.device)

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
    elif model_kind == "wav2vec2_ctc":
        wcfg = W2V2Config(
            pretrained=bool(get(cfg, "model.pretrained", True)),
            variant=str(get(cfg, "model.variant", "base")),
        )
        model = Wav2Vec2CTC(vocab_size=tokenizer.vocab_size, cfg=wcfg)
    else:
        raise ValueError(f"Unknown model.kind: {model_kind}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    manifests: List[Path] = []
    if args.test_manifests is not None and len(args.test_manifests) > 0:
        manifests = [Path(p) for p in args.test_manifests]
    elif args.test_manifest:
        manifests = [Path(args.test_manifest)]
    else:
        p = Path(str(get(cfg, "data.test_manifest", "")))
        if not p.exists():
            p = Path(str(get(cfg, "data.dev_manifest")))
        manifests = [p]

    corruptions = args.corruptions if args.corruptions is not None else list(get(cfg, "eval.corruptions", ["clean"]))

    res = {
        "run_dir": str(run_dir),
        "model_kind": model_kind,
        "checkpoint": "best.pt" if (run_dir / "checkpoints" / "best.pt").exists() else "last.pt",
        "meta": meta,
        "decode": {"kind": "greedy_ctc"},
        "eval": {
            "sets": {
                m.stem: eval_manifest(
                    model,
                    model_kind,
                    cfg,
                    m,
                    tokenizer,
                    device,
                    corruptions=corruptions,
                    snr_sweep=list(args.snr_sweep),
                    seed=int(args.seed),
                )
                for m in manifests
            }
        },
    }
    write_json(args.out, res)


if __name__ == "__main__":
    main()
