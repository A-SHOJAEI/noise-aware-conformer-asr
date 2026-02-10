from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from asr.utils.io import read_json


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a human-readable report.md from artifacts/results.json.")
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    res = read_json(args.results)
    eval_block = res.get("eval", {}) or {}
    sets = eval_block.get("sets", None)
    if sets is None:
        # Backward-compat: older format.
        sets = {"test": eval_block}

    lines: List[str] = []
    lines.append("# ASR Evaluation Report")
    lines.append("")
    lines.append(f"- Run: `{res.get('run_dir')}`")
    lines.append(f"- Model: `{res.get('model_kind')}`")
    lines.append(f"- Checkpoint: `{res.get('checkpoint')}`")
    if res.get("decode"):
        lines.append(f"- Decode: `{(res.get('decode') or {}).get('kind')}`")
    meta = res.get("meta") or {}
    if meta:
        lines.append(f"- Seed: `{meta.get('seed')}` Deterministic: `{meta.get('deterministic')}` Device: `{meta.get('device')}`")
        if meta.get("git_commit"):
            lines.append(f"- Git commit: `{meta.get('git_commit')}`")
    lines.append("")

    lines.append("## WER Summary")
    lines.append("")
    for set_name in sorted(sets.keys()):
        block = sets[set_name] or {}
        corrs: Dict[str, Dict[str, Any]] = (block.get("corruptions", {}) or {})
        lines.append(f"### `{set_name}`")
        lines.append("")
        lines.append("| Condition | WER | RTF | #utts | wall_time_s |")
        lines.append("|---|---:|---:|---:|---:|")
        for k in sorted(corrs.keys()):
            v = corrs[k]
            lines.append(
                f"| `{k}` | {_fmt(v.get('wer'))} | {_fmt(v.get('rtf'))} | {_fmt(v.get('num_utts'))} | {_fmt(v.get('wall_time_s'))} |"
            )

        # Robustness aggregates (mean/worst over snr sweep) if present.
        snr_items = [(k, v) for k, v in corrs.items() if k.startswith("noise_snr_") and k.endswith("db")]
        if snr_items:
            wers = [float(v["wer"]) for _, v in snr_items if v.get("wer") is not None]
            if wers:
                lines.append("")
                lines.append(f"- Mean WER over noise SNR sweep: `{sum(wers)/len(wers):.4f}`")
                lines.append(f"- Worst-case WER over noise SNR sweep: `{max(wers):.4f}`")
                lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
