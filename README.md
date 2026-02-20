# Noise-Aware Conformer-CTC ASR

## Problem Statement
Quantify whether *explicit noise-conditioning* improves ASR robustness to additive noise and reverberation without regressing clean-set WER, relative to augmentation-only training and a wav2vec 2.0 CTC baseline.

This repo implements (and smoke-tests) that experiment end-to-end: training (`asr/train.py`), corruption-based evaluation (`asr/eval.py`), and reporting (`asr/report.py`) for a **Conformer-CTC** model with **FiLM noise-conditioning** (`asr/models/conformer_ctc.py`) plus a **wav2vec2-base CTC** baseline (`asr/models/wav2vec2_ctc.py`).

## Dataset Provenance (Code-Defined)
The data pipeline is wired for OpenSLR-hosted corpora; downloads and checksums are encoded in `asr/data/openslr_resources.py` and executed by `asr/data/download_openslr.py`.

Artifacts expected by experiment configs under `configs/experiments/*.yaml`:
- **LibriSpeech (OpenSLR resource 12)**: `train-clean-100`, `train-clean-360`, `train-other-500`, `dev-{clean,other}`, `test-{clean,other}`. Manifests are built by `asr/data/make_manifests.py` into `data/manifests/*.jsonl` (including `train-960.jsonl`).
- **MUSAN (OpenSLR resource 17)**: additive noise source directory `data/raw/musan` (used by `asr/data/augment.py`).
- **RIRS_NOISES (OpenSLR resource 28)**: impulse responses under `data/raw/RIRS_NOISES` (used by `asr/data/augment.py`).
- **Optional 4-gram LM (OpenSLR mirror resource 11)**: downloaded as `data/raw/lm/4-gram.arpa.gz` by `asr/data/download_openslr.py` (note: no LM decoder is implemented; decode is greedy CTC only).

## Methodology (Implemented)
**Text / labels**
- Transcript normalization: lowercase, hyphen-to-space, strip non `[a-z' ]`, collapse whitespace (`asr/text/normalize.py`).
- Tokenization: character CTC vocab `["<blk>", "|", "a".."z", "'"]` (`asr/text/tokenizer.py`).

**Acoustics**
- Conformer path: waveform -> 80-dim log-mel features (`asr/data/features.py`), optional SpecAugment time/freq masking during training (`asr/train.py`).
- wav2vec2 path: raw waveform into torchaudio wav2vec2-base encoder (`asr/models/wav2vec2_ctc.py`).

**Noise/reverb protocol**
- Augmentation and evaluation corruptions use deterministic per-utterance RNG seeds derived from `(seed, utt_id, corruption, snr_db)` (`asr/data/augment.py`, `asr/utils/stable_hash.py`).
- Corruptions supported by `asr/eval.py`: `clean`, `noise_snr_sweep` (expands to keys like `noise_snr_0db`, `noise_snr_5db`, ...), `reverb_only`, `noise_plus_reverb`.
- Additive noise is mixed to target SNR in dB; noise is sampled from MUSAN if present, otherwise deterministic Gaussian fallback (smoke-friendly) (`asr/data/augment.py`).
- Reverb uses 1D convolution with an IR (trimmed to 1s) from RIRS_NOISES if present; otherwise a deterministic no-op fallback (`asr/data/augment.py`).

**Noise-aware Conformer (FiLM)**
- Conditioner computes mean+std stats over time from early encoder states and maps them to an embedding (`NoiseConditioner` in `asr/models/conformer_ctc.py`).
- A FiLM module generates per-layer `(gamma, beta)` from the embedding and modulates hidden states before each Conformer block (`FiLM` + `ConformerCTC.forward` in `asr/models/conformer_ctc.py`).

## Baselines And Ablations (Config-Defined)
All experiment knobs are explicit in `configs/experiments/*.yaml`:
- Baseline: wav2vec2-base CTC fine-tuning: `configs/experiments/baseline_w2v2.yaml` (`model.kind: wav2vec2_ctc`, `model.variant: base`, `model.pretrained: true`).
- Main: noise-aware Conformer-CTC with FiLM: `configs/experiments/noise_aware_film.yaml` (`model.noise_conditioning.enabled: true`, `embed_dim: 64`).
- Ablation (remove conditioning): `configs/experiments/conformer_no_condition.yaml` (`model.noise_conditioning.enabled: false`).
- Ablation (noise-only augmentation): `configs/experiments/musan_only.yaml` (`augment.reverb: false`).
- Ablation (no SpecAugment): `configs/experiments/specaug_off.yaml` (`specaugment.enabled: false`).

## Results (From This Repo's Artifacts)
Only the **smoke** pipeline has been executed in the current repo state (`runs/smoke`), producing `artifacts/results.json` and `artifacts/report.md`.

Exact numbers:

| Run | Eval manifest | Condition | WER | RTF | #utts |
|---|---|---|---:|---:|---:|
| `runs/smoke` | `data/smoke/manifests/test.jsonl` | `clean` | 1.0000 | 0.0113 | 8 |

References:
- Machine-readable: `artifacts/results.json` at `eval.sets.test.corruptions.clean`.
- Training loss at epoch 2: 3.7, dev WER: 1.0.

Interpretation constraints: this is a synthetic sine-wave dataset (20 train, 8 dev, 8 test utterances) trained for 2 epochs (`configs/smoke.yaml`), intended to validate plumbing (training/eval/report) rather than model quality.

## Repro Instructions
### Smoke (What Produced The Current Artifacts)
```bash
make all
cat artifacts/report.md
cat artifacts/results.json
```

### Full LibriSpeech Experiments (Implemented, Not Run Here)
Setup:
```bash
make setup
.venv/bin/python -m asr.data.download_openslr --out data/raw --verify
.venv/bin/python -m asr.data.make_manifests --librispeech-root data/raw/LibriSpeech --out data/manifests
```

Train (examples; each writes `runs/<name>/{config_resolved.yaml,meta.json,checkpoints/*.pt}`):
```bash
.venv/bin/python -m asr.train --config configs/experiments/baseline_w2v2.yaml --run-dir runs/baseline_w2v2 --overwrite
.venv/bin/python -m asr.train --config configs/experiments/noise_aware_film.yaml --run-dir runs/noise_aware_film --overwrite
.venv/bin/python -m asr.train --config configs/experiments/conformer_no_condition.yaml --run-dir runs/conformer_no_condition --overwrite
.venv/bin/python -m asr.train --config configs/experiments/musan_only.yaml --run-dir runs/musan_only --overwrite
.venv/bin/python -m asr.train --config configs/experiments/specaug_off.yaml --run-dir runs/specaug_off --overwrite
```

Evaluate robustness (greedy CTC decode; writes a new results file):
```bash
.venv/bin/python -m asr.eval --run-dir runs/noise_aware_film --out artifacts/results_noise_aware_film.json \
  --test-manifest data/manifests/test-clean.jsonl \
  --corruptions clean noise_snr_sweep reverb_only noise_plus_reverb \
  --snr-sweep 0 5 10 15 20
```

Generate a report:
```bash
.venv/bin/python -m asr.report --results artifacts/results_noise_aware_film.json --out artifacts/report_noise_aware_film.md
```

CUDA note: `requirements.txt` pins CPU wheels (`torch==2.4.1`, `torchaudio==2.4.1`). If you want CUDA wheels, replace them after `make setup`, e.g.:
```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 --upgrade torch torchaudio
```

## Limitations (Current Implementation)
- Decoding is greedy CTC only (`asr/text/ctc_decode.py`); the optional 4-gram LM download is not used.
- Default experiment configs evaluate only `test-clean` (`configs/experiments/*.yaml`); `test-other` is supported but must be passed via `asr/eval.py --test-manifest`.
- No confidence intervals / multiple corruption draws: robustness is measured from a single deterministic corruption per utterance per condition (`asr/data/augment.py`).
- Training loop is intentionally minimal: no LR schedule, no DDP, no gradient accumulation, no checkpoint averaging (`asr/train.py`).

## Next Research Steps
1. Run full LibriSpeech sweeps and report (at minimum) `test-clean` and `test-other` across `clean`, SNR sweep, `reverb_only`, and `noise_plus_reverb` using identical decoding/text normalization.
2. Add a real decoder path: beam search with lexicon-free 4-gram LM (the repo already downloads `4-gram.arpa.gz`) and report both greedy and LM-decoded WER to separate acoustic vs LM effects.
3. Strengthen the conditioning study by varying where FiLM is applied (pre/post-block, subset of layers), the pooling used by the conditioner (attentive pooling vs mean/std), and `embed_dim`/regularization.
4. Robustness protocol upgrades: multiple noise segment draws per utterance (while keeping seeds fixed per draw), and confidence intervals over utterances and draws.
