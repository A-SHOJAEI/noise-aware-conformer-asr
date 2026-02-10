Overwrote `README.md` with a project-specific writeup grounded in:

- Implemented code paths (`asr/train.py`, `asr/eval.py`, `asr/report.py`, `asr/models/conformer_ctc.py`, `asr/models/wav2vec2_ctc.py`, `asr/data/augment.py`, `asr/data/download_openslr.py`, `asr/data/make_manifests.py`)
- The exact artifact contents in `artifacts/results.json` and `artifacts/report.md` (including the smoke-run WER/RTF table and recorded git commit/seed/device/decode)

It includes problem statement, dataset provenance (OpenSLR resources as encoded in code), methodology, baselines/ablations (by config), exact results references, reproducibility commands, limitations, and concrete next research steps.