from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ReproSettings:
    seed: int
    deterministic: bool


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool) -> None:
    # Determinism comes with performance cost; keep configurable.
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may still be non-deterministic depending on build/hardware.
            pass


def setup_repro(seed: int, deterministic: bool) -> ReproSettings:
    seed_all(seed)
    configure_determinism(deterministic)
    return ReproSettings(seed=seed, deterministic=deterministic)

