"""Reproducibility utilities — seed control and device selection.

CLAUDE.md §3.7 (statistical robustness) + §8 (reproducibility checklist).
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNG seeds (Python, NumPy, PyTorch) for a single run.

    `deterministic=True` enforces cuDNN determinism — required for the
    paired-bootstrap CIs in DESIGN.md §5.3 to be reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Resolve a torch device. CUDA > MPS > CPU unless `prefer` is given."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SEEDS: tuple[int, ...] = (42, 1337, 2024)
"""Canonical seed triple per CLAUDE.md §3.7 / §8."""
