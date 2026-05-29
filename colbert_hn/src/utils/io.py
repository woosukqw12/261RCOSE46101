"""Artifact path conventions and JSON / pickle I/O.

Output layout (CLAUDE.md §5):
    outputs/{exp_name}/{dataset}/seed_{seed}/{file}

`exp_name` mirrors the `experiments/{NN}_*/` directory name so the artifact
tree is in 1:1 correspondence with the experiment tree. Example:
    outputs/00_baseline/scifact/seed_42/metrics_aggregate.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def artifact_dir(
    exp_name: str,
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    base: PathLike = "outputs",
) -> Path:
    """Resolve (and create) `outputs/{exp_name}/{dataset}/seed_{seed}/`."""
    p = Path(base) / exp_name
    if dataset is not None:
        p = p / dataset
    if seed is not None:
        p = p / f"seed_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: PathLike) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def save_pickle(obj: Any, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: PathLike) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
