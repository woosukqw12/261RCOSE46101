"""Declarative experiment configuration.

Maps the design choices in DESIGN.md §3 (architecture) and §4 (training) to
serializable dataclasses. Every config that is run becomes a row in the
ablation matrix (DESIGN.md §6); every config is persisted alongside its
artifacts so a (config, seed, dataset) triple is reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import yaml

from src.utils.io import PathLike, save_json


GateType = Literal["linear", "global", "mlp", "off"]
DirectionInit = Literal["zero", "small_random"]
InterventionForm = Literal["subtract", "add", "project_out"]
HookPosition = Literal["post_layer", "pre_ln"]
LossType = Literal["pairwise_margin", "infonce", "kd"]
Optimizer = Literal["adamw", "sgd"]
Schedule = Literal["cosine", "constant"]
HNSource = Literal["colbert_top20", "bm25_top100", "in_batch"]
ConfusedDef = Literal["top1_ne_rel", "top3_ne_rel"]


@dataclass
class SteeringConfig:
    """DESIGN.md §3."""
    enabled: bool = True
    layers: Tuple[int, ...] = (0, 3, 6, 9, 12)
    gate_type: GateType = "linear"
    direction_init: DirectionInit = "zero"
    direction_frozen: bool = False
    intervention_form: InterventionForm = "subtract"
    gate_bias_init: float = -3.0
    direction_dim: int = 1  # rank for low-rank extension (T3.01)
    share_qd: bool = True
    hook_position: HookPosition = "post_layer"
    encoder_eval_mode: bool = True


@dataclass
class TrainConfig:
    """DESIGN.md §4."""
    loss: LossType = "pairwise_margin"
    margin: float = 0.2
    lambda_anchor: float = 0.01
    lambda_gate: float = 0.001
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 5
    patience: int = 2
    warmup_steps: int = 100
    schedule: Schedule = "cosine"
    grad_clip: float = 1.0
    optimizer: Optimizer = "adamw"
    mixed_precision: bool = True
    hn_source: HNSource = "colbert_top20"


@dataclass
class EvalConfig:
    """DESIGN.md §5."""
    metrics_k: Tuple[int, ...] = (1, 3, 5, 10, 20)
    retrieval_top_k: int = 100
    confused_slice_def: ConfusedDef = "top1_ne_rel"
    bootstrap_iter: int = 10000
    bootstrap_ci: float = 0.95


@dataclass
class ExpConfig:
    config_id: str
    description: str
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    encoder_name: str = "colbert-ir/colbertv2.0"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: PathLike) -> None:
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            save_json(self.to_dict(), p)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpConfig":
        return cls(
            config_id=d["config_id"],
            description=d["description"],
            encoder_name=d.get("encoder_name", "colbert-ir/colbertv2.0"),
            steering=SteeringConfig(**d.get("steering", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**d.get("eval", {})),
        )

    @classmethod
    def load(cls, path: PathLike) -> "ExpConfig":
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            with p.open() as f:
                d = yaml.safe_load(f)
        else:
            import json
            with p.open() as f:
                d = json.load(f)
        return cls.from_dict(d)


BASELINE = ExpConfig(
    config_id="T1.00_baseline",
    description="Frozen ColBERT v2, no intervention (DESIGN.md §6.1).",
    steering=SteeringConfig(enabled=False),
)


DEFAULT_FULL_5L = ExpConfig(
    config_id="T1.02_full_5L",
    description="LSR at {0,3,6,9,12}, default config (DESIGN.md §3, §6.1).",
)


SINGLE_L6 = ExpConfig(
    config_id="T1.01_single_L6",
    description="LSR at layer 6 only (DESIGN.md §6.1).",
    steering=SteeringConfig(layers=(6,)),
)


REGISTRY: dict[str, ExpConfig] = {
    c.config_id: c for c in [BASELINE, DEFAULT_FULL_5L, SINGLE_L6]
}
