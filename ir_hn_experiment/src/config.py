"""실험 설정"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model - RLHN 논문 설정 그대로
    model_name: str = "intfloat/e5-base-unsupervised"
    max_seq_length: int = 350
    num_epochs: int = 4
    batch_size: int = 32              # RLHN과 동일
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    num_hard_negatives: int = 7       # RLHN과 동일
    temperature: float = 0.05
    seed: int = 42

    # Memory 관리
    gradient_checkpointing: bool = True
    bf16: bool = True                 # bfloat16 — scaler 불필요, RTX 5090 네이티브

    # Data
    data_dir: str = "data/bge-full-data"
    rlhn_label_dir: str = "data/rlhn-data"
    datasets: List[str] = field(default_factory=lambda: [
        "msmarco", "hotpotqa", "nq", "fever", "scidocsrr", "fiqa", "arguana",
    ])

    # Paths
    log_dir: str = "logs/training_scores"
    checkpoint_dir: str = "checkpoints"
    reencode_dir: str = "logs/reencode_scores"
    output_dir: str = "results"