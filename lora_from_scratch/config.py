"""Configuration dataclasses for LoRA experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])


@dataclass
class TrainConfig:
    """Training configuration."""

    model_name: str = "gpt2"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512
    seed: int = 42
    output_dir: str = "results/checkpoints"
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000


@dataclass
class ExperimentConfig:
    """Combined experiment configuration."""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: Optional[str] = None
