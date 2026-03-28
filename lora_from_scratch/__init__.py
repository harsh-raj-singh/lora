"""LoRA From Scratch — Low-Rank Adaptation implemented from first principles."""

from .layers import LoRALayer, LoRALinear, LoRAConv1D
from .inject import inject_lora, merge_lora, unmerge_lora, get_lora_state_dict, count_parameters
from .config import LoRAConfig, TrainConfig, ExperimentConfig
from .trainer import train, evaluate

__version__ = "1.0.0"
__all__ = [
    "LoRALayer",
    "LoRALinear",
    "LoRAConv1D",
    "inject_lora",
    "merge_lora",
    "unmerge_lora",
    "get_lora_state_dict",
    "count_parameters",
    "LoRAConfig",
    "TrainConfig",
    "ExperimentConfig",
    "train",
    "evaluate",
]
