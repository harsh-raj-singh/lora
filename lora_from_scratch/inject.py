"""Utilities for injecting LoRA adapters into transformer models."""

import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .layers import LoRALinear, LoRAConv1D


def inject_lora(
    model: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = True,
) -> nn.Module:
    """Inject LoRA adapters into a model's Linear and Conv1D layers.

    Args:
        model: The transformer model to adapt.
        target_modules: Substring matches for layer names to inject into.
            Defaults to ["c_attn", "c_proj"] (GPT-2 attention layers).
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout: Dropout probability on LoRA input.
        verbose: Print injection log.

    Returns:
        The model with LoRA layers injected (in-place).
    """
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]

    count = 0

    def _replace_recursive(module: nn.Module, prefix: str = ""):
        nonlocal count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                if any(t in name for t in target_modules):
                    wrapper = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, wrapper)
                    count += 1
                    if verbose:
                        print(
                            f"  [LoRA] {full_name:60s} "
                            f"{child.weight.shape} -> rank={rank}"
                        )
                continue

            if isinstance(child, Conv1D):
                if any(t in name for t in target_modules):
                    wrapper = LoRAConv1D(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, wrapper)
                    count += 1
                    if verbose:
                        print(
                            f"  [LoRA] {full_name:60s} "
                            f"{child.weight.shape} -> rank={rank}"
                        )
                continue

            _replace_recursive(child, full_name)

    _replace_recursive(model)
    if verbose:
        print(f"\nInjected {count} LoRA adapter(s) (rank={rank}, alpha={alpha})")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base weights for inference."""
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1D)):
            module.merge()
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Unmerge all LoRA weights (restore separate adapters)."""
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1D)):
            module.unmerge()
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only the LoRA parameters (for checkpointing)."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_" in k
    }


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and LoRA-specific parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora_params": lora,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
    }
