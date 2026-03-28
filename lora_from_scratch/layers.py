"""
Core LoRA (Low-Rank Adaptation) layer implementations.

Implements the decomposed weight-update Delta W = B @ A from
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer(nn.Module):
    """Low-rank adaptation layer: Delta W = (B @ A) * (alpha / rank).

    Matrix A is initialized with Kaiming uniform; B is zero-initialized
    so the adapter starts as an identity (Delta W = 0 at init).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Low-rank matrices: A projects down, B projects back up
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout on input before low-rank projection
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        x = self.lora_dropout(x)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merge_weights(self) -> torch.Tensor:
        """Return the merged low-rank delta: (out_features, in_features)."""
        return (self.lora_B @ self.lora_A) * self.scale


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a frozen base + trainable LoRA adapter.

    output = W_frozen @ x + (B @ A) * scale @ x
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Freeze the original linear layer
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        return self.linear(x) + self.lora(x)

    def merge(self):
        """Fold LoRA weights into the base weight matrix for inference."""
        if self.merged:
            return
        delta = self.lora.merge_weights()
        self.linear.weight.data += delta
        self.merged = True

    def unmerge(self):
        """Unfold LoRA weights (reverse of merge)."""
        if not self.merged:
            return
        delta = self.lora.merge_weights()
        self.linear.weight.data -= delta
        self.merged = False

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class LoRAConv1D(nn.Module):
    """LoRA wrapper for HuggingFace-style Conv1D layers (used in GPT-2).

    Conv1D stores weights as (in_features, out_features), transposed
    relative to nn.Linear.
    """

    def __init__(
        self,
        conv1d: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.nf = conv1d.nf  # out_features
        self.nx = conv1d.weight.shape[0]  # in_features

        self.conv1d = conv1d
        for param in self.conv1d.parameters():
            param.requires_grad = False

        # LoRA matrices in nn.Linear orientation (transposed relative to Conv1D)
        self.lora = LoRALayer(
            in_features=self.nx,
            out_features=self.nf,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.conv1d(x)
        # Conv1D computes x @ weight, while LoRA computes x @ A^T @ B^T
        size_out = x.size()[:-1] + (self.nf,)
        x_reshaped = x.view(-1, self.nx)
        base_out = self.conv1d(x).view(*size_out)
        lora_out = self.lora(x_reshaped).view(*size_out)
        return base_out + lora_out
