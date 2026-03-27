from __future__ import annotations

import math

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


def orthogonal_init_recurrent(cell: nn.GRUCell, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(cell.weight_ih, gain=gain)
    nn.init.orthogonal_(cell.weight_hh, gain=gain)
    nn.init.constant_(cell.bias_ih, 0.0)
    nn.init.constant_(cell.bias_hh, 0.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = values * mask
    return masked_values.sum() / mask.sum().clamp_min(1.0)
