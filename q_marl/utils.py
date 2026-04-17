from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn


def get_device(device: torch.device | str = "auto") -> torch.device:
    if device == "auto":
        device = "cuda"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2.0)) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def radial_basis_expansion(
    distances: np.ndarray,
    *,
    n_max: int,
    view_range: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    delta_d = max(view_range / max(n_max, 1), 1e-6)
    n_vals = np.arange(n_max, dtype=np.float32)
    return np.exp(-((distances[..., None] - n_vals * delta_d) ** 2) / delta_d).astype(
        np.float32
    )


def build_depth_mask(adjacency: np.ndarray, depth: int) -> np.ndarray:
    num_agents = adjacency.shape[0]
    mask = np.zeros((num_agents, num_agents), dtype=np.bool_)
    for root in range(num_agents):
        visited = np.zeros(num_agents, dtype=np.bool_)
        queue: deque[tuple[int, int]] = deque([(root, 0)])
        visited[root] = True
        while queue:
            node, dist = queue.popleft()
            mask[root, node] = True
            if dist >= depth:
                continue
            for nxt in np.flatnonzero(adjacency[node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append((int(nxt), dist + 1))
    return mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)
