from __future__ import annotations

import torch
import torch.nn as nn


class ValueNormalizer(nn.Module):
    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("var", torch.ones((), dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp_min(self.epsilon))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1).float()
        if values.numel() == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = torch.tensor(float(values.numel()), device=values.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(self.epsilon))
        self.count.copy_(total_count)
