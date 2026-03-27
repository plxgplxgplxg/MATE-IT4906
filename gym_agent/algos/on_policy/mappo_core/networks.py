from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import orthogonal_init, orthogonal_init_recurrent


class RecurrentBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, fc_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, fc_dim)
        self.gru = nn.GRUCell(fc_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        orthogonal_init(self.fc_in)
        orthogonal_init(self.fc_out)
        orthogonal_init_recurrent(self.gru)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected recurrent input with shape (batch, seq_len, input_dim), got {inputs.shape}."
            )

        outputs = []
        hidden = hidden_state
        for step in range(inputs.shape[1]):
            start_mask = (1.0 - episode_starts[:, step].float()).unsqueeze(-1)
            hidden = hidden * start_mask
            x = self.input_norm(inputs[:, step])
            x = F.relu(self.fc_in(x))
            hidden = self.gru(x, hidden)
            outputs.append(F.relu(self.fc_out(hidden)))

        return torch.stack(outputs, dim=1), hidden


class RecurrentGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        fc_dim: int,
        log_std_min: float,
        log_std_max: float,
        last_layer_gain: float,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.backbone = RecurrentBackbone(obs_dim, hidden_dim, fc_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        orthogonal_init(self.mean_head, gain=last_layer_gain)
        orthogonal_init(self.log_std_head, gain=last_layer_gain)

    def forward(
        self,
        observations: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, next_hidden = self.backbone(observations, hidden_state, episode_starts)
        means = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return means, log_std, next_hidden

    def act(
        self,
        observations: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = observations.unsqueeze(1)
        episode_starts = episode_starts.unsqueeze(1)
        means, log_std, next_hidden = self.forward(
            observations,
            hidden_state,
            episode_starts,
        )
        means = means[:, 0]
        log_std = log_std[:, 0]
        distribution = Normal(means, log_std.exp())
        actions = means if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        return actions, log_prob, next_hidden

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, log_std, _ = self.forward(observations, hidden_state, episode_starts)
        distribution = Normal(means, log_std.exp())
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy


class RecurrentValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, fc_dim: int) -> None:
        super().__init__()
        self.backbone = RecurrentBackbone(state_dim, hidden_dim, fc_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.value_head, gain=1.0)

    def forward(
        self,
        states: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, next_hidden = self.backbone(states, hidden_state, episode_starts)
        return self.value_head(features).squeeze(-1), next_hidden

    def predict_values(
        self,
        states: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = states.unsqueeze(1)
        episode_starts = episode_starts.unsqueeze(1)
        values, next_hidden = self.forward(states, hidden_state, episode_starts)
        return values[:, 0], next_hidden
