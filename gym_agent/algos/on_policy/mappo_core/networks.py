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
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dim: int,
        fc_dim: int,
        log_std_min: float,
        log_std_max: float,
        last_layer_gain: float,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        action_low = action_low.to(dtype=torch.float32)
        action_high = action_high.to(dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._squash_epsilon = 1e-6
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

    def _scale_action(self, squashed_action: torch.Tensor) -> torch.Tensor:
        return squashed_action * self.action_scale + self.action_bias

    def _unsquash_action(self, scaled_action: torch.Tensor) -> torch.Tensor:
        normalized = (scaled_action - self.action_bias) / self.action_scale
        normalized = normalized.clamp(
            min=-1.0 + self._squash_epsilon,
            max=1.0 - self._squash_epsilon,
        )
        return torch.atanh(normalized)

    def _log_prob_from_pre_tanh(
        self,
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = distribution.log_prob(pre_tanh_action)
        squash_correction = torch.log(self.action_scale) + torch.log(
            1.0 - squashed_action.square() + self._squash_epsilon
        )
        return (log_prob - squash_correction).sum(dim=-1)

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
        pre_tanh_action = means if deterministic else distribution.rsample()
        squashed_action = torch.tanh(pre_tanh_action)
        actions = self._scale_action(squashed_action)
        log_prob = self._log_prob_from_pre_tanh(
            distribution,
            pre_tanh_action,
            squashed_action,
        )
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
        pre_tanh_action = self._unsquash_action(actions)
        squashed_action = (actions - self.action_bias) / self.action_scale
        squashed_action = squashed_action.clamp(
            min=-1.0 + self._squash_epsilon,
            max=1.0 - self._squash_epsilon,
        )
        log_prob = self._log_prob_from_pre_tanh(
            distribution,
            pre_tanh_action,
            squashed_action,
        )
        # Dung entropy Gaussian goc nhu xap xi on dinh cho entropy bonus.
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
