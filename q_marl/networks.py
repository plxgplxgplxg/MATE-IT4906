from __future__ import annotations

import torch
import torch.nn as nn

from .utils import orthogonal_init


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_relu = nn.Linear(hidden_dim, hidden_dim)
        self.edge_relu = nn.Linear(edge_dim, hidden_dim)
        self.node_out = nn.Linear(hidden_dim, hidden_dim)
        self.edge_out = nn.Linear(hidden_dim * 2 + edge_dim, edge_dim)
        for layer in (self.node_relu, self.edge_relu, self.node_out, self.edge_out):
            orthogonal_init(layer)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_msg = torch.relu(self.edge_relu(edge_features)) * adjacency.unsqueeze(-1)
        aggregated = neighbor_msg.sum(dim=3)
        node_hidden = torch.relu(self.node_relu(node_features)) + aggregated
        node_updated = (self.node_out(node_hidden) + node_features) * node_mask.unsqueeze(-1)

        src = node_updated.unsqueeze(3).expand(-1, -1, -1, node_updated.shape[2], -1)
        dst = node_updated.unsqueeze(2).expand(-1, -1, node_updated.shape[2], -1, -1)
        edge_input = torch.cat((src, dst, edge_features), dim=-1)
        edge_updated = torch.relu(self.edge_out(edge_input)) * adjacency.unsqueeze(-1)
        return node_updated, edge_updated


class QMARLNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        last_layer_gain: float,
        policy_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.policy_epsilon = policy_epsilon
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim=hidden_dim, edge_dim=edge_dim) for _ in range(num_layers)]
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.q_head = nn.Linear(hidden_dim, num_actions)

        for module in self.obs_encoder:
            orthogonal_init(module) if isinstance(module, nn.Linear) else None
        for module in self.edge_encoder:
            orthogonal_init(module) if isinstance(module, nn.Linear) else None
        orthogonal_init(self.policy_head, gain=last_layer_gain)
        orthogonal_init(self.q_head, gain=1.0)

    def _encode_subgraphs(
        self,
        observations: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        subgraph_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, _ = observations.shape
        node_mask = subgraph_mask
        root_obs = observations.unsqueeze(1).expand(-1, num_agents, -1, -1)
        root_edges = edge_features.unsqueeze(1).expand(-1, num_agents, -1, -1, -1)
        root_adj = adjacency.unsqueeze(1) * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

        node_features = self.obs_encoder(root_obs) * node_mask.unsqueeze(-1)
        edge_features = self.edge_encoder(root_edges) * root_adj.unsqueeze(-1)
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, root_adj, node_mask)
        return node_features

    def _ensemble(
        self,
        per_root_tensor: torch.Tensor,
        subgraph_mask: torch.Tensor,
    ) -> torch.Tensor:
        membership = subgraph_mask.transpose(1, 2).float()
        counts = membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bnr,brna->bna", membership, per_root_tensor) / counts

    def policy_and_q(
        self,
        observations: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        subgraph_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self._encode_subgraphs(observations, edge_features, adjacency, subgraph_mask)
        per_root_logits = self.policy_head(node_features)
        per_root_q = self.q_head(node_features)
        raw_probs = torch.softmax(per_root_logits, dim=-1)
        ensemble_raw_probs = self._ensemble(raw_probs, subgraph_mask)
        ensemble_probs = ensemble_raw_probs
        if self.policy_epsilon > 0.0:
            ensemble_probs = (1.0 - self.policy_epsilon) * ensemble_probs + (
                self.policy_epsilon / self.num_actions
            )
        ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ensemble_q = self._ensemble(per_root_q, subgraph_mask)
        return ensemble_probs, ensemble_q, ensemble_raw_probs

    def act(
        self,
        observations: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        subgraph_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, q_values, _ = self.policy_and_q(observations, edge_features, adjacency, subgraph_mask)
        dist = torch.distributions.Categorical(probs=probs)
        actions = probs.argmax(dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        baseline = (probs * q_values).sum(dim=-1)
        chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, log_probs, chosen_q, baseline

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        subgraph_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, q_values, raw_probs = self.policy_and_q(
            observations, edge_features, adjacency, subgraph_mask
        )
        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        raw_entropy = (-(raw_probs.clamp_min(1e-8) * raw_probs.clamp_min(1e-8).log()).sum(dim=-1))
        baseline = (probs * q_values).sum(dim=-1)
        chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return log_probs, entropy, chosen_q, baseline, raw_entropy, probs
