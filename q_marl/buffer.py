from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QMARLBatch:
    observations: np.ndarray
    edge_features: np.ndarray
    adjacency: np.ndarray
    subgraph_mask: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_observations: np.ndarray
    next_edge_features: np.ndarray
    next_adjacency: np.ndarray
    next_subgraph_mask: np.ndarray


class QMARLRolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        edge_dim: int,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.edge_dim = edge_dim
        self.reset()

    def reset(self) -> None:
        shape = (self.num_steps, self.num_envs, self.num_agents)
        self.pos = 0
        self.observations = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.edge_features = np.zeros((*shape, self.num_agents, self.edge_dim), dtype=np.float32)
        self.adjacency = np.zeros((*shape, self.num_agents), dtype=np.float32)
        self.subgraph_mask = np.zeros((*shape, self.num_agents), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.bool_)
        self.next_observations = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.next_edge_features = np.zeros(
            (*shape, self.num_agents, self.edge_dim), dtype=np.float32
        )
        self.next_adjacency = np.zeros((*shape, self.num_agents), dtype=np.float32)
        self.next_subgraph_mask = np.zeros((*shape, self.num_agents), dtype=np.float32)

    def add(
        self,
        *,
        observations: np.ndarray,
        edge_features: np.ndarray,
        adjacency: np.ndarray,
        subgraph_mask: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        next_edge_features: np.ndarray,
        next_adjacency: np.ndarray,
        next_subgraph_mask: np.ndarray,
    ) -> None:
        self.observations[self.pos] = observations
        self.edge_features[self.pos] = edge_features
        self.adjacency[self.pos] = adjacency
        self.subgraph_mask[self.pos] = subgraph_mask
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.next_observations[self.pos] = next_observations
        self.next_edge_features[self.pos] = next_edge_features
        self.next_adjacency[self.pos] = next_adjacency
        self.next_subgraph_mask[self.pos] = next_subgraph_mask
        self.pos += 1

    def iterate_batches(self, num_mini_batches: int) -> list[QMARLBatch]:
        total = self.num_steps * self.num_envs
        indices = np.random.permutation(total)
        batch_size = math.ceil(total / max(1, num_mini_batches))
        batches: list[QMARLBatch] = []
        for batch_start in range(0, total, batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            t_idx = batch_idx // self.num_envs
            e_idx = batch_idx % self.num_envs
            batches.append(
                QMARLBatch(
                    observations=self.observations[t_idx, e_idx],
                    edge_features=self.edge_features[t_idx, e_idx],
                    adjacency=self.adjacency[t_idx, e_idx],
                    subgraph_mask=self.subgraph_mask[t_idx, e_idx],
                    actions=self.actions[t_idx, e_idx],
                    rewards=self.rewards[t_idx, e_idx],
                    dones=self.dones[t_idx, e_idx],
                    next_observations=self.next_observations[t_idx, e_idx],
                    next_edge_features=self.next_edge_features[t_idx, e_idx],
                    next_adjacency=self.next_adjacency[t_idx, e_idx],
                    next_subgraph_mask=self.next_subgraph_mask[t_idx, e_idx],
                )
            )
        return batches
