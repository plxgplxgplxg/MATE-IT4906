from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

import mate
from mate.agents import GreedyTargetAgent
from mate.wrappers.discrete_action_spaces import DiscreteCamera

from .utils import build_depth_mask, radial_basis_expansion


class QMARLEnvBatch:
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        num_envs: int,
        graph_depth: int,
        edge_dim: int,
        action_levels: int,
        seed: Optional[int] = None,
    ) -> None:
        self.envs = [env_factory() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed
        self.graph_depth = graph_depth
        self.edge_dim = edge_dim
        self.action_levels = action_levels

        sample_env = self.envs[0]
        initial_obs, _ = sample_env.reset(seed=seed)
        self.num_agents = initial_obs.shape[0]
        self.obs_dim = initial_obs.shape[1]
        self.num_actions = int(action_levels * action_levels)
        self.action_high = np.asarray(
            [
                sample_env.unwrapped.camera_rotation_step,
                sample_env.unwrapped.camera_zooming_step,
            ],
            dtype=np.float32,
        )
        self.action_grid = DiscreteCamera.discrete_action_grid(levels=action_levels).astype(
            np.float32
        )
        self.view_range = float(sample_env.unwrapped.camera_max_sight_range)

    def _to_continuous_actions(self, action_indices: np.ndarray) -> np.ndarray:
        action_indices = np.asarray(action_indices, dtype=np.int64).reshape(self.num_agents)
        return self.action_high * self.action_grid[action_indices]

    def _graph_from_env(
        self,
        env: gym.Env,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_env = env.unwrapped
        positions = np.asarray([camera.location for camera in base_env.cameras], dtype=np.float32)
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        adjacency = np.asarray(base_env.camera_camera_view_mask, dtype=np.float32)
        subgraph_mask = build_depth_mask(adjacency.astype(np.bool_), self.graph_depth).astype(np.float32)
        edge_features = radial_basis_expansion(
            distances,
            n_max=self.edge_dim,
            view_range=max(self.view_range, 1e-3),
        )
        edge_features *= adjacency[..., None]
        return observations.astype(np.float32, copy=False), edge_features, adjacency, subgraph_mask

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_batch = []
        edge_batch = []
        adjacency_batch = []
        subgraph_batch = []
        for env_idx, env in enumerate(self.envs):
            if self.seed is None:
                observations, _ = env.reset()
            else:
                observations, _ = env.reset(seed=self.seed + env_idx)
            obs, edge, adjacency, subgraph = self._graph_from_env(env, observations)
            obs_batch.append(obs)
            edge_batch.append(edge)
            adjacency_batch.append(adjacency)
            subgraph_batch.append(subgraph)
        return (
            np.stack(obs_batch, axis=0),
            np.stack(edge_batch, axis=0),
            np.stack(adjacency_batch, axis=0),
            np.stack(subgraph_batch, axis=0),
        )

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
        obs_batch = []
        edge_batch = []
        adjacency_batch = []
        subgraph_batch = []
        rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        dones = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        next_episode_starts = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        infos: list[list[dict]] = []

        for env_idx, env in enumerate(self.envs):
            continuous_actions = self._to_continuous_actions(actions[env_idx])
            next_obs, reward, terminated, truncated, info = env.step(continuous_actions)
            done = bool(terminated or truncated)

            reward_array = np.asarray(reward, dtype=np.float32)
            if reward_array.ndim == 0:
                rewards[env_idx] = float(reward_array)
            else:
                rewards[env_idx] = reward_array.reshape(self.num_agents)
            dones[env_idx] = done
            infos.append(info)

            if done:
                next_obs, _ = env.reset()
                next_episode_starts[env_idx] = True

            obs, edge, adjacency, subgraph = self._graph_from_env(env, next_obs)
            obs_batch.append(obs)
            edge_batch.append(edge)
            adjacency_batch.append(adjacency)
            subgraph_batch.append(subgraph)

        return (
            np.stack(obs_batch, axis=0),
            np.stack(edge_batch, axis=0),
            np.stack(adjacency_batch, axis=0),
            np.stack(subgraph_batch, axis=0),
            rewards,
            dones,
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def default_camera_env_factory(
    env_id: str,
    env_config: str,
    render_mode: str,
    action_levels: int,
    opponent_agent_factory: Optional[Callable[[], GreedyTargetAgent]] = None,
) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        base_env = gym.make(env_id, config=env_config, render_mode=render_mode)
        return mate.MultiCamera.make(
            base_env,
            target_agent=(opponent_agent_factory or GreedyTargetAgent)(),
        )

    return _make
