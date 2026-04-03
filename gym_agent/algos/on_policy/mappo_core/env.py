from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiCameraEnvBatch:
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        num_envs: int,
        seed: Optional[int] = None,
    ) -> None:
        self.envs = [env_factory() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed

        sample_env = self.envs[0]
        initial_obs, _ = sample_env.reset(seed=seed)
        self.num_agents = initial_obs.shape[0]
        self.obs_dim = initial_obs.shape[1]
        camera_action_space = sample_env.unwrapped.camera_action_space
        if not isinstance(camera_action_space, spaces.Box):
            raise ValueError(
                f"MAPPO expects continuous camera actions, got {camera_action_space}."
            )
        self.action_dim = int(np.prod(camera_action_space.shape))
        self.action_low = np.asarray(camera_action_space.low, dtype=np.float32).reshape(
            self.action_dim
        )
        self.action_high = np.asarray(camera_action_space.high, dtype=np.float32).reshape(
            self.action_dim
        )

    def reset(self) -> np.ndarray:
        observations = []
        for env_idx, env in enumerate(self.envs):
            if self.seed is None:
                obs, _ = env.reset()
            else:
                obs, _ = env.reset(seed=self.seed + env_idx)
            observations.append(obs.astype(np.float32, copy=False))
        return np.stack(observations, axis=0)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
        next_observations = []
        rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        dones = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        next_episode_starts = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        infos: list[list[dict]] = []

        for env_idx, env in enumerate(self.envs):
            next_obs, reward, terminated, truncated, info = env.step(actions[env_idx])
            done = bool(terminated or truncated)

            if np.isscalar(reward):
                rewards[env_idx] = float(reward)
            else:
                reward_array = np.asarray(reward, dtype=np.float32).reshape(self.num_agents)
                rewards[env_idx] = reward_array

            dones[env_idx] = done
            infos.append(info)

            if done:
                next_obs, _ = env.reset()
                next_episode_starts[env_idx] = True

            next_observations.append(next_obs.astype(np.float32, copy=False))

        return (
            np.stack(next_observations, axis=0),
            rewards,
            dones,
            next_episode_starts,
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
