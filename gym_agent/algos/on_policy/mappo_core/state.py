from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class StateBuilder(Protocol):
    state_dim: int

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class LocalStateBuilder:
    obs_dim: int
    num_agents: int

    @property
    def state_dim(self) -> int:
        return self.obs_dim

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim != 3:
            raise ValueError(
                f"Expected observations with shape (num_envs, num_agents, obs_dim), got {observations.shape}."
            )
        return observations.astype(np.float32, copy=False)


@dataclass(frozen=True)
class GlobalStateBuilder:
    obs_dim: int
    num_agents: int

    @property
    def state_dim(self) -> int:
        return self.num_agents * self.obs_dim

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim != 3:
            raise ValueError(
                f"Expected observations with shape (num_envs, num_agents, obs_dim), got {observations.shape}."
            )

        flat_obs = observations.reshape(observations.shape[0], -1)
        return np.repeat(flat_obs[:, None, :], observations.shape[1], axis=1).astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class AgentSpecificStateBuilder:
    obs_dim: int
    num_agents: int

    @property
    def state_dim(self) -> int:
        return self.num_agents * self.obs_dim + self.obs_dim

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim != 3:
            raise ValueError(
                f"Expected observations with shape (num_envs, num_agents, obs_dim), got {observations.shape}."
            )

        flat_obs = observations.reshape(observations.shape[0], -1)
        repeated_global = np.repeat(flat_obs[:, None, :], observations.shape[1], axis=1)
        return np.concatenate([repeated_global, observations], axis=-1).astype(
            np.float32,
            copy=False,
        )


def build_state_builder(
    mode: str,
    *,
    obs_dim: int,
    num_agents: int,
) -> StateBuilder:
    if mode == "local":
        return LocalStateBuilder(obs_dim=obs_dim, num_agents=num_agents)
    if mode == "global":
        return GlobalStateBuilder(obs_dim=obs_dim, num_agents=num_agents)
    if mode == "agent_specific":
        return AgentSpecificStateBuilder(obs_dim=obs_dim, num_agents=num_agents)
    raise ValueError(
        "Unsupported critic_input_mode. Expected one of: 'local', 'global', 'agent_specific'."
    )
