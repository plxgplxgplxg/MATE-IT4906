from __future__ import annotations

import numpy as np


class AgentSpecificStateBuilder:
    """Build centralized critic inputs using [concat(all_obs), local_obs_i]."""

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
