from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecurrentBatch:
    observations: np.ndarray
    global_states: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_value_preds: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    actor_hidden_states: np.ndarray
    critic_hidden_states: np.ndarray
    episode_starts: np.ndarray
    loss_mask: np.ndarray


class MAPPORolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self) -> None:
        shape_ea = (self.num_steps, self.num_envs, self.num_agents)
        self.pos = 0
        self.observations = np.zeros((*shape_ea, self.obs_dim), dtype=np.float32)
        self.global_states = np.zeros((*shape_ea, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((*shape_ea, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(shape_ea, dtype=np.float32)
        self.dones = np.zeros(shape_ea, dtype=np.bool_)
        self.episode_starts = np.zeros(shape_ea, dtype=np.bool_)
        self.value_preds = np.zeros(shape_ea, dtype=np.float32)
        self.value_estimates = np.zeros(shape_ea, dtype=np.float32)
        self.log_probs = np.zeros(shape_ea, dtype=np.float32)
        self.advantages = np.zeros(shape_ea, dtype=np.float32)
        self.returns = np.zeros(shape_ea, dtype=np.float32)
        self.actor_hidden_states = np.zeros(
            (*shape_ea, self.hidden_dim),
            dtype=np.float32,
        )
        self.critic_hidden_states = np.zeros(
            (*shape_ea, self.hidden_dim),
            dtype=np.float32,
        )

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        value_preds: np.ndarray,
        value_estimates: np.ndarray,
        log_probs: np.ndarray,
        actor_hidden_states: np.ndarray,
        critic_hidden_states: np.ndarray,
    ) -> None:
        if self.pos >= self.num_steps:
            raise ValueError("Rollout buffer is full. Call reset() before reusing it.")

        self.observations[self.pos] = observations
        self.global_states[self.pos] = global_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.episode_starts[self.pos] = episode_starts
        self.value_preds[self.pos] = value_preds
        self.value_estimates[self.pos] = value_estimates
        self.log_probs[self.pos] = log_probs
        self.actor_hidden_states[self.pos] = actor_hidden_states
        self.critic_hidden_states[self.pos] = critic_hidden_states
        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_advantage = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.value_estimates[step + 1]

            next_non_terminal = 1.0 - self.dones[step].astype(np.float32)
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.value_estimates[step]
            )
            last_advantage = (
                delta
                + gamma * gae_lambda * next_non_terminal * last_advantage
            )
            self.advantages[step] = last_advantage

        self.returns = self.advantages + self.value_estimates

    def normalize_advantages(self) -> None:
        flat_advantages = self.advantages.reshape(-1)
        mean = flat_advantages.mean()
        std = flat_advantages.std()
        self.advantages = (self.advantages - mean) / (std + 1e-8)

    def iterate_recurrent_batches(
        self,
        chunk_length: int,
        num_mini_batches: int,
    ) -> list[RecurrentBatch]:
        chunk_starts = list(range(0, self.num_steps, chunk_length))
        sequence_indices = [
            (env_idx, agent_idx, chunk_start)
            for env_idx in range(self.num_envs)
            for agent_idx in range(self.num_agents)
            for chunk_start in chunk_starts
        ]

        permutation = np.random.permutation(len(sequence_indices))
        num_batches = max(1, min(num_mini_batches, len(sequence_indices)))
        batch_size = math.ceil(len(sequence_indices) / num_batches)
        batches: list[RecurrentBatch] = []

        for batch_start in range(0, len(sequence_indices), batch_size):
            current = permutation[batch_start : batch_start + batch_size]
            current_batch_size = len(current)

            observations = np.zeros(
                (current_batch_size, chunk_length, self.obs_dim),
                dtype=np.float32,
            )
            global_states = np.zeros(
                (current_batch_size, chunk_length, self.state_dim),
                dtype=np.float32,
            )
            actions = np.zeros(
                (current_batch_size, chunk_length, self.action_dim),
                dtype=np.float32,
            )
            old_log_probs = np.zeros((current_batch_size, chunk_length), dtype=np.float32)
            old_value_preds = np.zeros((current_batch_size, chunk_length), dtype=np.float32)
            advantages = np.zeros((current_batch_size, chunk_length), dtype=np.float32)
            returns = np.zeros((current_batch_size, chunk_length), dtype=np.float32)
            actor_hidden_states = np.zeros(
                (current_batch_size, self.hidden_dim),
                dtype=np.float32,
            )
            critic_hidden_states = np.zeros(
                (current_batch_size, self.hidden_dim),
                dtype=np.float32,
            )
            episode_starts = np.zeros((current_batch_size, chunk_length), dtype=np.bool_)
            loss_mask = np.zeros((current_batch_size, chunk_length), dtype=np.float32)

            for batch_idx, seq_idx in enumerate(current):
                env_idx, agent_idx, chunk_start = sequence_indices[seq_idx]
                chunk_end = min(chunk_start + chunk_length, self.num_steps)
                valid_length = chunk_end - chunk_start

                observations[batch_idx, :valid_length] = self.observations[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                global_states[batch_idx, :valid_length] = self.global_states[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                actions[batch_idx, :valid_length] = self.actions[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                old_log_probs[batch_idx, :valid_length] = self.log_probs[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                old_value_preds[batch_idx, :valid_length] = self.value_preds[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                advantages[batch_idx, :valid_length] = self.advantages[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                returns[batch_idx, :valid_length] = self.returns[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                episode_starts[batch_idx, :valid_length] = self.episode_starts[
                    chunk_start:chunk_end, env_idx, agent_idx
                ]
                actor_hidden_states[batch_idx] = self.actor_hidden_states[
                    chunk_start, env_idx, agent_idx
                ]
                critic_hidden_states[batch_idx] = self.critic_hidden_states[
                    chunk_start, env_idx, agent_idx
                ]
                loss_mask[batch_idx, :valid_length] = 1.0

            batches.append(
                RecurrentBatch(
                    observations=observations,
                    global_states=global_states,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    old_value_preds=old_value_preds,
                    advantages=advantages,
                    returns=returns,
                    actor_hidden_states=actor_hidden_states,
                    critic_hidden_states=critic_hidden_states,
                    episode_starts=episode_starts,
                    loss_mask=loss_mask,
                )
            )

        return batches
