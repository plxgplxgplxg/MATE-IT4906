from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mate
from mate.agents import GreedyTargetAgent

from gym_agent.utils import get_device

from .buffer import MAPPORolloutBuffer
from .config import MAPPOConfig
from .env import MultiCameraEnvBatch
from .networks import RecurrentGaussianActor, RecurrentValueCritic
from .normalization import ValueNormalizer
from .state import build_state_builder
from .utils import masked_mean


class MAPPO:
    def __init__(
        self,
        config: Optional[MAPPOConfig] = None,
        env_factory: Optional[Callable[[], gym.Env]] = None,
        opponent_agent_factory: Optional[Callable[[], GreedyTargetAgent]] = None,
    ) -> None:
        # Nap cau hinh
        self.config = config or MAPPOConfig()
        # Kiem tra tham so
        self._validate_config()
        # Dat seed
        self._seed()

        # Chon thiet bi
        self.device = get_device(self.config.device)
        # Chon doi thu
        self.opponent_agent_factory = opponent_agent_factory or GreedyTargetAgent
        # Tao env factory
        self.env_factory = env_factory or self._default_env_factory
        # Tao env batch
        self.env_batch = MultiCameraEnvBatch(
            env_factory=self.env_factory,
            num_envs=self.config.num_envs,
            seed=self.config.seed,
        )

        # Doc kich thuoc
        self.num_envs = self.env_batch.num_envs
        self.num_agents = self.env_batch.num_agents
        self.obs_dim = self.env_batch.obs_dim
        self.action_dim = self.env_batch.action_dim
        # Tao state builder sau khi biet kich thuoc env
        self.state_builder = build_state_builder(
            self.config.critic_input_mode,
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
        )
        self.state_dim = self.state_builder.state_dim

        # Tao actor
        self.actor = RecurrentGaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            fc_dim=self.config.fc_dim,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
            last_layer_gain=self.config.last_layer_gain,
        ).to(self.device)
        # Tao critic
        self.critic = RecurrentValueCritic(
            state_dim=self.state_dim,
            hidden_dim=self.config.hidden_dim,
            fc_dim=self.config.fc_dim,
        ).to(self.device)
        # Tao normalizer
        self.value_normalizer = ValueNormalizer().to(self.device)

        # Tao optim actor
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.lr,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.weight_decay,
        )
        # Tao optim critic
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.lr,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.weight_decay,
        )

        # Tao rollout buffer
        self.buffer = MAPPORolloutBuffer(
            num_steps=self.config.rollout_length,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
        )

        # Reset env dau
        self.current_obs = self.env_batch.reset()
        # Hidden actor dau
        self.current_actor_hidden = np.zeros(
            (self.num_envs, self.num_agents, self.config.hidden_dim),
            dtype=np.float32,
        )
        # Hidden critic dau
        self.current_critic_hidden = np.zeros_like(self.current_actor_hidden)
        # Dau episode moi
        self.current_episode_starts = np.ones(
            (self.num_envs, self.num_agents),
            dtype=np.bool_,
        )

        # Dem so buoc
        self.total_env_steps = 0
        self.total_agent_steps = 0
        self.n_updates = 0
        self.completed_episodes = 0
        self.last_training_stats: dict[str, float] = {}

    def _validate_config(self) -> None:
        if self.config.num_envs <= 0:
            raise ValueError("config.num_envs must be greater than 0.")
        if self.config.rollout_length <= 0:
            raise ValueError("config.rollout_length must be greater than 0.")
        if self.config.recurrent_chunk_length <= 0:
            raise ValueError("config.recurrent_chunk_length must be greater than 0.")
        if self.config.n_epochs <= 0:
            raise ValueError("config.n_epochs must be greater than 0.")
        if self.config.num_mini_batches <= 0:
            raise ValueError("config.num_mini_batches must be greater than 0.")
        if self.config.critic_input_mode not in {"local", "global", "agent_specific"}:
            raise ValueError(
                "config.critic_input_mode must be one of: 'local', 'global', 'agent_specific'."
            )

    def _seed(self) -> None:
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def _default_env_factory(self) -> gym.Env:
        base_env = gym.make(
            self.config.env_id,
            config=self.config.env_config,
            render_mode=self.config.render_mode,
        )
        return mate.MultiCamera.make(
            base_env,
            target_agent=self.opponent_agent_factory(),
        )

    def _torch(
        self,
        array: np.ndarray,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(array, device=self.device)
        return tensor if dtype is None else tensor.to(dtype=dtype)

        #value normalization
    def _denormalize_values(self, raw_values: torch.Tensor) -> torch.Tensor:
        if self.config.use_value_normalization:
            return self.value_normalizer.denormalize(raw_values)
        return raw_values


    def _value_targets(self, returns: torch.Tensor) -> torch.Tensor:
        if self.config.use_value_normalization:
            return self.value_normalizer.normalize(returns)
        return returns

    def close(self) -> None:
        # Dong tat ca env
        self.env_batch.close()

    def collect_rollout(self) -> None:
        # Xoa buffer cu
        self.buffer.reset()

        # Lap rollout
        for _ in range(self.config.rollout_length):
            # Tao global state cho critic
            global_states = self.state_builder(self.current_obs)

            # Gop thanh batch
            flat_obs = self.current_obs.reshape(self.num_envs * self.num_agents, self.obs_dim)
            flat_states = global_states.reshape(
                self.num_envs * self.num_agents,
                self.state_dim,
            )
            flat_actor_hidden = self.current_actor_hidden.reshape(
                self.num_envs * self.num_agents,
                self.config.hidden_dim,
            )
            flat_critic_hidden = self.current_critic_hidden.reshape(
                self.num_envs * self.num_agents,
                self.config.hidden_dim,
            )
            flat_episode_starts = self.current_episode_starts.reshape(
                self.num_envs * self.num_agents
            )

            # Suy luan actor critic
            with torch.no_grad():
                action_tensor, log_prob_tensor, next_actor_hidden = self.actor.act(
                    self._torch(flat_obs, dtype=torch.float32),
                    self._torch(flat_actor_hidden, dtype=torch.float32),
                    self._torch(flat_episode_starts, dtype=torch.bool),
                    deterministic=False,
                )
                raw_value_tensor, next_critic_hidden = self.critic.predict_values(
                    self._torch(flat_states, dtype=torch.float32),
                    self._torch(flat_critic_hidden, dtype=torch.float32),
                    self._torch(flat_episode_starts, dtype=torch.bool),
                )
                value_estimate_tensor = self._denormalize_values(raw_value_tensor)

            # Dua ve numpy
            actions = action_tensor.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
                self.action_dim,
            )
            log_probs = log_prob_tensor.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
            )
            value_preds = raw_value_tensor.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
            )
            value_estimates = value_estimate_tensor.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
            )
            next_actor_hidden_np = next_actor_hidden.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
                self.config.hidden_dim,
            )
            next_critic_hidden_np = next_critic_hidden.cpu().numpy().reshape(
                self.num_envs,
                self.num_agents,
                self.config.hidden_dim,
            )

            # Chay mot buoc
            next_obs, rewards, dones, next_episode_starts, _ = self.env_batch.step(actions)
            # Reset hidden done
            env_done_mask = dones[:, :1].astype(np.float32)
            next_actor_hidden_np *= 1.0 - env_done_mask[..., None]
            next_critic_hidden_np *= 1.0 - env_done_mask[..., None]

            # Luu vao buffer
            self.buffer.add(
                observations=self.current_obs,
                global_states=global_states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                episode_starts=self.current_episode_starts,
                value_preds=value_preds,
                value_estimates=value_estimates,
                log_probs=log_probs,
                actor_hidden_states=self.current_actor_hidden,
                critic_hidden_states=self.current_critic_hidden,
            )

            # Dem episode xong
            self.completed_episodes += int(dones[:, 0].sum())

            # Cap nhat trang thai
            self.current_obs = next_obs
            self.current_actor_hidden = next_actor_hidden_np
            self.current_critic_hidden = next_critic_hidden_np
            self.current_episode_starts = next_episode_starts

        # Gia tri cuoi
        final_global_states = self.state_builder(self.current_obs)
        with torch.no_grad():
            final_values, _ = self.critic.predict_values(
                self._torch(
                    final_global_states.reshape(
                        self.num_envs * self.num_agents,
                        self.state_dim,
                    ),
                    dtype=torch.float32,
                ),
                self._torch(
                    self.current_critic_hidden.reshape(
                        self.num_envs * self.num_agents,
                        self.config.hidden_dim,
                    ),
                    dtype=torch.float32,
                ),
                self._torch(
                    self.current_episode_starts.reshape(self.num_envs * self.num_agents),
                    dtype=torch.bool,
                ),
            )
            final_values = self._denormalize_values(final_values)

        # Tinh return gae
        self.buffer.compute_returns_and_advantages(
            last_values=final_values.cpu().numpy().reshape(self.num_envs, self.num_agents),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if self.config.normalize_advantage:
            # Chuan hoa adv
            self.buffer.normalize_advantages()

        # Cong so buoc
        self.total_env_steps += self.num_envs * self.config.rollout_length
        self.total_agent_steps += self.num_envs * self.config.rollout_length * self.num_agents

    def update(self) -> dict[str, float]:
        if self.config.use_value_normalization:
            # Cap nhat normalizer
            self.value_normalizer.update(
                self._torch(self.buffer.returns.reshape(-1), dtype=torch.float32)
            )

        # Tao bien log
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        batch_count = 0

        # Lap theo epoch
        for _ in range(self.config.n_epochs):
            # Chia recurrent batch
            for batch in self.buffer.iterate_recurrent_batches(
                chunk_length=self.config.recurrent_chunk_length,
                num_mini_batches=self.config.num_mini_batches,
            ):
                # Dua len torch
                observations = self._torch(batch.observations, dtype=torch.float32)
                global_states = self._torch(batch.global_states, dtype=torch.float32)
                actions = self._torch(batch.actions, dtype=torch.float32)
                old_log_probs = self._torch(batch.old_log_probs, dtype=torch.float32)
                old_value_preds = self._torch(batch.old_value_preds, dtype=torch.float32)
                advantages = self._torch(batch.advantages, dtype=torch.float32)
                returns = self._torch(batch.returns, dtype=torch.float32)
                actor_hidden = self._torch(batch.actor_hidden_states, dtype=torch.float32)
                critic_hidden = self._torch(batch.critic_hidden_states, dtype=torch.float32)
                episode_starts = self._torch(batch.episode_starts, dtype=torch.bool)
                loss_mask = self._torch(batch.loss_mask, dtype=torch.float32)

                # Tinh logprob moi
                new_log_probs, entropy = self.actor.evaluate_actions(
                    observations,
                    actions,
                    actor_hidden,
                    episode_starts,
                )
                # Tinh value moi
                new_values, _ = self.critic.forward(
                    global_states,
                    critic_hidden,
                    episode_starts,
                )
                # Tinh ti le ppo
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate_1 = ratio * advantages
                surrogate_2 = ratio.clamp(
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages
                # Tinh actor loss
                actor_loss = -masked_mean(
                    torch.min(surrogate_1, surrogate_2),
                    loss_mask,
                ) - self.config.entropy_coef * masked_mean(entropy, loss_mask)

                # Tao target value
                value_targets = self._value_targets(returns)
                clipped_values = old_value_preds + (
                    new_values - old_value_preds
                ).clamp(-self.config.clip_range, self.config.clip_range)
                unclipped_value_loss = F.huber_loss(
                    new_values,
                    value_targets,
                    reduction="none",
                    delta=self.config.huber_delta,
                )
                clipped_value_loss = F.huber_loss(
                    clipped_values,
                    value_targets,
                    reduction="none",
                    delta=self.config.huber_delta,
                )
                # Tinh critic loss
                critic_loss = masked_mean(
                    torch.max(unclipped_value_loss, clipped_value_loss),
                    loss_mask,
                )

                # Gop tong loss
                total_loss = actor_loss + self.config.value_coef * critic_loss

                # Xoa grad cu
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                # Lan truyen nguoc
                total_loss.backward()
                # Cat gradient
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                # Cap nhat tham so
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Cong don thong ke
                actor_loss_total += actor_loss.item()
                critic_loss_total += critic_loss.item()
                entropy_total += masked_mean(entropy, loss_mask).item()
                batch_count += 1

        # Tang so update
        self.n_updates += 1
        # Luu log cuoi
        self.last_training_stats = {
            "actor_loss": actor_loss_total / max(batch_count, 1),
            "critic_loss": critic_loss_total / max(batch_count, 1),
            "entropy": entropy_total / max(batch_count, 1),
            "total_env_steps": float(self.total_env_steps),
            "total_agent_steps": float(self.total_agent_steps),
            "completed_episodes": float(self.completed_episodes),
        }
        return self.last_training_stats

    def learn(self, total_env_steps: int) -> dict[str, float]:
        # Lap train chinh
        while self.total_env_steps < total_env_steps:
            # Thu thap rollout
            self.collect_rollout()
            # Cap nhat model
            self.update()
        return self.last_training_stats

    def predict(
        self,
        observation: np.ndarray,
        actor_hidden_state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.num_agents, self.obs_dim):
            raise ValueError(
                f"Expected observation with shape {(self.num_agents, self.obs_dim)}, got {observation.shape}."
            )

        hidden_was_missing = actor_hidden_state is None
        if actor_hidden_state is None:
            actor_hidden_state = np.zeros(
                (self.num_agents, self.config.hidden_dim),
                dtype=np.float32,
            )
        if episode_start is None:
            episode_start = np.full(self.num_agents, hidden_was_missing, dtype=np.bool_)

        # Suy luan action
        with torch.no_grad():
            actions, _, next_hidden = self.actor.act(
                self._torch(observation, dtype=torch.float32),
                self._torch(actor_hidden_state, dtype=torch.float32),
                self._torch(episode_start, dtype=torch.bool),
                deterministic=deterministic,
            )

        return actions.cpu().numpy(), next_hidden.cpu().numpy()

    def save(self, path: str | Path) -> None:
        # Tao duong dan
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Luu checkpoint
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "value_normalizer": self.value_normalizer.state_dict(),
                "training_state": {
                    "total_env_steps": self.total_env_steps,
                    "total_agent_steps": self.total_agent_steps,
                    "n_updates": self.n_updates,
                    "completed_episodes": self.completed_episodes,
                },
            },
            save_path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env_factory: Optional[Callable[[], gym.Env]] = None,
        opponent_agent_factory: Optional[Callable[[], GreedyTargetAgent]] = None,
        map_location: str | torch.device = "cpu",
    ) -> "MAPPO":
        # Doc checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        # Tao lai agent
        agent = cls(
            config=MAPPOConfig(**checkpoint["config"]),
            env_factory=env_factory,
            opponent_agent_factory=opponent_agent_factory,
        )
        # Nap trong so
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.value_normalizer.load_state_dict(checkpoint["value_normalizer"])

        # Phuc hoi thong ke
        training_state = checkpoint.get("training_state", {})
        agent.total_env_steps = int(training_state.get("total_env_steps", 0))
        agent.total_agent_steps = int(training_state.get("total_agent_steps", 0))
        agent.n_updates = int(training_state.get("n_updates", 0))
        agent.completed_episodes = int(training_state.get("completed_episodes", 0))
        return agent
