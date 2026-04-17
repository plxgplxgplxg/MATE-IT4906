from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mate.agents import GreedyTargetAgent

from .buffer import QMARLRolloutBuffer
from .config import QMARLConfig
from .env import QMARLEnvBatch, default_camera_env_factory
from .networks import QMARLNet
from .utils import get_device


class QMARL:
    def __init__(
        self,
        config: Optional[QMARLConfig] = None,
        env_factory: Optional[Callable[[], gym.Env]] = None,
        opponent_agent_factory: Optional[Callable[[], GreedyTargetAgent]] = None,
    ) -> None:
        self.config = config or QMARLConfig()
        self.device = get_device(self.config.device)
        self._seed()

        self.env_factory = env_factory or default_camera_env_factory(
            env_id=self.config.env_id,
            env_config=self.config.env_config,
            render_mode=self.config.render_mode,
            action_levels=self.config.action_levels,
            opponent_agent_factory=opponent_agent_factory,
        )
        self.env_batch = QMARLEnvBatch(
            env_factory=self.env_factory,
            num_envs=self.config.num_envs,
            graph_depth=self.config.graph_depth,
            edge_dim=self.config.edge_dim,
            action_levels=self.config.action_levels,
            seed=self.config.seed,
        )

        self.num_envs = self.env_batch.num_envs
        self.num_agents = self.env_batch.num_agents
        self.obs_dim = self.env_batch.obs_dim
        self.num_actions = self.env_batch.num_actions

        self.network = QMARLNet(
            obs_dim=self.obs_dim,
            num_actions=self.num_actions,
            hidden_dim=self.config.hidden_dim,
            edge_dim=self.config.edge_dim,
            num_layers=self.config.num_message_passing_layers,
            last_layer_gain=self.config.last_layer_gain,
            policy_epsilon=self.config.policy_epsilon,
        ).to(self.device)
        actor_params = list(self.network.obs_encoder.parameters()) + list(
            self.network.edge_encoder.parameters()
        ) + list(self.network.layers.parameters()) + list(self.network.policy_head.parameters())
        critic_params = list(self.network.q_head.parameters())
        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=self.config.actor_lr,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = None
        if self.config.use_reduce_on_plateau:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode="max",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
            )

        self.buffer = QMARLRolloutBuffer(
            num_steps=self.config.rollout_length,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            edge_dim=self.config.edge_dim,
        )

        (
            self.current_obs,
            self.current_edge_features,
            self.current_adjacency,
            self.current_subgraph_mask,
        ) = self.env_batch.reset()

        self.total_env_steps = 0
        self.total_agent_steps = 0
        self.n_updates = 0
        self.completed_episodes = 0
        self.last_training_stats: dict[str, float] = {}

    def _seed(self) -> None:
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def _torch_float(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _torch_long(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.long, device=self.device)

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_rewards:
            return rewards * self.config.reward_scale
        scaled = rewards * self.config.reward_scale
        mean = scaled.mean()
        std = scaled.std(unbiased=False).clamp_min(1e-6)
        return (scaled - mean) / std

    def _critic_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.critic_loss == "huber":
            return F.huber_loss(
                prediction,
                target,
                delta=self.config.critic_huber_delta,
            )
        if self.config.critic_loss == "mse":
            return F.mse_loss(prediction, target)
        raise ValueError(f"Unsupported critic_loss: {self.config.critic_loss}")

    def close(self) -> None:
        self.env_batch.close()

    def collect_rollout(self) -> None:
        self.buffer.reset()
        for _ in range(self.config.rollout_length):
            with torch.no_grad():
                actions, _, _, _ = self.network.act(
                    self._torch_float(self.current_obs),
                    self._torch_float(self.current_edge_features),
                    self._torch_float(self.current_adjacency),
                    self._torch_float(self.current_subgraph_mask),
                    deterministic=False,
                )

            action_np = actions.cpu().numpy()
            (
                next_obs,
                next_edge_features,
                next_adjacency,
                next_subgraph_mask,
                rewards,
                dones,
                infos,
            ) = self.env_batch.step(action_np)

            self.buffer.add(
                observations=self.current_obs,
                edge_features=self.current_edge_features,
                adjacency=self.current_adjacency,
                subgraph_mask=self.current_subgraph_mask,
                actions=action_np,
                rewards=rewards,
                dones=dones,
                next_observations=next_obs,
                next_edge_features=next_edge_features,
                next_adjacency=next_adjacency,
                next_subgraph_mask=next_subgraph_mask,
            )

            self.completed_episodes += int(dones[:, 0].sum())
            self.current_obs = next_obs
            self.current_edge_features = next_edge_features
            self.current_adjacency = next_adjacency
            self.current_subgraph_mask = next_subgraph_mask

        self.total_env_steps += self.num_envs * self.config.rollout_length
        self.total_agent_steps += self.num_envs * self.config.rollout_length * self.num_agents

    def update(self) -> dict[str, float]:
        actor_loss_total = 0.0
        actor_total_loss = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        delta_total = 0.0
        baseline_total = 0.0
        q_total = 0.0
        advantage_abs_total = 0.0
        q_minus_baseline_abs_total = 0.0
        policy_max_prob_total = 0.0
        raw_entropy_total = 0.0
        batch_count = 0

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.iterate_batches(self.config.num_mini_batches):
                observations = self._torch_float(batch.observations)
                edge_features = self._torch_float(batch.edge_features)
                adjacency = self._torch_float(batch.adjacency)
                subgraph_mask = self._torch_float(batch.subgraph_mask)
                actions = self._torch_long(batch.actions)
                rewards = self._normalize_rewards(self._torch_float(batch.rewards))
                dones = self._torch_float(batch.dones.astype(np.float32))
                next_observations = self._torch_float(batch.next_observations)
                next_edge_features = self._torch_float(batch.next_edge_features)
                next_adjacency = self._torch_float(batch.next_adjacency)
                next_subgraph_mask = self._torch_float(batch.next_subgraph_mask)

                log_probs, entropy, chosen_q, baseline, raw_entropy, probs = self.network.evaluate_actions(
                    observations,
                    edge_features,
                    adjacency,
                    subgraph_mask,
                    actions,
                )
                with torch.no_grad():
                    _, _, _, next_baseline = self.network.act(
                        next_observations,
                        next_edge_features,
                        next_adjacency,
                        next_subgraph_mask,
                        deterministic=True,
                    )

                target = rewards + self.config.gamma * (1.0 - dones) * next_baseline

                for _ in range(max(0, self.config.critic_extra_steps)):
                    _, _, chosen_q_extra, baseline_extra, _, _ = self.network.evaluate_actions(
                        observations,
                        edge_features,
                        adjacency,
                        subgraph_mask,
                        actions,
                    )
                    critic_loss_extra = self._critic_loss(chosen_q_extra, target)
                    baseline_reg = self._critic_loss(baseline_extra, target)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coef * (critic_loss_extra + 0.5 * baseline_reg)).backward()
                    nn.utils.clip_grad_norm_(self.network.q_head.parameters(), self.config.max_grad_norm)
                    self.critic_optimizer.step()

                log_probs, entropy, chosen_q, baseline, raw_entropy, probs = self.network.evaluate_actions(
                    observations,
                    edge_features,
                    adjacency,
                    subgraph_mask,
                    actions,
                )
                delta = target - baseline
                advantage = delta.detach()
                actor_pg_loss = -(advantage * log_probs).mean()
                entropy_bonus = entropy.mean()
                actor_loss = actor_pg_loss - self.config.entropy_coef * entropy_bonus
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.network.policy_head.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.network.obs_encoder.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.network.edge_encoder.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.network.layers.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                _, _, chosen_q_critic, _, _, _ = self.network.evaluate_actions(
                    observations,
                    edge_features,
                    adjacency,
                    subgraph_mask,
                    actions,
                )
                critic_loss = self._critic_loss(chosen_q_critic, target)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.network.q_head.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                actor_loss_total += float(actor_pg_loss.item())
                actor_total_loss += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_bonus.item())
                delta_total += float(delta.mean().item())
                baseline_total += float(baseline.mean().item())
                q_total += float(chosen_q.mean().item())
                advantage_abs_total += float(advantage.abs().mean().item())
                q_minus_baseline_abs_total += float((chosen_q - baseline).abs().mean().item())
                policy_max_prob_total += float(probs.max(dim=-1).values.mean().item())
                raw_entropy_total += float(raw_entropy.mean().item())
                batch_count += 1

        mean_delta = delta_total / max(batch_count, 1)
        if self.scheduler is not None:
            self.scheduler.step(mean_delta)

        self.n_updates += 1
        self.last_training_stats = {
            "actor_loss": actor_loss_total / max(1, batch_count),
            "actor_total_loss": actor_total_loss / max(1, batch_count),
            "critic_loss": critic_loss_total / max(1, batch_count),
            "entropy": entropy_total / max(1, batch_count),
            "mean_delta": mean_delta,
            "baseline_mean": baseline_total / max(1, batch_count),
            "q_mean": q_total / max(1, batch_count),
            "advantage_abs_mean": advantage_abs_total / max(1, batch_count),
            "q_minus_baseline_abs_mean": q_minus_baseline_abs_total / max(1, batch_count),
            "policy_max_prob_mean": policy_max_prob_total / max(1, batch_count),
            "raw_entropy_mean": raw_entropy_total / max(1, batch_count),
            "total_env_steps": float(self.total_env_steps),
            "total_agent_steps": float(self.total_agent_steps),
            "completed_episodes": float(self.completed_episodes),
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
        }
        return self.last_training_stats

    def learn(self, total_env_steps: int) -> dict[str, float]:
        while self.total_env_steps < total_env_steps:
            self.collect_rollout()
            self.update()
        return self.last_training_stats

    def predict_env(
        self,
        env: gym.Env,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        observations = np.asarray(observation, dtype=np.float32)
        _, edge_features, adjacency, subgraph_mask = self.env_batch._graph_from_env(env, observations)
        with torch.no_grad():
            actions, _, _, _ = self.network.act(
                self._torch_float(observations[None, ...]),
                self._torch_float(edge_features[None, ...]),
                self._torch_float(adjacency[None, ...]),
                self._torch_float(subgraph_mask[None, ...]),
                deterministic=deterministic,
            )
        return actions[0].cpu().numpy()

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "network": self.network.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
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
    ) -> "QMARL":
        checkpoint = torch.load(path, map_location=map_location)
        agent = cls(
            config=QMARLConfig(**checkpoint["config"]),
            env_factory=env_factory,
            opponent_agent_factory=opponent_agent_factory,
        )
        agent.network.load_state_dict(checkpoint["network"])
        if "actor_optimizer" in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if agent.scheduler is not None and checkpoint["scheduler"] is not None:
            agent.scheduler.load_state_dict(checkpoint["scheduler"])
        training_state = checkpoint.get("training_state", {})
        agent.total_env_steps = int(training_state.get("total_env_steps", 0))
        agent.total_agent_steps = int(training_state.get("total_agent_steps", 0))
        agent.n_updates = int(training_state.get("n_updates", 0))
        agent.completed_episodes = int(training_state.get("completed_episodes", 0))
        return agent
