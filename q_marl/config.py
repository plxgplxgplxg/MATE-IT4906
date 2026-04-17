from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class QMARLConfig:
    env_id: str = "MultiAgentTracking-v0"
    env_config: str = "MATE-4v4-9.yaml"
    render_mode: str = "rgb_array"
    num_envs: int = 8
    rollout_length: int = 256
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    actor_lr: float = 0.003
    critic_lr: float = 0.0005
    optimizer_eps: float = 1e-5
    weight_decay: float = 0.0
    n_epochs: int = 4
    num_mini_batches: int = 4
    critic_extra_steps: int = 1
    hidden_dim: int = 128
    edge_dim: int = 10
    num_message_passing_layers: int = 3
    graph_depth: int = 3
    action_levels: int = 5
    policy_epsilon: float = 0.02
    max_grad_norm: float = 0.5
    normalize_rewards: bool = True
    reward_scale: float = 1.0
    critic_loss: str = "huber"
    critic_huber_delta: float = 10.0
    last_layer_gain: float = 0.01
    use_reduce_on_plateau: bool = True
    scheduler_factor: float = 0.95
    scheduler_patience: int = 10
    device: str = "auto"
    seed: Optional[int] = None
