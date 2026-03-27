from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class MAPPOConfig:
    env_id: str = "MultiAgentTracking-v0"
    env_config: str = "MATE-4v2-9.yaml"
    render_mode: str = "rgb_array"
    num_envs: int = 8
    rollout_length: int = 400
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 5e-4
    optimizer_eps: float = 1e-5
    weight_decay: float = 0.0
    n_epochs: int = 5
    num_mini_batches: int = 1
    recurrent_chunk_length: int = 10
    hidden_dim: int = 64
    fc_dim: int = 64
    max_grad_norm: float = 10.0
    huber_delta: float = 10.0
    use_value_normalization: bool = True
    normalize_advantage: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    last_layer_gain: float = 0.01
    device: str = "auto"
    seed: Optional[int] = None
