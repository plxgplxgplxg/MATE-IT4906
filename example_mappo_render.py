from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym

import mate
from mate.agents import GreedyTargetAgent

from gym_agent import MAPPO, MAPPOConfig


# Duong dan checkpoint
CHECKPOINT_PATH = Path("checkpoints/mappo_global_clip010_entropy0001_best_eval.pt")
MAX_RENDER_STEPS = 1000


def make_camera_env(render_mode: str):
    # Tao env goc
    base_env = gym.make(
        "MultiAgentTracking-v0",
        config="MATE-8v8-9.yaml",
        render_mode=render_mode,
    )
    # Boc env camera
    return mate.MultiCamera.make(
        base_env,
        target_agent=GreedyTargetAgent(),
    )


def build_agent() -> MAPPO:
    # Cau hinh MAPPO
    config = MAPPOConfig(
        env_config="MATE-8v8-9.yaml",
        num_envs=8,
        rollout_length=400,
        n_epochs=5,
        num_mini_batches=1,
        recurrent_chunk_length=20,
        device="cpu",
        render_mode="rgb_array",
    )
    # Tao agent moi
    return MAPPO(config=config)


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint tai {CHECKPOINT_PATH}. "
            "Hay chay example_mappo_train.py truoc."
        )

    print(f"Dang load checkpoint tu: {CHECKPOINT_PATH}")
    agent = MAPPO.load(CHECKPOINT_PATH)

    print("Dang mo cua so render...")
    # Tao env render
    env = make_camera_env(render_mode="human")
    observation, _ = env.reset()
    actor_hidden_state: np.ndarray | None = None
    episode_start: np.ndarray | None = None

    # Vong lap render
    for step in range(MAX_RENDER_STEPS):
        # Suy ra action
        action, actor_hidden_state = agent.predict(
            observation,
            actor_hidden_state=actor_hidden_state,
            episode_start=episode_start,
            deterministic=True,
        )
        # Chay mot buoc
        observation, reward, terminated, truncated, _ = env.step(action)

        # Ve khung hinh
        should_continue = env.render()
        done = bool(terminated or truncated)
        # Danh dau done
        episode_start = np.full(env.unwrapped.num_cameras, done, dtype=np.bool_)

        if step % 50 == 0:
            print(
                f"render_step={step}, reward={np.asarray(reward).mean():.3f}, done={done}"
            )

        # Dieu kien dung
        if not should_continue or done:
            break

    # Dong tai nguyen
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
