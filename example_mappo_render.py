from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym

import mate
from mate.agents import GreedyTargetAgent

from gym_agent import MAPPO, MAPPOConfig


# Duong dan checkpoint
CHECKPOINT_PATH = Path("checkpoints/mappo_global_clip010_entropy0001_best_eval.pt")


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
    max_render_steps = int(getattr(env.unwrapped, "max_episode_steps", 10000))
    episode_rewards: list[float] = []
    final_info: dict | None = None

    # Vong lap render
    for step in range(max_render_steps):
        # Suy ra action
        action, actor_hidden_state = agent.predict(
            observation,
            actor_hidden_state=actor_hidden_state,
            episode_start=episode_start,
            deterministic=True,
        )
        # Chay mot buoc
        observation, reward, terminated, truncated, info = env.step(action)
        step_reward = float(np.asarray(reward).mean())
        episode_rewards.append(step_reward)
        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
            final_info = info[0]
        elif isinstance(info, dict):
            final_info = info

        # Ve khung hinh
        should_continue = env.render()
        done = bool(terminated or truncated)
        # Danh dau done
        episode_start = np.full(env.unwrapped.num_cameras, done, dtype=np.bool_)

        if step % 50 == 0:
            print(
                f"render_step={step}, reward={step_reward:.3f}, done={done}"
            )

        # Dieu kien dung
        if not should_continue or done:
            if done:
                print(f"Episode ket thuc o step={step}.")
            break
    else:
        print(
            f"Da cham gioi han render_step={max_render_steps} truoc khi env bao done."
        )

    episode_total_reward = float(np.sum(episode_rewards)) if episode_rewards else 0.0
    episode_mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    coverage = np.nan
    transport = np.nan
    cargo = np.nan
    if final_info is not None:
        coverage = float(final_info.get("coverage_rate", np.nan))
        transport = float(final_info.get("mean_transport_rate", np.nan))
        cargo = float(final_info.get("num_delivered_cargoes", np.nan))
    else:
        coverage = float(getattr(env.unwrapped, "coverage_rate", np.nan))
        transport = float(getattr(env.unwrapped, "mean_transport_rate", np.nan))
        cargo = float(getattr(env.unwrapped, "num_delivered_cargoes", np.nan))

    print(f"Tong reward episode = {episode_total_reward:.3f}")
    print(f"Reward trung binh moi step = {episode_mean_reward:.3f}")
    print(f"Coverage = {coverage:.3f}")
    print(f"Transport = {transport:.3f}")
    print(f"Delivered cargoes = {cargo:.2f}")

    # Dong tai nguyen
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
