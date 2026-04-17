from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

import mate
from mate.agents import GreedyTargetAgent

from q_marl import QMARL
from mate.wrappers.discrete_action_spaces import DiscreteCamera


CHECKPOINT_PATH = Path("checkpoints/qmarl_mate_8v8_9.pt")


def make_camera_env(render_mode: str):
    base_env = gym.make(
        "MultiAgentTracking-v0",
        config="MATE-8v8-9.yaml",
        render_mode=render_mode,
    )
    return mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint tai {CHECKPOINT_PATH}.")

    agent = QMARL.load(CHECKPOINT_PATH)
    env = make_camera_env(render_mode="human")
    observation, _ = env.reset()
    max_render_steps = int(getattr(env.unwrapped, "max_episode_steps", 10000))
    rewards: list[float] = []

    for step in range(max_render_steps):
        action = agent.predict_env(env, observation, deterministic=True)
        continuous_actions = env.unwrapped.camera_rotation_step, env.unwrapped.camera_zooming_step
        grid = DiscreteCamera.discrete_action_grid(levels=5).astype(np.float32)
        action_high = np.asarray(continuous_actions, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action_high * grid[action])
        rewards.append(float(np.asarray(reward).mean()))
        should_continue = env.render()
        done = bool(terminated or truncated)
        if step % 50 == 0:
            print(f"render_step={step}, reward={rewards[-1]:.3f}, done={done}")
        if done or not should_continue:
            break

    print(f"Tong reward episode = {float(np.sum(rewards)):.3f}")
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
