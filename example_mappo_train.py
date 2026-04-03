from __future__ import annotations

from pathlib import Path

from gym_agent import MAPPO, MAPPOConfig


CHECKPOINT_PATH = Path("checkpoints/mappo_mate_8v8_9.pt")
TRAIN_ENV_STEPS = 50_000
TRAIN_CHUNK_STEPS = 10_000


def build_agent() -> MAPPO:
    config = MAPPOConfig(
        env_config="MATE-8v8-9.yaml",
        num_envs=8,
        rollout_length=400,
        n_epochs=5,
        num_mini_batches=1,
        recurrent_chunk_length=20,
        critic_input_mode="agent_specific",
        clip_range=0.1,
        device="cpu",
        render_mode="rgb_array",
    )
    return MAPPO(config=config)


def main() -> None:
    agent = build_agent()
    print(f"Dang train tong cong {TRAIN_ENV_STEPS} env steps...")

    while agent.total_env_steps < TRAIN_ENV_STEPS:
        target_steps = min(agent.total_env_steps + TRAIN_CHUNK_STEPS, TRAIN_ENV_STEPS)
        stats = agent.learn(total_env_steps=target_steps)
        print(
            "train_progress:",
            {
                "env_steps": int(stats["total_env_steps"]),
                "agent_steps": int(stats["total_agent_steps"]),
                "actor_loss": round(stats["actor_loss"], 4),
                "critic_loss": round(stats["critic_loss"], 4),
                "entropy": round(stats["entropy"], 4),
                "completed_episodes": int(stats["completed_episodes"]),
            },
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent.save(CHECKPOINT_PATH)
    print(f"Da luu checkpoint vao: {CHECKPOINT_PATH}")
    agent.close()


if __name__ == "__main__":
    main()
