from __future__ import annotations

from pathlib import Path

from q_marl import QMARL, QMARLConfig


CHECKPOINT_PATH = Path("checkpoints/qmarl_mate_8v8_9.pt")
TRAIN_ENV_STEPS = 50_000
TRAIN_CHUNK_STEPS = 10_000


def build_agent() -> QMARL:
    return QMARL(
        config=QMARLConfig(
            env_config="MATE-8v8-9.yaml",
            num_envs=8,
            rollout_length=256,
            n_epochs=4,
            num_mini_batches=4,
            graph_depth=3,
            edge_dim=10,
            action_levels=5,
            actor_lr=0.003,
            critic_lr=0.0005,
            entropy_coef=0.02,
            normalize_rewards=True,
            critic_loss="huber",
            critic_extra_steps=1,
            device="cpu",
        )
    )


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
                "mean_delta": round(stats["mean_delta"], 4),
                "baseline_mean": round(stats["baseline_mean"], 4),
                "q_mean": round(stats["q_mean"], 4),
                "actor_lr": round(stats["actor_lr"], 6),
                "critic_lr": round(stats["critic_lr"], 6),
            },
        )
    agent.save(CHECKPOINT_PATH)
    print(f"Da luu checkpoint vao: {CHECKPOINT_PATH}")
    agent.close()


if __name__ == "__main__":
    main()
