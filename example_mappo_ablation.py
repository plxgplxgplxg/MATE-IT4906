from __future__ import annotations

import csv
from pathlib import Path

from gym_agent import MAPPO, MAPPOConfig


RESULTS_PATH = Path("results/mappo_ablation_summary.csv")
TRAIN_ENV_STEPS = 50_000


def run_trial(*, critic_input_mode: str, clip_range: float, seed: int) -> dict[str, float | str]:
    config = MAPPOConfig(
        env_config="MATE-8v8-9.yaml",
        num_envs=8,
        rollout_length=400,
        n_epochs=5,
        num_mini_batches=1,
        recurrent_chunk_length=20,
        critic_input_mode=critic_input_mode,
        clip_range=clip_range,
        device="cpu",
        render_mode="rgb_array",
        seed=seed,
    )

    agent = MAPPO(config=config)
    try:
        stats = agent.learn(total_env_steps=TRAIN_ENV_STEPS)
        return {
            "critic_input_mode": critic_input_mode,
            "clip_range": clip_range,
            "seed": seed,
            "actor_loss": float(stats["actor_loss"]),
            "critic_loss": float(stats["critic_loss"]),
            "entropy": float(stats["entropy"]),
            "total_env_steps": float(stats["total_env_steps"]),
            "completed_episodes": float(stats["completed_episodes"]),
        }
    finally:
        agent.close()


def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "critic_input_mode",
        "clip_range",
        "seed",
        "actor_loss",
        "critic_loss",
        "entropy",
        "total_env_steps",
        "completed_episodes",
    ]

    trials = [
        {"critic_input_mode": "global", "clip_range": 0.05, "seed": 42},
        {"critic_input_mode": "global", "clip_range": 0.10, "seed": 42},
        {"critic_input_mode": "agent_specific", "clip_range": 0.05, "seed": 42},
        {"critic_input_mode": "agent_specific", "clip_range": 0.10, "seed": 42},
    ]

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for trial in trials:
            print("running_trial:", trial)
            result = run_trial(**trial)
            writer.writerow(result)
            csv_file.flush()
            print("trial_result:", result)

    print(f"Da luu ket qua ablation vao: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
