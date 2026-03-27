from gym_agent import MAPPO, MAPPOConfig


def main() -> None:
    config = MAPPOConfig(
        env_config="MATE-4v2-9.yaml",
        num_envs=2,
        rollout_length=20,
        n_epochs=1,
        num_mini_batches=1,
        recurrent_chunk_length=10,
        device="cpu",
    )
    agent = MAPPO(config=config)
    stats = agent.learn(total_env_steps=40)
    print(stats)
    agent.close()


if __name__ == "__main__":
    main()
