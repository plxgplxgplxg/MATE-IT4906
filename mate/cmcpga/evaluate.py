import argparse

import gymnasium as gym
import numpy as np

import mate
from mate.cmcpga.agent import CMCPGACameraAgent


def run_cmcpga_episode(env, agent: CMCPGACameraAgent, seed: int) -> float:
    observations, infos = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(observations, infos)
        observations, reward, terminated, truncated, infos = env.step(action)
        total_reward += reward
        done = bool(terminated or truncated)
    return float(total_reward)


def run_greedy_episode(env, agents, seed: int) -> float:
    observations, _ = env.reset(seed=seed)
    infos = None
    mate.group_reset(agents, observations)
    total_reward = 0.0
    done = False
    while not done:
        action = mate.group_step(env.unwrapped, agents, observations, infos)
        observations, reward, terminated, truncated, infos = env.step(action)
        total_reward += reward
        done = bool(terminated or truncated)
    return float(total_reward)


def make_env(config: str):
    base_env = gym.make("MultiAgentTracking-v0", config=config)
    return mate.MultiCamera.make(base_env, target_agent=mate.GreedyTargetAgent())


def evaluate(config: str, episodes: int, epsilon_deg: float) -> dict[str, float]:
    env_cmcpga = make_env(config)
    env_greedy = make_env(config)
    cmcpga_agent = CMCPGACameraAgent(env_cmcpga, epsilon_deg=epsilon_deg)
    greedy_agents = mate.GreedyCameraAgent().spawn(env_greedy.unwrapped.num_cameras)

    cmcpga_rewards = []
    greedy_rewards = []

    for episode in range(episodes):
        seed = episode * 42
        cmcpga_reward = run_cmcpga_episode(env_cmcpga, cmcpga_agent, seed)
        greedy_reward = run_greedy_episode(env_greedy, greedy_agents, seed)
        cmcpga_rewards.append(cmcpga_reward)
        greedy_rewards.append(greedy_reward)
        print(
            f"Episode {episode + 1:03d} | "
            f"CMCPGA: {cmcpga_reward:8.2f} | "
            f"Greedy: {greedy_reward:8.2f}"
        )

    env_cmcpga.close()
    env_greedy.close()

    mean_cmcpga = float(np.mean(cmcpga_rewards))
    mean_greedy = float(np.mean(greedy_rewards))
    denominator = max(abs(mean_greedy), 1e-9)
    improvement = 100.0 * (mean_cmcpga - mean_greedy) / denominator

    return {
        "cmcpga_mean_reward": mean_cmcpga,
        "greedy_mean_reward": mean_greedy,
        "relative_improvement_percent": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="MATE-4v8-9.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epsilon-deg", type=float, default=5.0)
    args = parser.parse_args()

    results = evaluate(args.config, args.episodes, args.epsilon_deg)
    print()
    print(f"CMCPGA mean reward : {results['cmcpga_mean_reward']:.2f}")
    print(f"Greedy mean reward : {results['greedy_mean_reward']:.2f}")
    print(
        "Relative improvement: "
        f"{results['relative_improvement_percent']:+.2f}%"
    )


if __name__ == "__main__":
    main()
