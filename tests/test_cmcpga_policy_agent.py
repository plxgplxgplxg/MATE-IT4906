import unittest

import mate
from mate.environment import read_config


class CMCPGAPolicyAgentTests(unittest.TestCase):
    def test_policy_agent_runs_in_multitarget_pipeline(self):
        config = read_config("MATE-4v8-9.yaml", max_episode_steps=4)
        env = mate.make(
            "MultiAgentTracking-v0",
            config=config,
            wrappers=[mate.WrapperSpec(mate.MultiTarget, camera_agent=mate.CMCPGACameraPolicyAgent())],
        )

        target_agents = mate.GreedyTargetAgent().spawn(env.unwrapped.num_targets)
        observation, _ = env.reset(seed=0)
        mate.group_reset(target_agents, observation)

        action = mate.group_step(env, target_agents, observation, None)
        next_observation, reward, terminated, truncated, infos = env.step(action)

        self.assertEqual(len(action), env.unwrapped.num_targets)
        self.assertEqual(len(next_observation), env.unwrapped.num_targets)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(len(infos), env.unwrapped.num_targets)

        env.close()


if __name__ == "__main__":
    unittest.main()
