import unittest
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from models.q_policy_basic import QPolicyBasic
from models.mocks import FakeGameEnv, QMockPolicy


class TestQPolicyBasic(unittest.TestCase):
    def setUp(self):
        self.env = FakeGameEnv()
    
    def test_protocol(self):
        policy = QPolicyBasic(
            self.env.action_space,
            env=self.env,
            gamma=0.9
        )
        self.assertIsInstance(policy, QPolicyBasic)

    def test_get_action_max(self):
        observation = 0
        policy = QPolicyBasic(
            self.env.action_space,
            env=self.env,
            gamma=0.9
        )
        for i in range(4):
            policy.update_reward_state(observation, i, i, False, {})
        action = policy.get_action(observation)
        self.assertEqual(action, 3)

    def test_get_reward_default(self):
        policy = QPolicyBasic(
            self.env.action_space,
            env=self.env,
            gamma=0.9
        )
        observation = 0
        action = 1
        reward = policy.get_predicted_reward(observation, action, 1, False, {})
        self.assertEqual(reward, 0)  # Check the reward is within the action space

    def test_get_max_reward(self):
        observation = 0
        policy = QPolicyBasic(
            self.env.action_space,
            env=self.env,
            gamma=0.9
        )
        for i in range(4):
            policy.update_reward_state(observation, i, 1 if i == 0 else 0, False, {})
        max_reward = policy.get_max_reward(observation)
        self.assertEqual(max_reward, 1)

    def test_get_predicted_reward(self):
        observation = 0
        action = 1
        reward = 0.1  # Example reward
        done = False
        info = {}
        policy = QPolicyBasic(
            self.env.action_space,
            env=self.env,
            gamma=0.9
        )
        for i in range(4):
            policy.update_reward_state(observation, i, 1 if i == 0 else 0, False, {})
        new_reward = policy.get_predicted_reward(
            observation,
            action,
            reward,
            done,
            info,
        )
        self.assertEqual(new_reward, 1)
