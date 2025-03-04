from collections import defaultdict
from gymnasium import Env
from gymnasium.spaces import Discrete
from unittest.mock import Mock
from typing import Dict, Tuple
from src.models.q_policy_basic import QPolicy


class FakeGameEnv(Env):
    def __init__(self):
        super(FakeGameEnv, self).__init__()
        # The action space will be 0 to 3 and represent 0. UP, 1. LEFT, 2. DOWN, 3. RIGHT
        self.action_space = Discrete(4)
        # The observation space will represent 4 x 4 grid for the player to move.
        self.observation_space = Discrete(16)
        self.num_steps = 0

    def step(self, action):
        observation = self.observation_space.sample()
        reward = 1 if observation == 1 else 0  # Example reward
        terminated = False  # Replace with your termination logic
        truncated = self.num_steps >= 100  # Example termination condition
        info = {}  # Add any extra info
        self.num_steps += 1
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}



class QMockPolicy(QPolicy):
    def __init__(self, env: Env):
        self.reward_state: Dict[Tuple[int, int], float] = defaultdict(float)
        self.action_space_size = 4
    
    def update_reward_state(self, observation, action, new_reward: float, done: bool, info):
        self.reward_state[(observation, action)] += new_reward

    def get_predicted_action(self, observation, epsilon=0.1, generator=None):
        return 0
    
    def get_predicted_reward(
        self,
        next_observation,
        observation,
        action,
        reward,
        done,
        info=None,
        lr: float = 0.1
    ):
        return 0