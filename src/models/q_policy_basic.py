import torch
from gymnasium import Env
from dataclasses import dataclass
from typing import Protocol, List, Dict, Tuple, Any
from src.utils.logs import setup_logger

logger = setup_logger("QPolicyBasic", f"{__name__}.log")


class ActionPolicy(Protocol):
    def get_predicted_action(self, observation, env=None, epsilon=0.1) -> int:
        """
        Returns the action with the highest reward for the current observation.
        """
        pass


class RewardPolicy(Protocol):
    def get_predicted_reward(self, observation, action, reward, done, info, env=None):
        """
        Calculates the Q-learning update for the rewards based on the
        current reward and the maximum future reward.
        """
        pass


class QPolicy(ActionPolicy, RewardPolicy):
    history: List


class QPolicyBasic(QPolicy):
    def __init__(self, env: Env, gamma: float = 0.9, generator = None):
        self.reward_state: Dict[Tuple[int, int], float] = {}
        self.action_space = torch.tensor([i for i in range(env.action_space.n)])
        self.action_space_size = env.action_space.n
        self.generator = generator if generator else torch.Generator().manual_seed(101)
        self.env = env
        self.gamma = gamma

    def update_reward_state(self, observation, action, new_reward: float, done: bool, info):
        self.reward_state[(observation, action)] = new_reward

    def get_predicted_action(
        self,
        observation,
        epsilon: float = 0.1
    ):
        """
        Returns the action with the highest reward for the current observation.
        If no reward exists for the current observation, returns a random action.
        """
        if torch.rand(1, generator=self.generator).item() < epsilon:
            return torch.randint(
                0,
                self.action_space_size,
                (1,),
                generator=self.generator
            ).item()
        
        max_idx = torch.argmax(
            torch.tensor(
                [
                  self.get_reward(observation, action.item())
                  for action in self.action_space
                ]
            )
        ).item()
        return self.action_space[max_idx].item()

    def get_reward(
        self,
        observation,
        action,
        done: bool = False,
        info = None
    ):
        """
        Returns the reward for the current observation and action.
        If no reward exists for the current observation and action, returns 0.
        """
        return self.reward_state.get((observation, action), 0)

    def get_max_reward(
        self,
        observation,
    ):
        """
        Returns the action with the highest reward for the current observation.
        If no reward exists for the current observation, returns a random action.
        """
        return torch.max(
            torch.tensor(
                [
                  self.get_reward(observation, action.item())
                  for action in self.action_space
                ]
            )
        ).item()

    def get_predicted_reward(
        self,
        next_observation,
        observation,
        action,
        reward,
        done,
        info,
        lr: float = 0.01
    ):

        """
        Calculates the Q-learning update for the rewards based on the
        current reward and the maximum future reward.
        """
        logger.info(f"Calculating Q-learning update for observation, action pair")
        future_reward_value = self.get_max_reward(next_observation)
        past_reward_value = self.get_reward(observation, action)
        new_reward = (reward + (self.gamma * future_reward_value) - past_reward_value)
        new_overall_reward = (1 - lr) * reward + lr * new_reward
        logger.info(f"New overall reward: {new_overall_reward}")
        self.update_reward_state(observation, action, new_overall_reward, done, info)
        return 

    def loss(
        self,
        new_reward: int,
        reward: int,
        lr: float = 0.01  # we increase the learning rate over time
    ):
        return (1 - lr) * reward + lr * new_reward