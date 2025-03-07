import torch
from gymnasium import Env
from gymnasium.spaces.discrete import Discrete
from typing import Protocol, List, Dict, Tuple
from collections import defaultdict
from src.utils.logs import setup_logger


logger = setup_logger("QPolicyBasic", f"{__name__}.log")


class ActionPolicy(Protocol):
    def get_predicted_action(self, observation, epsilon=0.1) -> int:
        """
        Returns the action with the highest reward for the current observation.
        """
        pass


class RewardPolicy(Protocol):
    def get_predicted_reward(
        self,
        next_observation,
        observation,
        action,
        reward,
        done=False,
        info=None,
        lr: float = 0.1
    ):
        """
        Calculates the Q-learning update for the rewards based on the
        current reward and the maximum future reward.
        """
        pass


class QPolicy(ActionPolicy, RewardPolicy):
    reward_state: Dict[Tuple[int, int], float]
    history: List
    env: Env
    gamma: float

    
class QPolicyBasic(QPolicy):
    def __init__(self, env: Env, gamma: float = 0.9, generator = None):
        print("QPolicyBasic", env.action_space, type(env.action_space))
        if not isinstance(env.action_space, Discrete):
            raise ValueError("Only Discrete action spaces are supported")
        self.action_space_size = env.action_space.n
        self.reward_state: Dict[Tuple[int, int], float] = defaultdict(float)
        self.generator = generator if generator else torch.Generator().manual_seed(101)
        self.env = env
        self.gamma = gamma

    def update_reward_state(self, observation, action, new_reward: float, done: bool, info):
        self.reward_state[(observation, action)] += new_reward

    def get_all_rewards_for_observation(self, observation):
        return [
            self.get_reward(observation, action)
            for action in range(self.action_space_size)
        ]

    def get_predicted_action(
        self,
        observation,
        epsilon: float = 0.1
    ):
        """
        Returns the action with the highest reward for the current observation.
        If no reward exists for the current observation, returns a random action.
        """
        action = None
        prob = torch.rand(1, generator=self.generator).item()
        if prob > epsilon:
            logger.info(f"Selecting a random action based on epsilon-greedy policy")
            action = self.env.action_space.sample().item()
        else:
            logger.info(f"Selecting the maximum reward action")
            action = torch.argmax(
                torch.tensor(self.get_all_rewards_for_observation(observation))
            ).item()
        logger.info(f"Selected action: {action}")
        return action

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
            torch.tensor(self.get_all_rewards_for_observation(observation))
        ).item()

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
        return new_overall_reward
