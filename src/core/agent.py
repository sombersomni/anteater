import wandb
from functools import partial
from typing import Dict, Tuple, NewType, Callable, Any
from dataclasses import dataclass
from gymnasium import Env
import numpy as np
from collections import defaultdict
from src.utils.logs import setup_logger
from src.utils.plotting import get_reward_action_heatmaps


logger = setup_logger("Gym Simulation", f"{__name__}.log")


RewardMapping = NewType("RewardMapping", Dict[Tuple[int, int], float])

@dataclass
class StateActionRewardPacket:
    state: int
    action: int
    reward: int


def time_diff_q_learning(
    env: Env,
    action,
    reward: float,
    observation,
    next_observation,
    rewards_by_action_state: RewardMapping,
    epsilon=0.01,
    gamma=0.9
):
    """
    Calculates the Q-learning update for the rewards based on the
    current reward and the maximum future reward.
    """
    max_future_reward = np.max(
        [
            rewards_by_action_state.get(
                (next_observation, a),
                0
            ) for a in range(env.action_space.n)
        ]
    ).item()
    previous_reward = rewards_by_action_state.get(
        (observation, action), 0
    )
    new_reward = reward + (gamma * max_future_reward) - previous_reward
    reward_grade = (1 - epsilon) * previous_reward + (epsilon) * new_reward
    return reward_grade


class Agent:
    """
    The agent class is responsible for selecting actions
    and updating the rewards for the actions based on the
    rewards received from the environment. The agent uses"""
    def __init__(
        self,
        env: Env = None,
        name="Agent.v1",
        debug: bool = False,
        reward_fn: Callable[
            [Env, int, Any, Any, int, float, float],
            float
        ] = time_diff_q_learning
    ):
        self._env = env
        self._rewards_by_action_state = defaultdict(int)
        self.reward_fn = partial(reward_fn, self._env)
        self.queue = []
        self.action = None
        self.debug = debug
        self.name = name

    def set_env(self, env: Env):
        self._env = env

    def select_action(
        self,
        observation,
        epsilon
    ):
        if np.random.rand() > epsilon:
            return self._env.action_space.sample()
        # Get the action with the highest reward
        else:
            self.action = np.argmax(
                [
                    self._rewards_by_action_state.get(
                        (observation, a),
                        0
                    ) for a in range(self._env.action_space.n)
                ]
            ).item()
        return self.action

    def update(
        self,
        reward: int,
        observation,
        next_observation,
        action: int,
        gamma=0.9,
        epsilon=0.01
    ):
        """
        Calculates the new reward based on the previous reward
        and the maximum future reward. The future reward
        is calculated by getting the maximum reward in the action space
        for the next observation. Uses Q-learning to update the rewards.

        Args:
            reward (int): The reward for the current action
            observation (int): The current observation
            next_observation (int): The next observation
            action (int): The current action
            gamma (float): The discount factor
            epsilon (float): The learning rate

        Returns:
            None
        """
        reward_grade = self.reward_fn(
            action,
            reward,
            observation,
            next_observation,
            self._rewards_by_action_state,
            epsilon=epsilon,
            gamma=gamma
        )
        self.queue.append(
            StateActionRewardPacket(
                observation,
                action,
                reward_grade
            )
        )

    def compute_rewards(
        self,
        win_state: bool = False,
        lr=0.1
    ):
        """
        Computes the rewards for the current queue of packets.
        We apply a time-based reward decay to the rewards in the queue.
        If it is a win state, we multiply the rewards by 1, otherwise -1,
        ensuring that the rewards are negative for losing states.

        Args:
            win_state (bool): Did the agent win the game?

        Returns:
            float: The total rewards for the queue
        """
        logger.info(f"Agenet queue before computing rewards: {self.queue}")
        queue_length = len(self.queue)
        logger.info(f"Queue length: {queue_length}")
        time_decay = np.pow(
            np.linspace(0, 1, queue_length),
            2
        )
        rewards = np.array([packet.reward for packet in self.queue])
        rewards *= time_decay # Apply time-based reward decay
        rewards += (1 if win_state else -1) * time_decay
        for idx, packet in enumerate(self.queue):
            self._rewards_by_action_state[(packet.state, packet.action)] += lr * rewards[idx]
        total_rewards = np.sum(rewards).item() / queue_length
        self.clear_queue()
        return total_rewards

    def clear_queue(self):
        logger.info("Clearing observation/action queue")
        self.queue = []

    def log_metrics(self):
        if self.debug:
            wandb.log({
                "train/obs-reward-heatmap": wandb.Image(
                    get_reward_action_heatmaps(
                        self._rewards_by_action_state,
                        num_actions=self._env.action_space.n,
                        grid_size=4
                    )
                )
            })

    def __str__(self):
        return f"{self.name} | current action: {self.action}"
