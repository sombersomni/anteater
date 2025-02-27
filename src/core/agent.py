import wandb
from gymnasium import Env
import numpy as np
from functools import partial
from typing import Dict, Tuple, NewType, Callable, Any, Optional
from dataclasses import dataclass

from collections import defaultdict
from src.utils.logs import setup_logger
from src.utils.plotting import get_reward_action_heatmaps
from src.storage.base import ImageStorage


logger = setup_logger("Gym Simulation", f"{__name__}.log")


RewardMapping = NewType("RewardMapping", Dict[Tuple[int, int], float])


@dataclass
class ObservationInfoPacket:
    render_image: Optional[np.ndarray] = None


@dataclass
class StateActionRewardPacket:
    state: int
    action: int
    reward: int
    observation_info: Optional[ObservationInfoPacket] = None


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
        name="Agent-v1",
        debug: bool = False,
        reward_fn: Callable[
            [Env, int, Any, Any, int, float, float],
            float
        ] = time_diff_q_learning
    ):
        self._env = env
        self._rewards_by_action_state = defaultdict(int)
        self.reward_fn = partial(reward_fn, self._env)
        self.action = None
        self.debug = debug
        self.name = name
        # Storage
        self.storage = ImageStorage()
        self.memory = []

    def before_reset_hook(self, current_episode: int = 0):
        """
        This hook is called before the agent's memory is reset.
        It is typically used to save the current memory to storage.
        """
        logger.info("Running before reset hook")
        self.storage.write_multiple(
            f"{self.name}_{current_episode}_run",
            (packet.observation_info.render_image for packet in self.memory)
        )

    def add_to_memory(self, packet: StateActionRewardPacket):
        self.memory.append(packet)
        logger.info(f"Added packet to memory: {packet}")

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
        self.memory.append(
            StateActionRewardPacket(
                observation,
                action,
                reward_grade,
                observation_info=ObservationInfoPacket(
                    render_image=self._env.render()
                )
            )
        )

    def compute_rewards(
        self,
        win_state: bool = False,
        lr=0.1
    ):
        """
        Computes the rewards for the current memory of packets.
        We apply a time-based reward decay to the rewards in the memory.
        If it is a win state, we multiply the rewards by 1, otherwise -1,
        ensuring that the rewards are negative for losing states.

        Args:
            win_state (bool): Did the agent win the game?

        Returns:
            float: The total rewards in the window of memory
        """
        logger.info(f"Agenet memory before computing rewards: {self.memory}")
        memory_length = len(self.memory)
        logger.info(f"Queue length: {memory_length}")
        hindsight = np.pow(
            np.linspace(0, 1, memory_length),
            2
        )
        rewards = np.array([packet.reward for packet in self.memory])
        rewards += (1 if win_state else -1) * hindsight
        for idx, packet in enumerate(self.memory):
            self._rewards_by_action_state[(packet.state, packet.action)] += lr * rewards[idx]
        total_rewards = np.sum(rewards).item() / memory_length
        return total_rewards

    def reset(self):
        logger.info("Clearing observation/action memory")
        self.before_reset_hook()
        self.memory = []

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
