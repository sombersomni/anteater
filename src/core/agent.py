import wandb
import torch
import torch.nn.functional as F
from gymnasium import Env
from typing import Any, Optional, List
from dataclasses import dataclass

from src.utils.logs import setup_logger
from src.utils.plotting import get_reward_action_heatmaps
from src.storage.base import ImageStorage
from src.models.q_policy_basic import QPolicy, QPolicyBasic

logger = setup_logger("Gym Simulation", f"{__name__}.log")


@dataclass
class ObservationInfo:
    render_image: torch.Tensor

    def __str__(self):
        return f"ObservationInfo: img_shape={self.render_image.shape}"


@dataclass
class MemoryPacket:
    observation: Any
    action: int
    reward: float
    info: Optional[ObservationInfo] = None
    done: bool = False

    def __str__(self):
        return f"MemoryPacket: obs:{self.observation}, act:{self.action}, reward:{self.reward}"


def format_packets_to_image_batch(
    packets: List[MemoryPacket],
    device='cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Formats the observation for the model.
    """
    images = torch.stack(
        [packet.info.render_image for packet in packets],
        dtype=torch.float32
    ).to(device)
    # Images should be 4D tensors (batch_size, channels, height, width)
    # Channels should be 1 for grayscale images
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.dim() == 2:
        images = images.unsqueeze(1).unsqueeze(0)
    return images


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
        policy: QPolicy = None,
    ):
        self._env = env
        self.action = None
        self.debug = debug
        self.name = name
        # Storage
        self.storage = ImageStorage()
        self.memory: List[MemoryPacket]  = []
        # Models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = QPolicyBasic(env) if policy is None else policy

    def before_reset_hook(self, current_episode: int = 0):
        """
        This hook is called before the agent's memory is reset.
        It is typically used to save the current memory to storage.
        """
        logger.info("Running before reset hook")
        self.storage.write_multiple(
            f"{self.name}_{current_episode}_run",
            (packet.info.render_image for packet in self.memory)
        )

    def add_to_memory(self, packet: MemoryPacket):
        self.memory.append(packet)
        logger.info(f"Added packet to memory: {packet}")

    def set_env(self, env: Env):
        self._env = env

    def predict_reward(
        self,
        next_observation,
        observation,
        action,
        reward,
        done=False,
        info=None,
        lr=0.1
    ) -> int:
        """
        Returns the action with the highest reward for the current observation.
        """
        return self.policy.get_predicted_reward(
            next_observation,
            observation,
            action,
            reward,
            done,
            info,
            lr=lr
        )
    
    def select_action(
        self,
        observation,
        epsilon: float = 0.1
    ):
        action = self.policy.get_predicted_action(
            observation,
            epsilon=epsilon
        )
        logger.info(f"ObservationInfo action: {action}")
        return action

    def update(
        self,
        reward: int,
        observation,
        next_observation,
        action: int,
        done: bool = False,
        info: Optional[ObservationInfo] = None,
        lr=0.1
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
            lr (float): The learning rate

        Returns:
            None
        """
        reward_grade = self.predict_reward(
            next_observation,
            observation,
            action,
            reward,
            done=done,
            info=info,
            lr=lr
        )
        self.memory.append(
            MemoryPacket(
                observation,
                action,
                reward_grade,
                info=ObservationInfo(
                    render_image=self._env.render()
                ),
                done=done
            )
        )
        return reward_grade

    def compute_loss(
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
        logger.info(f"Agent memory before computing rewards: {tuple(str(m) for m in self.memory)}")
        memory_length = len(self.memory)
        logger.info(f"Queue length: {memory_length}")
        hindsight = torch.pow(
            torch.linspace(0, 1, steps=memory_length),
            2
        )
        hindsight = (1 if win_state else -1) * hindsight
        for idx, packet in enumerate(self.memory):
            print(packet)
            self.policy.reward_state[(packet.observation, packet.action)] += hindsight[idx].item()
        logger.info(f"Agent memory after computing rewards: {tuple(str(m) for m in self.memory)}")

    def reset(self, current_episode: int = 0):
        logger.info("Clearing observation/action memory")
        self.before_reset_hook(current_episode=current_episode)
        self.memory = []

    def log_metrics(self):
        if self.debug:
            wandb.log({
                "train/obs-reward-heatmap": wandb.Image(
                    get_reward_action_heatmaps(
                        self.policy.reward_state,
                        num_actions=self._env.action_space.n,
                        grid_size=4
                    )
                )
            })
            logger.info("Logging metrics to wandb")

    def __str__(self):
        return f"{self.name}"
