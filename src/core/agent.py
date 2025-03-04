import wandb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from gymnasium import Env
import numpy as np
from functools import partial
from typing import Dict, Tuple, NewType, Callable, Any, Optional, List
from dataclasses import dataclass

from collections import defaultdict
from src.utils.logs import setup_logger
from src.utils.plotting import get_reward_action_heatmaps
from src.storage.base import ImageStorage
from src.models.q_policy_basic import QPolicy, QPolicyBasic

logger = setup_logger("Gym Simulation", f"{__name__}.log")


RewardMapping = NewType("RewardMapping", Dict[Tuple[int, int], float])


@dataclass
class ObservationInfo:
    render_image: torch.Tensor


@dataclass
class MemoryPacket:
    observation: Any
    action: int
    reward: float
    info: Optional[ObservationInfo] = None
    done: bool = False



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


# def image_reward_policy(
#     reward_model: ResNet16,
#     image: torch.Tensor
# ):
#     C, W, H = image.shape
#     images = image.reshape(1, C, W, H)
#     # Get predicted reward (Q-value) from reward model
#     reward_logits = reward_model(images)
#     reward_probs = reward_model.post_process(reward_logits),
#     reward_pred = torch.argmax(
#         reward_probs,
#         dim=1
#     ).item()
#     return reward_pred


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
        info=None
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
        )
    
    def select_action(
        self,
        observation,
        epsilon
    ):
        action = self.policy.get_predicted_action(
            observation,
            epsilon=epsilon
        )
        logger.info("The agent selected action: {action}")
        return action

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
        reward_grade = self.predict_reward(
            next_observation,
            observation,
            action,
            reward,
            done=False,
            info=None
        )
        self.memory.append(
            MemoryPacket(
                observation,
                action,
                reward_grade,
                info=ObservationInfo(
                    render_image=self._env.render()
                )
            )
        )

    def train_rewards(
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
        hindsight = torch.pow(
            torch.linspace(0, 1, steps=memory_length),
            2
        )
        rewards = torch.tensor([packet.reward for packet in self.memory])
        rewards = F.sigmoid(rewards)
        logger.info(f"Rewards configured for training: {rewards}")
        loss_value = 0
        # image_tensor_list = [
        #     torch.from_numpy(
        #         packet.observation_info.render_image
        #     ).to(device=self.device) for packet in self.memory
        # ]
        # logger.info(f"Image tensor list: {image_tensor_list}")
        # observations = torch.stack(image_tensor_list)
        # rewards += (1 if win_state else -1) * hindsight
        # # Compare the rewards to the model's prediction
        # self.reward_optimizer.zero_grad()
        # reward_logits = self.model_reward(observations)
        # reward_probs = self.model_reward.post_process(reward_logits)
        # reward_loss = self.reward_criterion(
        #     reward_probs,
        #     rewards
        # ) * (memory_length ** -1)
        # total_reward_loss = torch.sum(reward_loss) * (memory_length ** -1)
        # logger.info(f"Total rewards: {rewards}")
        # total_rewards = torch.sum(rewards).item() / memory_length
        # logger.info(f"Total rewards: {total_rewards}")
        # total_predicted_rewards = torch.sum(reward_probs).item() / memory_length
        # logger.info(f"Total predicted rewards: {total_predicted_rewards}")
        # logger.info(f"Total reward loss: {reward_loss.item()}")
        # loss_value = total_reward_loss.item()
        # total_reward_loss.backward()
        # self.reward_optimizer.step()
        return loss_value

    def train_actions(
        self,
        win_state: bool = False,
        lr=0.1
    ):
        """
        Updates the action model based on the current memory of packets.
        The action model is trained to predict the action to take
        Args:
            win_state (bool): Did the agent win the game?

        Returns:
            float: The total rewards in the window of memory
        """
        logger.info(f"Agenet memory before computing rewards: {self.memory}")
        loss_value = 0
        # memory_length = len(self.memory)
        # actions = torch.tensor([packet.action for packet in self.memory])
        # logger.info(f"Queue length: {memory_length}")
        # observations = torch.stack([packet.render_images for packet in self.memory])
        # # Compare the rewards to the model's prediction
        # self.action_optimizer.zero_grad()
        # action_logits = self.model_action(observations)
        # action_probs = self.model_action.post_process(action_logits)
        # action_loss = self.action_criterion(
        #     action_probs,
        #     actions
        # )
        # total_action_loss = torch.sum(action_loss) * (memory_length ** -1)
        # loss_value = total_action_loss.item()
        # logger.info(f"Total action loss: {total_action_loss.item()}")
        # total_action_loss.backward()
        # self.action_optimizer.step()
        return loss_value

    def reset(self):
        logger.info("Clearing observation/action memory")
        self.before_reset_hook()
        self.memory = []

    def log_metrics(self):
        if self.debug:
            # wandb.log({
            #     "train/obs-reward-heatmap": wandb.Image(
            #         get_reward_action_heatmaps(
            #             self._rewards_by_action_state,
            #             num_actions=self._env.action_space.n,
            #             grid_size=4
            #         )
            #     )
            # })
            logger.info("Logging metrics to wandb")

    def __str__(self):
        return f"{self.name}"
