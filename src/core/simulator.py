from tqdm import tqdm
import wandb
from gymnasium import Env
from typing import List, Tuple

from src.models.q_policy_basic import QPolicy
from src.utils.logs import setup_logger
from src.core.agent import Agent, MemoryPacket, ObservationInfo


logger = setup_logger("Simulator", f"{__name__}.log")


class Simulator:
    def __init__(
        self,
        env: Env=None,
        agent: Agent=None,
        debug: bool=False
    ):
        if agent is None:
            raise ValueError("Agent cannot be None.")
        if env is None:
            raise ValueError("Environment cannot be None.")
        self.env = env
        self.agent = agent
        self.agent.set_env(env)
        # Set up the pygame window
        self.env.reset()
        self.init_frame = self.env.render()
        self.debug = debug

    def step(
        self,
        observation,
        epsilon=0.01,
        lr=0.1
    ):
        action = self.agent.select_action(
            observation,
            epsilon
        )
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.agent.update(
            reward,
            observation,
            next_observation,
            action,
            lr=lr
        )
        return next_observation, reward, terminated, truncated, info

    def reset(self, current_episode: int = 0):
        self.agent.reset(current_episode=current_episode)
        return self.env.reset()

    def render(self):
        return self.env.render()

    def start(
        self,
        episodes: int = 1,
        move_limit: int = 10,
        starting_epsilon: float = 0.01,
        lr: float = 0.1
    ):
        idx = 0
        epsilon = starting_epsilon
        for episode in tqdm(range(episodes)):
            num_steps_taken = 0
            epsilon = max(starting_epsilon, (episode / episodes) ** 2)
            total_rewards = 0
            done = False
            observation, info = self.reset(current_episode=episode)
            while not done and num_steps_taken < move_limit:
                logger.info(f"Episode: {episode}")
                logger.info(f"Step: {num_steps_taken}")
                logger.debug(f"Move limit: {move_limit}")
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                if episodes <= 0:
                    logger.warning("Episodes must be greater than 0.")
                    break
                action = self.agent.select_action(
                    observation,
                    epsilon
                )
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.agent.update(
                    reward,
                    observation,
                    next_observation,
                    action,
                    done=done,
                    info=info,
                    lr=lr
                )
                total_rewards += reward
                # Get the current frame to render the environment
                # If the episode has ended then we can reset to start a new episode
                if done:
                    logger.info(f"Episode {episode} terminated with total reward: {total_rewards}.")
                    win_state = info.get("win_state", False)
                    # Add the final state to agent memory
                    self.agent.add_to_memory(
                        MemoryPacket(
                            next_observation,
                            action,
                            reward,
                            info=ObservationInfo(
                                render_image=self.env.render()
                            ),
                            done=done
                        )
                    )
                    total_reward_loss = self.agent.compute_loss(win_state=win_state)
                    logger.info(f"Episode {episode} terminated with total reward: {total_rewards}.")
                    logger.info(f"Win state: {info.get('win_state', False)}")
                    logger.info(f"Total steps taken: {num_steps_taken}.")
                    if self.debug:
                        wandb.log({
                            "train/total_reward": total_rewards,
                            "train/total_reward_loss": total_reward_loss
                        })
                    idx += 1
                    num_steps_taken += 1
                    observation = next_observation
                    break
                idx += 1
                num_steps_taken += 1
                observation = next_observation
        self.agent.log_metrics()

