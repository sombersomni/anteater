from gym import Env
from tqdm import tqdm
import pygame
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import (
    FrozenLakeEnv,
    generate_random_map
)
import cv2
import os
import sys

from src.utils.file_tools import apply_action_to_files
from src.utils.argparse import create_arg_parser
from src.utils.logging import setup_logger


logger = setup_logger("FrozenLake", "frozenlake.log")


class Simulator:
    def __init__(
        self,
        env: Env,
        num_episodes: int = 1
    ):
        self.env = env
        # Set up the pygame window
        self.env.reset()
        self.init_frame = self.env.render()

    def step(self):
        logger.info(f"Starting Info: {info}")
        action = self.env.action_space.sample()
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        logger.info(f"Action, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return next_observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def start(self):
        idx = 0
        threshold = 10
        episodes = 1
        for episode in tqdm(range(episodes)):
            logger.info(f"Episode: {episode}")
            t = 0
            done = False
            observation, info =self.reset()
            while not done or t < threshold:
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                next_observation, reward, terminated, truncated, info, = self.step()
                # Get the current frame to render the environment
                # If the episode has ended then we can reset to start a new episode
                if terminated:
                    logger.info("Episode terminated.")
                    break
                # observation = next_observation
                idx += 1


def start_project():
    # Create the environment
    env_id = 'FrozenLake-v1'
    env = FrozenLakeEnv(
        desc=generate_random_map(),
        render_mode='human',
        is_slippery=False
    )
    simulator = Simulator(env)
    # Reset the environment to generate the first observation
    parser = create_arg_parser()
    args = parser.parse_args()
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    apply_action_to_files(
        args.output_dir,
        pattern='*.png',
        file_action=os.remove
    )
    try:
        simulator.start()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.warning(f"Error: {e}")
        sys.exit(1)
    finally:
        env.close()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == '__main__':
    start_project()
