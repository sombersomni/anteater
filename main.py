from gymnasium import Env
from tqdm import tqdm
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import (
    FrozenLakeEnv
)
from pprint import pprint
from collections import defaultdict
import cv2
import os
import sys

from src.utils.file_tools import apply_action_to_files
from src.utils.argparse import create_arg_parser
from src.utils.logging import setup_logger


logger = setup_logger("FrozenLake", "frozenlake.log")


class Player:
    def __init__(self, env: Env = None):
        self._env = env
        self._rewards_by_action_state = defaultdict(int)
        self.action = None
        self.ended = False

    def set_env(self, env: Env):
        self._env = env

    def select_action(
        self,
        observation,
        epsilon
    ):
        if np.random.rand() > epsilon:
            return self._env.action_space.sample().item()
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
        action,
        gamma=0.9,
        epsilon=0.01
    ):
        """
        Calculates the new reward based on the previous reward and the maximum future reward.
        The maximum future reward is calculated by getting the maximum reward for the next observation
        and all possible actions. The new reward is then calculated by adding the reward to the maximum future reward
        """
        max_future_reward = np.max(
            [
                self._rewards_by_action_state.get(
                    (next_observation, a),
                    0
                ) for a in range(self._env.action_space.n)
            ]
        ).item()
        previous_reward = self._rewards_by_action_state.get(
            (observation, action), 0
        )
        new_reward = reward + (gamma * max_future_reward) - previous_reward
        if (observation, action) not in self._rewards_by_action_state:
            self._rewards_by_action_state[(observation, action)] = new_reward
            logger.info(f"Adding reward: {new_reward}")
        else:
            old_reward = self._rewards_by_action_state[(observation, action)]
            loss = (1 - epsilon) * old_reward + (epsilon) * new_reward
            self._rewards_by_action_state[(observation, action)] = loss
            logger.info(f"Updating reward: {loss}")

    def __str__(self):
        return f"Action: {self.action}, Ended: {self.ended}"


class Simulator:
    def __init__(
        self,
        env: Env,
    ):
        self.env = env
        # Set up the pygame window
        self.env.reset()
        self.init_frame = self.env.render()

    def step(
        self,
        player: Player,
        observation,
        epsilon=0.01,
        gamma=0.9,
        time_reward=0,
        visted_state=False
    ):
        player.set_env(self.env)
        action = player.select_action(observation, epsilon)
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        goal_reached = 1 if terminated and reward == 1 else (-1 if terminated else 0)
        reward = time_reward + (int(visted_state) * -0.1) + goal_reached
        player.update(
            reward,
            observation,
            next_observation,
            action,
            gamma=gamma,
            epsilon=epsilon
        )
        return next_observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def start(
        self,
        player: Player,
        episodes: int = 3,
        limit: int = 100,
        gamma: float = 0.9,
        starting_epsilon: float = 0.01
    ):
        idx = 0
        epsilon = starting_epsilon
        # Create a time reward that decreases over time
        time_rewards = np.pow(
            np.linspace(0, 1, limit + 1),
            2
        )
        for episode in tqdm(range(episodes)):
            observation, info = self.reset()
            visted = set()
            t = 0
            done = False
            epsilon = max(starting_epsilon, (episode / episodes) ** 2)
            while not done or t < limit:
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                if t >= limit:
                    logger.info("Episode truncated.")
                    break
                if episodes <= 0:
                    logger.warning("Episodes must be greater than 0.")
                    break
                time_reward = time_rewards[t].item()
                visited_state = observation in visted
                visted.add(observation)
                next_observation, reward, terminated, truncated, info, = self.step(
                    player,
                    observation,
                    epsilon=epsilon,
                    gamma=gamma,
                    time_reward=time_reward,
                    visted_state=visited_state
                )
                # Get the current frame to render the environment
                # If the episode has ended then we can reset to start a new episode
                if terminated:
                    logger.info("Episode terminated.")
                    break
                # observation = next_observation
                idx += 1
                t += 1
                observation = next_observation
                logger.info(f"Ending epsiode with a reward of: {reward}")
                logger.info(f"Episode: {episode}, Epsilon: {epsilon}")


def start_project():
    # Create the environment
    env = FrozenLakeEnv(
        render_mode='human',
        is_slippery=False,
        map_name='4x4'
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
        player = Player()
        simulator.start(
            player,
            episodes=args.num_episodes,
            limit=20,
        )
        pprint(player._rewards_by_action_state)
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
