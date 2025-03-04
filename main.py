import wandb
import torch
from gymnasium import Env
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import (
    FrozenLakeEnv
)
import cv2
import os
import sys

from src.utils.file_tools import apply_action_to_files
from src.utils.terminal import create_gym_arg_parser
from src.utils.logs import setup_logger
from src.env_wrappers.path_finder import PathFinderRewardWrapper
from src.core.agent import Agent, MemoryPacket, ObservationInfo


logger = setup_logger("Gym Simulation", f"{__name__}.log")


GAMMA_FACTOR = 0.9
ARCHITECTURE = "Greedy"
EPOCHS = 100


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
        gamma=0.9,
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
            gamma=gamma,
            epsilon=epsilon
        )
        return next_observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def start(
        self,
        episodes: int = 1,
        move_limit: int = 10,
        gamma: float = GAMMA_FACTOR,
        starting_epsilon: float = 0.01
    ):
        idx = 0
        epsilon = starting_epsilon
        for episode in tqdm(range(episodes)):
            num_steps_taken = 0
            epsilon = max(starting_epsilon, (episode / episodes) ** 2)
            total_rewards = 0
            done = False
            observation, info = self.reset()
            logger.info(f"Starting episode: {episode}")
            while not done and num_steps_taken < move_limit:
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
                    gamma=gamma,
                    epsilon=epsilon
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
                            (1 if win_state else -1),
                            observation_info=ObservationInfo(
                                render_image=self.env.render()
                            ),
                            done=done,
                        )
                    )
                    # total_reward_loss = self.agent.train_rewards(win_state=win_state)
                    # total_action_loss = self.agent.train_actions(win_state=win_state)
                    logger.info(f"Episode {episode} terminated with total reward: {total_rewards}.")
                    logger.info(f"Win state: {info.get('win_state', False)}")
                    logger.info(f"Total steps taken: {num_steps_taken}.")
                    if self.debug:
                        wandb.log({
                            "total_reward": total_rewards,
                            # "total_reward_loss": total_reward_loss,
                            # "total_action_loss": total_action_loss
                        })
                    self.agent.reset()
                    idx += 1
                    num_steps_taken += 1
                    observation = next_observation
                    break
                idx += 1
                num_steps_taken += 1
                observation = next_observation
        self.agent.log_metrics()


def start_project():
    parser = create_gym_arg_parser()
    args = parser.parse_args()
    wrapped_env = PathFinderRewardWrapper(
        FrozenLakeEnv(
            render_mode=args.render_mode,
            is_slippery=False,
            map_name='4x4'
        )
    )
    # start a new wandb run to track this script
    if args.wb_debug:
        wandb.init(
            # set the wandb entity where your project will be logged (generally your team name)
            entity="sombersomni-sloparse-labs",
            # set the wandb project where this run will be logged
            project="frozenlake",
            # track hyperparameters and run metadata
            config={
                "discount_factor": args.discount_factor,
                "architecture": ARCHITECTURE,
                "dataset": "FrozenLake-v1",
                "episodes": args.episodes
            }
        )
    agent = Agent(
        env=wrapped_env,
        debug=args.wb_debug
    )
    simulator = Simulator(
        env=wrapped_env,
        agent=agent,
        debug=args.wb_debug
    )
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    apply_action_to_files(
        args.output_dir,
        pattern='*.png',
        file_action=os.remove
    )
    try:
        simulator.start(
            episodes=args.episodes,
            move_limit=args.move_limit,
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.warning(f"Error: {e}")
        sys.exit(1)
    finally:
        if args.wb_debug:
            wandb.finish()
        wrapped_env.close()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == '__main__':
    start_project()
