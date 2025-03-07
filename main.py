import wandb
import gymnasium as gym
import argparse
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
from src.core.agent import Agent
from src.core.simulator import Simulator
from src.env_wrappers.simulator import SimulatorWrapper

from nes_py import NESEnv

logger = setup_logger("Gym Simulation", f"{__name__}.log")


ARCHITECTURE = "Greedy"
EPOCHS = 100


GAME_MAPPING_BY_NAME = {
    "FrozenLake-v1": lambda args: PathFinderRewardWrapper(
            FrozenLakeEnv(
            render_mode=args.render_mode,
            is_slippery=False,
            map_name='4x4'
        )
    ),
    "CarRacing-v3": lambda args: gym.make(
        "CarRacing-v3",
        render_mode=args.render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False
    ),
    "Zelda": lambda args: NESEnv('./roms/the-legend-of-zelda.nes')
}


def get_game_environment_builder(args):
    def default_get_game_builder(args):
        return gym.make(args.game, render_mode=args.render_mode)
    return GAME_MAPPING_BY_NAME.get(args.game, default_get_game_builder)


def initialize():
    parser = create_gym_arg_parser()
    args = parser.parse_args()
    env_builder = get_game_environment_builder(args)
    wrapped_env = SimulatorWrapper(env_builder(args))
    return wrapped_env, args


def run():
    wrapped_env, args = initialize()
    # start a new wandb run to track this script
    if args.debug:
        wandb.init(
            # set the wandb entity where your project will be logged (generally your team name)
            entity="sombersomni-sloparse-labs",
            # set the wandb project where this run will be logged
            project=args.game,
            # track hyperparameters and run metadata
            config={
                "discount_factor": args.discount_factor,
                "architecture": ARCHITECTURE,
                "dataset": args.game,
                "episodes": args.episodes,
                "move_limit": args.move_limit,
                "lr": args.lr,
                "epsilon": args.epsilon,
                "gamma": args.discount_factor,
            }
        )

    agent, simulator = None, None
    agent = Agent(
        env=wrapped_env,
        debug=args.debug
    )
    simulator = Simulator(
        env=wrapped_env,
        agent=agent,
        debug=args.debug
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
            starting_epsilon=args.epsilon,
            lr=args.lr
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.warning(f"Error: {e}")
        sys.exit(1)
    finally:
        if args.debug:
            wandb.finish()
        wrapped_env.close()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == '__main__':
    run()
