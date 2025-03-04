import wandb
from gymnasium import gym, Env
from tqdm import tqdm

import cv2
import os
import sys

from src.utils.file_tools import apply_action_to_files
from src.utils.terminal import create_gym_arg_parser
from src.utils.logs import setup_logger
from src.env_wrappers.path_finder import PathFinderRewardWrapper
from src.core.agent import Agent, MemoryPacket, ObservationInfo


logger = setup_logger("Gym Simulation", f"{__name__}.log")


ARCHITECTURE = "Greedy"
EPOCHS = 100

def start_project():
    parser = create_gym_arg_parser()
    args = parser.parse_args()
    wrapped_env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95)
    # start a new wandb run to track this script
    if args.debug:
        wandb.init(
            # set the wandb entity where your project will be logged (generally your team name)
            entity="sombersomni-sloparse-labs",
            # set the wandb project where this run will be logged
            project="CarRacing-v0",
            # track hyperparameters and run metadata
            config={
                "discount_factor": args.discount_factor,
                "architecture": ARCHITECTURE,
                "dataset": "FrozenLake-v1",
                "episodes": args.episodes,
                "move_limit": args.move_limit,
                "lr": args.lr,
                "epsilon": 0.01,
                "gamma": args.discount_factor,
            }
        )
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
            gamma=args.discount_factor,
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
    start_project()
