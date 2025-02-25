import cv2
import os
import sys
import gymnasium as gym
import pygame
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from src.utils.keyboard import handle_keyboard_input, ActionState
from src.utils.file_tools import apply_action_to_files
import src.utils.terminal as tu
from src.utils.logs import setup_logger



def start_project():
    # Set up the logger
    logger = setup_logger("FrozenLake", "frozenlake.log")
    # Create the environment
    env_id = 'FrozenLake-v1'
    env = FrozenLakeEnv(
        desc=generate_random_map(),
        render_mode='human',
        is_slippery=False
    )
    env.reset()
    init_frame = env.render()
    # Reset the environment to generate the first observation
    parser = tu.create_arg_parser()
    args = parser.parse_args()
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    apply_action_to_files(
        args.output_dir,
        pattern='*.png',
        file_action=os.remove
    )
    idx = 0
    threshold = 10
    episodes = 1
    try:
        for episode in range(episodes):
            done = False
            t = 0
            reward = 0
            env.reset()
            logger.info(f"Episode: {episode}")
            while not done or t < threshold:
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                action = handle_keyboard_input({
                    pygame.K_LEFT: 0,
                    pygame.K_DOWN: 1,
                    pygame.K_RIGHT: 2,
                    pygame.K_UP: 3,
                })
                if action.ended:
                    break
                if action.value == -1:
                    logger.info("No action was selected. Skipping...")
                    continue
                next_observation, reward, terminated, truncated, info = env.step(action.value)
                logger.info(f"Action, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                # Get the current frame to render the environment
                frame = np.transpose(
                    pygame.surfarray.array3d(env.window_surface),
                    axes=(1, 0, 2)
                )
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                logger.info(f"Frame shape: {frame.shape}")
                # Save the frame to disk
                frame_idx_with_pading = f'{idx:04d}'
                cv2.imwrite(
                    os.path.join(
                        args.output_dir,
                        f'{env_id}_{frame_idx_with_pading}.png'
                    ),
                    frame_bgr
                )
                # If the episode has ended then we can reset to start a new episode
                if terminated:
                    logger.info("Episode terminated.")
                    break
                # observation = next_observation
                idx += 1

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.warning(f"Error: {e}")
        sys.exit(1)
    finally:
        print("Environment and pygame closed.")
        env.close()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == '__main__':
    start_project()
