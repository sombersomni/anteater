import argparse
# Creates parsed arguments for the terminal
# Create a parser for getting the output directory and the input directory
# from the terminal
def create_gym_arg_parser(
    output_dir: str='out',
    description: str='Aggregates additional data useful for training'
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--output_dir',
        type=str,
        default=output_dir,
        help='The directory where the gym render images for your episodes will be saved'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='The number of episodes to run the simulation'
    )
    parser.add_argument(
        '--discount-factor',
        type=float,
        default=0.9,
        help='The discount factor for the reward function (gamma)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='The learning rate for the reward function'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='The epsilon value for the epsilon-greedy policy if applicable'
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='The debug flag to enable WandB metrics'
    )
    parser.add_argument(
        '--move-limit',
        type=int,
        default=30,
        help='The maximum number of moves per episode'
    )
    parser.add_argument(
        '--render-mode',
        type=str,
        default='rgb_array',
        help='The rendering mode for the gym environment'
    )
    return parser


def create_playback_arg_parser(
    images_dir: str='out',
    description: str='Passes config data to the playback script'
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--images-dir',
        type=str,
        default=images_dir,
        help='The directory used to read and display the images'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='The number of frames per second to play the images'
    )

    return parser