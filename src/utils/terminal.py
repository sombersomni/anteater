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
        '--num-episodes',
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
        '--wb-debug',
        type=bool,
        default=False,
        help='The debug flag to enable WandB metrics'
    )
    parser.add_argument(
        '--move-limit',
        type=int,
        default=100,
        help='The maximum number of moves per episode'
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