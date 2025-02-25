import argparse
# Creates parsed arguments for the terminal
# Create a parser for getting the output directory and the input directory
# from the terminal
def create_arg_parser(
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
    return parser
