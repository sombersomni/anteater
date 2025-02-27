import os
import cv2
from src.utils.file_tools import search_files
from src.utils.terminal import create_playback_arg_parser
from src.utils.logs import setup_logger


# Set up the logger
logger = setup_logger("Playback", f"{__name__}.log")

# Count the number of frames saved
output_dir = 'out'

def playback_images(
    images_dir: str,
    fps: int = 12
):
    if not os.path.isdir(images_dir):
        raise ValueError(f"Folder '{images_dir}' does not exist")
    files_found = search_files(images_dir, '*.png')
    iterations = len(files_found)
    if iterations == 0:
        raise ValueError(f"No frames found in '{images_dir}'")
    logger.info(f"Found {iterations} frames in '{images_dir}'")
    # Replay the saved frames
    for file_path in files_found:
        frame = cv2.imread(file_path)
        if frame is None:
            logger.error(f"Failed to read frame from '{file_path}'")
            continue
        cv2.imshow('Render', frame)
        cv2.waitKey(fps)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = create_playback_arg_parser()
    args = parser.parse_args()
    playback_images(args.images_dir, args.fps)
