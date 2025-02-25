import os
import cv2
from src.utils.file_tools import count_matching_files


# Count the number of frames saved
output_dir = 'out'
if not os.path.isdir(output_dir):
    raise ValueError(f"Folder '{output_dir}' does not exist")
iterations = count_matching_files(output_dir, '*.png', 'extension')
if iterations == 0:
    raise ValueError(f"No frames found in '{output_dir}'")
print(f"Found {iterations} frames in '{output_dir}'")
# Replay the saved frames
for i in range(iterations):
    frame = cv2.imread(
        os.path.join(
            'out',
            f'FrozenLake-v1_{i:04d}.png'
        )
    )
    cv2.imshow('Render', frame)
    cv2.waitKey(100)
cv2.destroyAllWindows()