import cv2
import numpy as np
from src.utils.imaging.analysis import pca_image_analysis
from src.utils.imaging.read_image import read_cv2_image
# Read an image from disk
image_path = "out\Agent-v1_0_run_003.png"
image = read_cv2_image(image_path, cv2.IMREAD_GRAYSCALE)
# cast to a float32
image = image.astype(np.float32)
# Call the function
pca_image_analysis(image)