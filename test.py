import cv2
import numpy as np
from src.utils.imaging.analysis import visualize_top_pca_components
from src.utils.imaging.read_image import read_cv2_image
# Read an image from disk
image_path = "out/megaman.jpg"
image = read_cv2_image(image_path, cv2.IMREAD_GRAYSCALE)
# cast to a float32
image = image.astype(np.float32)
# Call the function
visualize_top_pca_components(image)