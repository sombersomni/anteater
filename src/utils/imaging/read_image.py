import cv2
import numpy as np
from typing import Optional
from src.utils.logs import setup_logger


logger = setup_logger("Image Reading", f"{__name__}.log")


def read_cv2_image(file_path: str, color_mode: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Read an image from disk using cv2 and return it as a numpy array.
    
    Args:
        file_path: String path to the image file
        color_mode: cv2 color mode flag (default: cv2.IMREAD_COLOR for BGR)
                   Other options: cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
    
    Returns:
        numpy.ndarray: Image as numpy array if successful, None if failed
    """
    try:
        # Read the image from disk
        image = cv2.imread(file_path, color_mode)
        
        # Check if image was successfully loaded
        if image is None:
            logger.info(f"Error: Could not load image from {file_path}")
            return None
            
        # Verify the image is a valid numpy array
        if not isinstance(image, np.ndarray) or image.size == 0:
            logger.warning(f"Error: Invalid image data from {file_path}")
            return None
            
        return image
        
    except FileNotFoundError:
        logger.warning(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logger.warning(f"Error reading image from {file_path}: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    # Read an image in default BGR color mode
    img_color = read_cv2_image("test_output/frame_001.png")
    if img_color is not None:
        logger.debug(f"Loaded color image with shape: {img_color.shape}")
    
    # Read an image in grayscale
    img_gray = read_cv2_image("test_output/frame_001.png", cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
        logger.debug(f"Loaded grayscale image with shape: {img_gray.shape}")

