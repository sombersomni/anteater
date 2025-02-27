import cv2
import os
import numpy as np
from typing import List, Tuple, Iterable
from src.utils.logs import setup_logger


logger = setup_logger("Image Storage", f"{__name__}.log")


def save_multiple_cv2_images(
    images: Iterable[np.ndarray],
    output_dir: str = "out",
    base_name: str = "image",
    extension: str = ".jpg",
    start_index: int = 0,
    quality: int = 95
) -> bool:
    """
    Save a list of cv2 numpy array images to disk.
    
    Args:
        images: List of numpy arrays representing images in cv2 format
        output_dir: Directory where images will be saved (created if doesn't exist)
        base_name: Base name for files (e.g., "image_000.jpg")
        extension: File extension (e.g., ".jpg", ".png")
        start_index: Starting number for sequential naming
        quality: JPEG quality (0-100) if applicable
        
    Returns:
        bool: success_status
    """
    # Ensure extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension
        
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {str(e)}")
        return False, []
    
    saved_paths = []
    all_success = True
    jpeg_format = extension.lower() in [".jpg", ".jpeg"]
    
    # Process each image
    for idx, img in enumerate(images, start=start_index):
        full_path = os.path.join(output_dir, f"{base_name}_{idx:03d}{extension}")
        
        if img is None or img.size == 0:
            print(f"Skipping invalid image at index {idx}")
            all_success = False
            continue
            
        try:
            success = (
                cv2.imwrite(
                    full_path,
                    img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                ) if jpeg_format else cv2.imwrite(full_path, img)
            )
            if success:
                saved_paths.append(full_path)
            else:
                print(f"Failed to save image {full_path}")
                all_success = False        
        except Exception as e:
            print(f"Error saving {full_path}: {str(e)}")
            all_success = False
    
    if all_success:
        logger.info(f"Successfully saved {len(saved_paths)} images to {output_dir}")
    elif saved_paths:
        logger.info(f"Partially successful: saved {len(saved_paths)} of {len(images)} images")
    else:
        logger.info("Failed to save any images")
   
    return all_success

# Example usage:
if __name__ == "__main__":
    # Create sample images
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    images = [img1, img2]
    
    # Save the images
    success, paths = save_multiple_cv2_images(
        images,
        output_dir="test_output",
        base_name="frame",
        extension=".png",
        start_index=1
    )
    
    print(f"Success: {success}")
    print(f"Saved paths: {paths}")