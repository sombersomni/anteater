import cv2
import numpy as np

def image_difference(
    image1_path,
    image2_path,
    output_path=None,
    use_squared_diff=True,
    threshold=30
):
    """
    Calculate and display the difference between two images using OpenCV.
    
    Args:
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image
        output_path (str, optional): Path to save the difference image. If None, image is only displayed.
    
    Returns:
        numpy.ndarray: The difference image
    """
    try:
        # Read the images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        # Check if images loaded successfully
        if img1 is None or img2 is None:
            raise ValueError("One or both images failed to load. Check the file paths.")

        # Ensure images are the same size
        if img1.shape != img2.shape:
            # Resize second image to match first image's dimensions
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        diff = cv2.absdiff(img1, img2) ** 2   
        diff = np.clip(diff, 0, 255).astype(np.uint8)   
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        # Save the difference image if output_path is provided
        if output_path:
            cv2.imwrite(output_path, diff)
            print(f"Difference image saved to: {output_path}")
        return diff
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual image paths
    img1_path = "out\Agent-v1_0_run_001.png"
    img2_path = "out\Agent-v1_0_run_002.png"
    output_path = "difference.jpg"  # Optional
    difference = image_difference(img1_path, img2_path, output_path)
    if difference is not None:
        cv2.imshow("Difference", difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()