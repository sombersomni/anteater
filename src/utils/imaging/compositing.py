import numpy as np
from PIL import Image

def create_heatmap_composite(image, reward_gradient_field, alpha=0.6):
    """
    Creates an alpha composite of a game screenshot with a reward gradient heatmap overlay.
    
    Args:
        image: numpy array of shape (H, W, 3) representing the game screenshot (RGB)
        reward_gradient_field: numpy array of shape (h, w) representing the reward scalar field
        alpha: float between 0 and 1, opacity of the heatmap overlay (default 0.6)
    
    Returns:
        numpy array of shape (H, W, 3) with the composite image
    """
    # Convert input image to PIL Image if it isn't already
    if isinstance(image, np.ndarray):
        base_img = Image.fromarray(image.astype('uint8'))
    else:
        base_img = image

    # Get dimensions of the base image
    target_width, target_height = base_img.size
    
    # Normalize the reward gradient field to 0-1 range
    gradient_norm = (reward_gradient_field - np.min(reward_gradient_field)) / \
                   (np.max(reward_gradient_field) - np.min(reward_gradient_field) + 1e-8)
    
    # Create a colormap (red for high values, blue for low)
    # Using a simple red-blue gradient
    heatmap = np.zeros((*gradient_norm.shape, 3))
    heatmap[..., 0] = gradient_norm  # Red channel
    heatmap[..., 2] = 1 - gradient_norm  # Blue channel
    
    # Convert to 0-255 range
    heatmap = (heatmap * 255).astype('uint8')
    
    # Create PIL Image from heatmap and resize to match base image
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
    # Convert base image and heatmap to RGBA
    base_img = base_img.convert('RGBA')
    heatmap_img = heatmap_img.convert('RGBA')
    
    # Create the composite image
    composite = Image.blend(base_img, heatmap_img, alpha)
    
    # Convert back to RGB and numpy array
    composite_rgb = composite.convert('RGB')
    return np.array(composite_rgb)

# Example usage:
if __name__ == "__main__":
    # Create sample data
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    sample_gradient = np.random.rand(100, 100)
    
    # Create composite
    result = create_heatmap_composite(sample_image, sample_gradient)
    
    # Display or save the result
    Image.fromarray(result).show()
    # Image.fromarray(result).save('composite_output.png')
