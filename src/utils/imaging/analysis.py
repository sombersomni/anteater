import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_top_pca_components(image, n_components=5):
    """
    Perform PCA on a grayscale image and visualize the top 5 principal components
    as both 1D line plots and 2D grid images.
    
    Parameters:
    - image: 2D NumPy array (height x width) representing a grayscale image
    - n_components: Number of principal components to visualize (default=5)
    """
    
    # Ensure the image is a 2D array (grayscale)
    if len(image.shape) != 2:
        raise ValueError("Input must be a 2D grayscale image (1 channel).")
    
    height, width = image.shape
    
    # Standardize the image data (zero mean, unit variance) for PCA
    image_standardized = (image - np.mean(image)) / np.std(image)
    
    # Apply PCA, treating rows as samples and pixels in each row as features
    pca = PCA(n_components=n_components)
    pca.fit(image_standardized)  # Shape: (height, width)
    
    # Get the principal components (eigenvectors)
    components = pca.components_  # Shape: (n_components, width)
    
    # Explained variance ratio for context
    explained_variance = pca.explained_variance_ratio_
    
    # Create a figure with two rows: 1D plots (top) and 2D images (bottom)
    plt.figure(figsize=(15, 6))
    
    # First row: 1D line plots of components
    for i in range(n_components):
        plt.subplot(2, n_components, i + 1)
        plt.plot(components[i], color='black')
        plt.title(f'PC {i+1}\nVar: {explained_variance[i]:.3f}')
        plt.xlabel('Pixel Index')
        plt.ylabel('Component Value')
        plt.grid(True)
    
    # Second row: 2D grid images of components
    for i in range(n_components):
        # Reshape the component to a 2D form
        # Since components are (width,) vectors, we need to interpret them spatially
        # For visualization, we'll repeat or reshape based on context
        component_2d = components[i].reshape(1, width)  # Treat as a single row
        plt.subplot(2, n_components, n_components + i + 1)
        plt.imshow(component_2d, cmap='gray', aspect='auto')
        plt.title(f'PC {i+1} (2D)')
        plt.axis('off')  # Hide axes for cleaner display
    
    plt.tight_layout()
    plt.show()

# Example usage with a synthetic image
if __name__ == "__main__":
    import cv2
    from src.utils.imaging.read_image import read_cv2_image
    # Read an image from disk
    image_path = "out/frame_001.png"
    image = read_cv2_image(image_path, cv2.IMREAD_GRAYSCALE)
    # Change to black and white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cast to a float32
    image = image.astype(np.float32)
    # Call the function
    visualize_top_pca_components(image)