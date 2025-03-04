import numpy as np
import matplotlib.pyplot as plt

def pca_image_analysis(image):
    """
    Perform PCA on an image and display 2D reconstructions using the top 5, top 10,
    bottom 5, and bottom 10 principal components.
    
    Parameters:
    image : numpy.ndarray
        A 2D numpy array representing a grayscale image of shape (M, N).
    """
    # Compute the mean row of the image
    mean_row = np.mean(image, axis=0)
    
    # Center the image by subtracting the mean row from each row
    image_centered = image - mean_row
    
    # Perform Singular Value Decomposition (SVD) on the centered image
    U, S, Vt = np.linalg.svd(image_centered, full_matrices=False)
    
    # Reconstruct the image using the top 5 components
    I_top_5 = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :] + mean_row
    
    # Reconstruct the image using the top 10 components
    I_top_10 = U[:, :10] @ np.diag(S[:10]) @ Vt[:10, :] + mean_row
    
    # Reconstruct the image using the bottom 5 components
    I_bottom_5 = U[:, -5:] @ np.diag(S[-5:]) @ Vt[-5:, :] + mean_row
    
    # Reconstruct the image using the bottom 10 components
    I_bottom_10 = U[:, -10:] @ np.diag(S[-10:]) @ Vt[-10:, :] + mean_row
    
    # Create a 2x2 subplot to display the reconstructed images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Display reconstruction with top 5 components
    axs[0, 0].imshow(I_top_5, cmap='gray')
    axs[0, 0].set_title('Top 5 Components')
    
    # Display reconstruction with top 10 components
    axs[0, 1].imshow(I_top_10, cmap='gray')
    axs[0, 1].set_title('Top 10 Components')
    
    # Display reconstruction with bottom 5 components
    axs[1, 0].imshow(I_bottom_5, cmap='gray')
    axs[1, 0].set_title('Bottom 5 Components')
    
    # Display reconstruction with bottom 10 components
    axs[1, 1].imshow(I_bottom_10, cmap='gray')
    axs[1, 1].set_title('Bottom 10 Components')
    
    # Remove axes for better visualization
    for ax in axs.flat:
        ax.axis('off')
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Create a random grayscale image of size 100x100
    image = np.random.randint(0, 256, (100, 100)).astype(np.float32)
    
    # Perform PCA analysis on the image
    pca_image_analysis(image)