import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


def get_reward_action_heatmaps(
    reward_dict: Dict[Tuple[int, int], float],
    num_actions: int,
    grid_size: int=4
):
    """
    Plots a heatmap for each action based on a reward dictionary.
    
    Args:
        reward_dict: Dictionary with (state, action) tuples as keys and float rewards as values.
        num_actions: Number of possible actions (e.g., 2 for left/right, 4 for up/down/left/right).
    """
    # Initialize a 4x4 grid for each action
    heatmaps = [np.zeros((grid_size, grid_size)) for _ in range(num_actions)]  
    # Populate the grids with rewards
    for (state, action), reward in reward_dict.items():
        if 0 <= state <= 15:  # Ensure state is valid for 4x4 grid
            # Convert state to row, col (0 at top-left, 15 at bottom-right)
            row = state // grid_size  # Integer division for row
            col = state % grid_size   # Modulo for column
            heatmaps[action][row, col] = reward
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, num_actions, figsize=(5 * num_actions, 5))
    if num_actions == 1:
        axes = [axes]  # Ensure axes is iterable for single action
    
    for action in range(num_actions):
        sns.heatmap(
            heatmaps[action],
            annot=True,  # Show reward values on the heatmap
            fmt=".2f",   # Format numbers to 2 decimal places
            cmap="YlGnBu",  # Color scheme (yellow-green-blue)
            ax=axes[action],
            vmin=min(reward_dict.values()),  # Consistent scale across heatmaps
            vmax=max(reward_dict.values())
        )
        axes[action].set_title(f"Action {action}")
        axes[action].set_xlabel("Column")
        axes[action].set_ylabel("Row")
    return fig


# Example usage
if __name__ == "__main__":
    # Sample reward dictionary: (state, action) -> reward
    sample_rewards = {
        (0, 0): 1.5,   # State 0, Action 0
        (1, 0): 2.0,
        (14, 0): -1.0,
        (15, 0): 3.5,
        (5, 1): 0.8,   # State 5, Action 1
        (10, 1): 2.2,
        (15, 1): -0.5
    }
    
    # Assume 2 actions for this example
    fig = get_reward_action_heatmaps(sample_rewards, num_actions=2)
    plt.tight_layout()
    plt.show()