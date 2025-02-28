import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from models.ResNet16 import build_reward_model
from utils.logs import setup_logger


# Set up logger
logger = setup_logger("Training", f"{__name__}.log")


# Function to compute weighted average of weights (same as before)
def compute_weighted_average(checkpoints, weights):
    """
    Args:
        checkpoints: Dict of {step: state_dict} from saved models
        weights: List or dict of weights corresponding to each checkpoint (sums to 1)
    Returns:
        averaged_state_dict: Averaged weights
    """
    if len(checkpoints) != len(weights):
        raise ValueError("Number of checkpoints must match number of weights")
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    
    averaged_state_dict = OrderedDict()
    print(checkpoints)
    first_state_dict = list(checkpoints.values())[0]
    for key in first_state_dict.keys():
        averaged_state_dict[key] = torch.zeros_like(first_state_dict[key])
    
    for (step, state_dict), w in zip(checkpoints.items(), weights):
        for key in state_dict.keys():
            averaged_state_dict[key] += w * state_dict[key].float()
    
    return averaged_state_dict

# Training loop with weight extraction
def train_and_store_weights(model, num_steps=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Store checkpoints
    checkpoints = {}
    
    # Dummy data (replace with your dataset)
    dummy_inputs = torch.randn(100, 1, 64, 64)  # ResNet-18 expects 3 channels
    dummy_rewards = torch.rand(100, 1)
    dataset = TensorDataset(dummy_inputs, dummy_rewards)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for step in range(num_steps):
        model.train()
        for inputs, rewards in loader:
            inputs, rewards = inputs.to(device), rewards.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, rewards)
            loss.backward()
            optimizer.step()
            break  # Just one batch per step for demo
        
        # Extract and store weights after each step
        checkpoints[step] = copy.deepcopy(model.state_dict())
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    
    return checkpoints

# Main execution
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    model = build_reward_model()
    model.to(device)
    # Train and get checkpoints
    checkpoints = train_and_store_weights(model)
    
    # Define weights for averaging (e.g., exponential decay)
    num_steps = len(checkpoints)
    weights = [2 ** (-(num_steps - 1 - i)) for i in range(num_steps)]  # [0.125, 0.25, 0.5, 1, 2]
    print(f"Weights: {weights}")
    
    # Compute weighted average
    averaged_weights = compute_weighted_average(checkpoints, weights) 
    # Load averaged weights into the model
    model.load_state_dict(averaged_weights)
    # Inference with averaged weights
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 64, 64).to(model.device)
        logits = model(test_input)
        reward = model.post_process(logits).item()
        print(f"Predicted reward with averaged weights: {reward:.4f}")

if __name__ == "__main__":
    main()