import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ResNet16 import build_model
from utils.logs import setup_logger
# Assuming the previous model code is available


logger = setup_logger("Training", f"{__name__}.log")


def train_resnet16(
    model,
    train_loader,
    num_epochs,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the ResNet16 model
    
    Args:
        model: ResNet16 model instance
        train_loader: DataLoader with training data (images, labels)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (default: 0.001)
        device: Device to train on (default: cuda if available, else cpu)
    """
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    # Note: NLLLoss expects log probabilities, so we'll add log_softmax in the training loop
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        logger.info(f"Starting epoch: {epoch+1}")
        for i, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Ensure images have correct shape [batch_size, 1, height, width]
            if images.dim() == 3:  # If images are [batch_size, height, width]
                images = images.unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            # Apply log_softmax for NLLLoss
            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(log_probs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
        
        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}')

    print('Training finished!')
    return model

# Example usage with dummy data
def main():
    from torch.utils.data import TensorDataset
    # Create model
    model = build_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create dummy dataset (replace with your actual data)
    # Dummy data: 1000 samples of 32x32 grayscale images with 4 possible actions
    dummy_images = torch.randn(1000, 1, 32, 32)
    dummy_labels = torch.randint(0, 4, (1000,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
    
    # Train the model
    trained_model = train_resnet16(
        model=model,
        train_loader=train_loader,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Example inference
    trained_model = trained_model.to(device)
    trained_model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 32, 32).to(device)
        output = trained_model(test_input)
        action = torch.argmax(output, dim=1).item()
        action_map = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        print(f"Predicted action: {action_map[action]}")

if __name__ == "__main__":
    main()