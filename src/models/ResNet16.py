import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet Model
class ResNet16(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet16, self).__init__()
        self.in_channels = 64 
        # Initial convolution layer - input is 1 channel (black and white)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        # Global average pooling and final fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Create the model
def ResNet16_4actions():
    # 3 blocks with [2, 2, 2] residual units + initial conv + final fc
    # Approximately 16 layers total:
    # 1 (initial conv) + (2+2+2 blocks * 2 conv each) + 1 (fc) + pooling/batchnorm layers
    return ResNet16(BasicBlock, [2, 2, 2])

# Example usage
def main():
    # Initialize model
    model = ResNet16_4actions()
    
    # Example input: batch_size=4, channels=1 (B&W), height=32, width=32
    sample_input = torch.randn(4, 1, 32, 32)
    output = model(sample_input)
    
    print("Model architecture:")
    print(model)
    print("\nOutput shape:", output.shape)  # Should be [4, 4] for 4 samples, 4 actions
    print("Output example:", output[0])

if __name__ == "__main__":
    main()