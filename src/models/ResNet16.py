import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List
from src.utils.logs import setup_logger


logger = setup_logger("ResNet16", f"{__name__}.log")


# Basic Residual Block (same as before)
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
        logger.debug(f"Input shape: {x.shape}")
        out = F.relu(self.bn1(self.conv1(x)))
        logger.debug(f"After conv1 shape: {out.shape}")
        out = self.bn2(self.conv2(out))
        logger.debug(f"After conv2 shape: {out.shape}")
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Reward Prediction ResNet
class ResNet16(nn.Module):
    """
    General ResNet class for various configurations.
    """
    def __init__(
        self,
        block,
        num_blocks: List[int],
        out_channels: int = 1,  # Single output for reward
    ):
        super(ResNet16, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_block_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_block_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_block_layer(block, 256, num_blocks[2], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256 * block.expansion, out_channels)  # Single output for reward

    def _make_block_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits
    
    def post_process(self, logits: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(logits)

    def preprocess(
        self,
        observation: np.ndarray,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.Tensor:
        # The model will except observation of shape (batch_size, 1, 16, 16)
        C, H, W = observation.shape
        img_tensor = torch.from_numpy(
            observation
        ).unsqueeze(0)
        print(f"Image tensor shape: {img_tensor.shape}")
        img_tensor = torch.permute(
            img_tensor,
            (0, 3, 1, 2)
        )
        # Sum along the channels and normalize
        img_tensor = torch.sum(img_tensor, dim=1, keepdim=True) / C
        return img_tensor.to(dtype=torch.float32)


class RewardResNet16(ResNet16):
    def __init__(self, block, num_blocks, out_channels=1):
        super(RewardResNet16, self).__init__(
            block, num_blocks,
            out_channels
        )

    def post_process(self, logits: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(logits)

class ActionResNet16(ResNet16):
    def __init__(self, block, num_blocks, out_channels=1):
        super(ActionResNet16, self).__init__(
            block, num_blocks,
            out_channels
        )

    def post_process(self, logits) -> torch.Tensor:
        return F.softmax(logits, dim=1)


def build_reward_model():
    return RewardResNet16(BasicBlock, [2, 2, 2])

def build_action_model():
    return ActionResNet16(BasicBlock, [2, 2, 2], out_channels=4)
