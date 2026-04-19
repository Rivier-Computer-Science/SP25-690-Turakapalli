import torch
import torch.nn as nn


class ResidualCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return residual + x
