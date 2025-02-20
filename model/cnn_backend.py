import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import model_profile
from .cbam import CBAM
from .residual_block import ResidualBlock

class CnnBackend(nn.Module):

    def __init__(self, gridN):
        super(CnnBackend, self).__init__()

        self.gridN = gridN

        # (n, 1, 512, 512) -> (n, 16, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cbam1 = CBAM(16)

        # (n, 16, 128, 128) -> (n, 32, 64, 64)
        self.res16 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            CBAM(16),
            ResidualBlock(16, 32, stride=2, downsample=nn.Conv2d(16, 32, 1, 2))
        )


        # (n, 32, 64, 64) -> (n, 64, 32, 32)
        self.res32 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            CBAM(32),
            ResidualBlock(32, 64, stride=2, downsample=nn.Conv2d(32, 64, 1, 2))
        )

        # (n, 64, 32, 32) -> (n, 128, 16, 16)
        self.res64 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            CBAM(64),
            ResidualBlock(64, 128, stride=2, downsample=nn.Conv2d(64, 128, 1, 2))
        )

        # (n, 128, 16, 16) -> (n, 8, 8, 8)
        self.res128 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            CBAM(128)
        )
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, gridN, gridN)
        
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.cbam1(x)

        x = self.res16(x)
        x = self.res32(x)
        x = self.res64(x)
        x = self.res128(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)

        return x
