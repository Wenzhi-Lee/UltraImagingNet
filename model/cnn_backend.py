import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import model_profile

class CnnBackend(nn.Module):

    def __init__(self, gridN, td_num):
        super(CnnBackend, self).__init__()

        self.gridN = gridN
        self.td_num = td_num

        self.conv1 = nn.Conv2d(in_channels=td_num, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gridDown = gridN // (2 * 2 * 2)

        # Position encoding
        self.pe = nn.Parameter(torch.randn(1, self.gridDown, self.gridDown), requires_grad=True)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * self.gridDown * self.gridDown, 256)
        self.fc2 = nn.Linear(256, gridN * gridN)

    def forward(self, x):
        # x: (td_num, gridN, gridN)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Position encoding
        # x: (64, gridDown, gridDown)
        x += self.pe

        # Flatten for multi-head attention
        x = x.view(-1, self.gridDown * self.gridDown).permute(1, 0)

        x, _ = self.attention(x, x, x)

        # Flatten for fully connected layer
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(self.gridN, self.gridN)
    
x = torch.randn(32, 64, 64)
model = CnnBackend(64, 32)
model_profile(model, x)
print(model(x).shape)