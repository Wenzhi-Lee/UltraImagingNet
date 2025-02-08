import torch.nn as nn
import torch.nn.functional as F
import torch

from model.utils import model_profile

class FcnBackend(nn.Module):

    def __init__(self, gridN, feature_dim):
        super(FcnBackend, self).__init__()

        self.gridN = gridN

        self.conv1 = nn.Conv2d(feature_dim, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(32, 48, 3, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)

        # self.conv3 = nn.Conv2d(48, 128, 3, padding=1)
        # self.pool3 = nn.MaxPool2d(2, 2)

        # self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # self.pool4 = nn.MaxPool2d(2, 2)

        # self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(48, 32, 4, stride=4)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, stride=4)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (td_num, gridN, gridN)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # x = F.relu(self.conv3(x))
        # x = self.pool3(x)

        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)

        # x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        out = torch.softmax(self.final_conv(x), dim=1)

        return out
