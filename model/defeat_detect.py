import torch.nn as nn
import torch

from model.cnn_backend import CnnBackend

class DefeatDetectModel(nn.Module):

    def __init__(self, gridN):
        super(DefeatDetectModel, self).__init__()
        self.gridN = gridN

        self.cnn = CnnBackend(gridN)

    def forward(self, phase):

        pred = self.cnn(phase)

        return pred
