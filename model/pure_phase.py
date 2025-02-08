import torch.nn as nn
import torch

class PurePhase(nn.Module):

    def __init__(self, gridN, td_num):
        super(PurePhase, self).__init__()

        self.gridN = gridN
        self.td_num = td_num

    def forward(self, wavelet_feature, phase):
        # wavelet_feature: (td_num, feature_dim)
        # phase: (td_num, gridN, gridN)

        return phase
