import torch.nn as nn
import torch

class FeatureMerge(nn.Module):

    def __init__(self, gridN, td_num):
        super(FeatureMerge, self).__init__()

        self.gridN = gridN
        self.td_num = td_num

        self.weight = nn.Parameter(torch.randn(td_num, gridN, gridN), requires_grad=True)

    def forward(self, wavelet_feature, phase):
        # wavelet_feature: (td_num, feature_dim)
        # phase: (td_num, gridN, gridN)

        phase_weighted = torch.sigmoid(phase * self.weight)

        # f_output: (feature_dim, gridN, gridN)
        f_output = torch.zeros(
            (wavelet_feature.size(1), self.gridN, self.gridN),
            device=wavelet_feature.device
        )

        for b in range(self.td_num):
            f_output += phase_weighted[b] * wavelet_feature[b].unsqueeze(-1).unsqueeze(-1)

        f_output /= self.td_num

        return f_output
