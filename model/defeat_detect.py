import torch.nn as nn
import torch

from wavelet import WaveletFeature
from feature_merge import FeatureMerge
from cnn_backend import CnnBackend

class DefeatDetectModel(nn.Module):

    def __init__(self, td_num, gridN, freq_dim):
        super(DefeatDetectModel, self).__init__()

        self.td_num = td_num
        self.gridN = gridN
        self.freq_dim = freq_dim

        self.wavelet = WaveletFeature(td_num, freq_dim)
        self.feature_merge = FeatureMerge(td_num, freq_dim)
        self.cnn_backend = CnnBackend(gridN, td_num)

    def forward(self, wave, phase):
        # wave: 
        
