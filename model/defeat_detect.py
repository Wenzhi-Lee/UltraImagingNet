import torch.nn as nn
import torch

from model.wavelet import WaveletFeature
from model.feature_merge import FeatureMerge
from model.cnn_backend import CnnBackend
from model.fcn_backend import FcnBackend
from model.pure_phase import PurePhase

class DefeatDetectModel(nn.Module):

    def __init__(self, td_num, gridN, freq_dim):
        super(DefeatDetectModel, self).__init__()

        self.td_num = td_num
        self.gridN = gridN
        self.freq_dim = freq_dim
        self.feature_dim = 32

        self.wavelet = WaveletFeature(freq_dim, self.feature_dim, 2)
        self.feature_merge = PurePhase(gridN, td_num)
        self.backend = CnnBackend(gridN, self.feature_dim)

    def forward(self, wave, phase):
        # wave: 
        wave_feature = self.wavelet(wave)

        # merge
        feature = self.feature_merge(wave_feature, phase)

        pred = self.backend(feature)

        return pred
