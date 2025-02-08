import torch
import torch.nn as nn

class WaveletFeature(nn.Module):
    
    def __init__(self, freq_dim, feature_dim, num_layers):
        super(WaveletFeature, self).__init__()
        
        # Use LSTM to extract features from wavelet transformed data
        self.lstm = nn.LSTM(
            input_size=freq_dim,
            hidden_size=feature_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Use attention to extract features from LSTM output
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads=4, batch_first=True
        )

    def forward(self, x):
        # x: (td_num, time_dim, freq_dim)
        lstm_out, _ = self.lstm(x)
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attention_out.mean(dim=1)
