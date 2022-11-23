import torch
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class WindowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        # Config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Model Declaration
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x):
        # For image LSTM, input features should be of shape (B, L, hidden_dim)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.input_dim
        # Forward pass
        output, (hn, cn) = self.lstm(x)
        # hn, cn: (B, self.num_layers, self.hidden_dim)
        return output, hn, cn


if __name__ == '__main__':
    ic("Processing window of images")
    # Simple sanity check
    lstm = WindowLSTM(6, 64, 1)
    x = torch.zeros((1,11,6))
    output, hn, cn = lstm(x)
    ic(output.shape)
    ic(hn.squeeze().unsqueeze(0).shape)
    ic(cn.shape)
