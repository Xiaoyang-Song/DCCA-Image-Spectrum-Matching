import torch
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImgLSTM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    ic("Processing window of images")
