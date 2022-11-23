import torch
import torch.nn as nn
import numpy as np
from icecream import ic

# Credit: part (about 2/3) of the code is borrowed from the following Github Repo:
# https://github.com/Michaelvll/DeepCCA/blob/master/DeepCCAModels.py


# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_layers(layer_dims):
    layers = []
    for l_id in range(len(layer_dims) - 1):
        if l_id == len(layer_dims) - 2:
            layers.append(nn.Sequential(
                nn.BatchNorm1d(
                    num_features=layer_dims[l_id], affine=False),
                nn.Linear(layer_dims[l_id], layer_dims[l_id + 1]),
            ))
        else:
            layers.append(nn.Sequential(
                nn.Linear(layer_dims[l_id], layer_dims[l_id + 1]),
                nn.Sigmoid(),
                nn.BatchNorm1d(
                    num_features=layer_dims[l_id + 1], affine=False),
            ))
    return layers


class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims: list, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = [self.input_dim] + layer_dims + [self.output_dim]
        # Model Declaration
        self.layers = make_layers(self.layer_dims)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class DCCA(nn.Module):
    def __init__(self, img: dict, spec: dict, device=DEVICE):
        super().__init__()
        # Extract branch information: img
        self.img_in_dim = img['input_dim']
        self.img_layers = img['layers']
        self.img_out_dim = img['output_dim']
        # Extract branch information: spec
        self.spec_in_dim = spec['input_dim']
        self.spec_layers = spec['layers']
        self.spec_out_dim = spec['output_dim']
        # Assertion check
        assert self.img_out_dim == self.spec_out_dim

        # Models
        self.img_model = MLP(self.img_in_dim, self.img_layers,
                             self.img_out_dim)
        self.spec_model = MLP(self.spec_in_dim, self.spec_layers,
                              self.spec_out_dim)

    def forward(self, img, spec):

        # Shape: B x output_dim
        out1 = self.img_model(img)
        out2 = self.spec_model(spec)

        return out1, out2
