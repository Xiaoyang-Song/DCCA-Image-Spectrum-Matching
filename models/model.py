import torch
import torch.nn as nn
import numpy as np
from models.window_lstm import *
from models.dcca import *


class WindowDCCA(nn.Module):
    def __init__(self, img_input_dim, img_hidden_dim, img_num_layers,
                 spec_input_dim, spec_hidden_dim, spec_num_layers,
                 dcca_out_dim, dcca_img_layers, dcca_spec_layers):
        super().__init__()
        # Define Image WindowLSTM
        self.img_input_dim = img_input_dim
        self.img_hidden_dim = img_hidden_dim
        self.img_num_layers = img_num_layers

        self.ImgLSTM = WindowLSTM(
            self.img_input_dim, self.img_hidden_dim, self.img_num_layers)

        # Define Spectra WindowLSTM
        self.spec_input_dim = spec_input_dim
        self.spec_hidden_dim = spec_hidden_dim
        self.spec_num_layers = spec_num_layers

        self.SpecLSTM = WindowLSTM(
            self.spec_input_dim, self.spec_hidden_dim, self.spec_num_layers)

        # Define DCCA network
        self.dcca_out_dim = dcca_out_dim
        self.dcca_img_layers = dcca_img_layers
        self.dcca_spec_layers = dcca_spec_layers
        self.img_branch = dict(
            input_dim=self.img_hidden_dim,
            layers=self.dcca_img_layers,
            output_dim=self.dcca_out_dim
        )
        self.spec_branch = dict(
            input_dim=self.spec_hidden_dim,
            layers=self.dcca_spec_layers,
            output_dim=self.dcca_out_dim
        )
        self.DCCA = DCCA(self.img_branch, self.spec_branch)

    def forward(self, img, spec):
        # Forward pass of LSTM
        # ic(img)
        out_img, hn_img, cn_img = self.ImgLSTM(img)
        out_spec, hn_spec, cn_spec = self.SpecLSTM(spec)
        # Process hidden state
        hn_img = hn_img.squeeze().unsqueeze(0)
        hn_spec = hn_spec.squeeze().unsqueeze(0)
        # DCCA Forward Pass
        out1, out2 = self.DCCA(hn_img, hn_spec)
        return out1, out2
