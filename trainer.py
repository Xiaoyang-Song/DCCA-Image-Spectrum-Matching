import torch.nn as nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from objective.loss import CCA
from models.dcca import *
from dataset import *
from icecream import ic


def train(model, tri_dset, val_dset, optimizer,
          max_epoch, writer):
    ic("Training begins")
    for epoch in max_epoch:
        for (img, spec) in tri_dset:
            img = img.unsqueeze(0).to(DEVICE)
            spec = spec.unsqueeze(0).to(DEVICE)
            # Feed in LSTM

            out1, out2 = model(img, spec)

if __name__ == '__main__':
    ic("Training...")
