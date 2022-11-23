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
    criterion = CCA()

    model.train()

    tri_iter_count, val_iter_count = 0, 0
    for epoch in max_epoch:
        for step, (img, spec) in enumerate(tri_dset):
            img = img.unsqueeze(0).to(DEVICE)
            spec = spec.unsqueeze(0).to(DEVICE)

            # Zero Existing Gradients
            optimizer.zero_grad()
            # WindowDCCA Forward Pass
            out1, out2 = model(img, spec)
            loss = criterion(out1, out2)
            # Backward Pass
            loss.backward()
            # Gradient Update
            optimizer.step()

            # Log statistics
            writer.add_scalar("Loss/Train", loss.detach(), tri_iter_count)
            tri_iter_count += 1
            print(
                f"Epoch {epoch:<3} | Step {step:<4} | Train Loss: {loss:.5f}")

        # Evaluation
        with torch.no_grad:
            for step, (img, spec) in enumerate(val_dset):
                img = img.unsqueeze(0).to(DEVICE)
                spec = spec.unsqueeze(0).to(DEVICE)
                out1, out2 = model(img, spec)
                loss = criterion(out1, out2)
                # Log validation statistics
                writer.add_scalar("Loss/Eval", loss.detach(), val_iter_count)
                val_iter_count += 1
                print(
                    f"Epoch {epoch:<3} | Step {step:<4} | Eval Loss: {loss:.5f}")


if __name__ == '__main__':
    ic("Training...")
