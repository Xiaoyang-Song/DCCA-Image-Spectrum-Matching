import torch
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

DEVICE = 'cuda' if torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 16, kernel_size=(7, 7), padding=(3, 3))
        self.maxpool2d1 = nn.MaxPool2d((3, 3), return_indices=True)
        self.conv2d2 = nn.Conv2d(16, 8, kernel_size=(7, 7), padding=(3, 3))
        self.maxpool2d2 = nn.MaxPool2d((3, 3), return_indices=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=17496, out_features=2560)
        self.linear2 = nn.Linear(in_features=2560, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        batch_size = x.shape[0]
        output = F.relu(self.conv2d1(x))
        output, indices1 = self.maxpool2d1(output)
        output = F.relu(self.conv2d2(output))
        output, indices2 = self.maxpool2d2(output)
        output = self.flatten(output)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        return output, indices1, indices2


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 2560)
        self.linear4 = nn.Linear(2560, 17496)
        self.maxunpool2d1 = nn.MaxUnpool2d((3, 3))
        self.conv2d1 = nn.ConvTranspose2d(
            8, 16, kernel_size=(7, 7), padding=(3, 3))
        self.maxunpool2d2 = nn.MaxUnpool2d((3, 3))
        self.conv2d2 = nn.ConvTranspose2d(
            16, 1, kernel_size=(7, 7), padding=(3, 3))
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, indices1, indices2):
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        output = output.view((output.shape[0], 8, 27, 81))
        output = self.maxunpool2d1(
            output, indices2, output_size=torch.Size([output.shape[0], 8, 83, 243]))
        output = F.relu(self.conv2d1(output))
        output = self.maxunpool2d2(output, indices1, output_size=torch.Size([
                                   output.shape[0], 16, 250, 730]))
        output = self.conv2d2(output)
        output = self.sigmoid1(output)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        self.d = hidden_dim
        self.encoder = Encoder(self.d)
        self.decoder = Decoder(self.d)

    def forward(self, x):
        output, indc1, indc2 = self.encoder(x)
        output = self.decoder(output, indc1, indc2)
        return output


def train_autoencoder(model, num_epochs, optimizer, t_loader, device=DEVICE):

    model.train()
    train_loss_avg = []
    print('Training ...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        for image_batch, _ in t_loader:
            image_batch = image_batch.to(device)
            # autoencoder reconstruction
            image_batch_recon = model(image_batch)
            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            train_loss_avg[-1] += loss.item()
            num_batches += 1
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' %
              (epoch+1, num_epochs, train_loss_avg[-1]))


if __name__ == '__main__':
    ic("CT Image Autoencoder Backbone")
