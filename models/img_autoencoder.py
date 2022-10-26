import torch
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        # Credit: This architecture is obtained from Xiaoyang's previous research project
        super(Encoder, self).__init__()
        self.d = hidden_dim
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, hidden_dim)  # 512 -> 128

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        self.d = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(self.d, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.Unflatten(-1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        self.d = hidden_dim
        self.encoder = Encoder(self.d)
        self.decoder = Decoder(self.d)

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
