from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

class SimpleCNNEncoder(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.conv_layers = nn.ModuleList([nn.Conv2d(args.image_channels, args.encoder_inner_channels, 3, 1, 1)])

        for i in range(args.encoder_depth):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(args.encoder_inner_channels, args.encoder_inner_channels, 3, 2, 1),
                nn.BatchNorm2d(args.encoder_inner_channels)
            ))


        self.last_conv = nn.Conv2d(args.encoder_inner_channels, args.latent_channel, 3, 1, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = F.relu(x)
        return self.last_conv(x)

class SimpleCNNDecoder(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self._n_inner_channels = args.decoder_inner_channels
        self._starting_img_dim = (args.image_dim // (2**args.decoder_depth))
        self.conv_layers = nn.ModuleList([nn.ConvTranspose2d(args.latent_channel, self._n_inner_channels, 3, 1, 1)])

        for i in range(args.decoder_depth):
            self.conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(self._n_inner_channels, self._n_inner_channels, 3, 2, 1, 1),
                nn.BatchNorm2d(self._n_inner_channels)
            ))

        self.last_conv = nn.ConvTranspose2d(self._n_inner_channels, args.image_channels, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = F.relu(x)
        return torch.sigmoid(self.last_conv(x))


class SimpleCNNAE(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = SimpleCNNEncoder(args)
        self.decoder = SimpleCNNDecoder(args)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)



