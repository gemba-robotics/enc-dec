from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    img_size: int
    base_channels: int
    num_downsamples: int
    latent_dim: int


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.img_size % (2**cfg.num_downsamples) != 0:
            raise ValueError(
                f"img_size={cfg.img_size} must be divisible by 2**num_downsamples={2**cfg.num_downsamples}"
            )

        self.cfg = cfg
        feature_size = cfg.img_size // (2**cfg.num_downsamples)
        self.feature_size = feature_size

        channels: list[int] = []
        in_ch = 1
        for i in range(cfg.num_downsamples):
            out_ch = cfg.base_channels * (2**i)
            channels.append(out_ch)
            in_ch = out_ch

        encoder_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            encoder_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        flattened_dim = channels[-1] * feature_size * feature_size if channels else feature_size * feature_size
        self.to_latent = nn.Linear(flattened_dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, flattened_dim)

        decoder_layers: list[nn.Module] = []
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        decoder_layers.append(nn.ConvTranspose2d(in_ch, 1, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        self.out_act = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(z.shape[0], -1, self.feature_size, self.feature_size)
        x_hat = self.decoder(h)
        x_hat = self.out_act(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

