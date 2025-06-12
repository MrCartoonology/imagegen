import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from imagegen.utils import get_device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_mb = total_bytes / (1024**2)
    return total_params, trainable_params, total_mb


class UNetBasicLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
        )

    def forward(self, x):
        return self.layer(x)


class UNetDownLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
        )

    def forward(self, x):
        return self.layer(x)


class UNetUpResidualLayer(nn.Module):
    def __init__(self, in_ch: int, res_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
        )
        self.process_cat = nn.Sequential(
            nn.Conv2d(
                in_channels=out_ch + res_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
        )

    def forward(self, x, x_res):
        x_up = self.upsample(x)
        x_cat = torch.cat([x_res, x_up], dim=1)
        return self.process_cat(x_cat)


def get_timestep_embedding(timesteps, dim):
    assert len(timesteps.size()) == 1
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)).to(
        timesteps.device
    )
    angles = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # shape: [B, dim]


class UNet(nn.Module):
    def __init__(self, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.t_lyr0a = nn.Linear(in_features=self.t_dim, out_features=3)
        self.t_lyr0b = nn.Linear(in_features=self.t_dim, out_features=12)
        self.lyr01 = UNetBasicLayer(3, 12)
        self.lyr02 = UNetBasicLayer(12, 12)

        self.lyr10 = UNetDownLayer(12, 24)
        self.t_lyr1 = nn.Linear(in_features=self.t_dim, out_features=24)
        self.lyr11 = UNetBasicLayer(24, 24)
        self.lyr12 = UNetBasicLayer(24, 24)

        self.lyr20 = UNetDownLayer(24, 48)
        self.t_lyr2 = nn.Linear(in_features=self.t_dim, out_features=48)
        self.lyr21 = UNetBasicLayer(48, 48)
        self.lyr22 = UNetBasicLayer(48, 48)

        self.lyr30 = UNetDownLayer(48, 96)
        self.t_lyr3 = nn.Linear(in_features=self.t_dim, out_features=96)
        self.lyr31 = UNetBasicLayer(96, 96)
        self.lyr32 = UNetBasicLayer(96, 96)

        self.lyr40 = UNetUpResidualLayer(in_ch=96, res_ch=48, out_ch=48)
        self.lyr41 = UNetBasicLayer(48, 48)
        self.lyr42 = UNetBasicLayer(48, 48)

        self.lyr50 = UNetUpResidualLayer(in_ch=48, res_ch=24, out_ch=24)
        self.lyr51 = UNetBasicLayer(in_ch=24, out_ch=24)
        self.lyr52 = UNetBasicLayer(in_ch=24, out_ch=24)

        self.lyr60 = UNetUpResidualLayer(in_ch=24, res_ch=12, out_ch=12)
        self.lyr61 = UNetBasicLayer(in_ch=12, out_ch=12)
        self.lyr62 = UNetBasicLayer(in_ch=12, out_ch=3)

    def forward(self, x, t):
        # x is B x 3 x 64 x 64
        t_emb = self.t_mlp(get_timestep_embedding(t, dim=self.t_dim))
        t0a = self.t_lyr0a(t_emb).unsqueeze(-1).unsqueeze(-1)
        t0b = self.t_lyr0b(t_emb).unsqueeze(-1).unsqueeze(-1)
        t1 = self.t_lyr1(t_emb).unsqueeze(-1).unsqueeze(-1)
        t2 = self.t_lyr2(t_emb).unsqueeze(-1).unsqueeze(-1)
        t3 = self.t_lyr3(t_emb).unsqueeze(-1).unsqueeze(-1)

        x01 = self.lyr01(x + t0a)  # 12 x 64 x 64
        x02 = self.lyr02(x01)  # 12 x 64 x 64
        x10 = self.lyr10(x02)  # 24 x 32 x 32  downsample

        x11 = self.lyr11(x10 + t1)  # 24 x 32 x 32
        x12 = self.lyr12(x11)  # 24 x 32 x 32
        x20 = self.lyr20(x12)  # 48 x 16 x 16  downsample

        x21 = self.lyr21(x20 + t2)  # 48 x 16 x 16
        x22 = self.lyr22(x21)  # 48 x 16 x 16
        x30 = self.lyr30(x22)  # 96 x 8 x 8 downsample

        x31 = self.lyr31(x30 + t3)  # 96 x 8 x 8
        x32 = self.lyr32(x31)  # 96 x 8 x 8
        x40 = self.lyr40(x32, x22)  # 48 x 16 x 16 upsample

        x41 = self.lyr41(x40 + t2)
        x42 = self.lyr42(x41)
        x50 = self.lyr50(x42, x12)  # 24 x 32 x 32 upsample

        x51 = self.lyr51(x50 + t1)
        x52 = self.lyr52(x51)
        x60 = self.lyr60(x52, x02)  # 12 x 64 x 64 upsample

        x61 = self.lyr61(x60 + t0b)  # 12 x 64 x 64
        x62 = self.lyr62(x61)  # 3 x 64 x 64
        return x62
