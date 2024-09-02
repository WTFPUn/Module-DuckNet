from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, List

__all__ = [
    "ResidualBlock",
    "MidScope",
    "WideScope",
    "SeperateBlock",
]


def A():
    """
    for test `__all__` variable
    """
    pass


class MidScope(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, normalization: Literal["batch"]
    ):
        super(MidScope, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class WideScope(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, normalization: Literal["batch"]
    ):
        super(WideScope, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, normalization: Literal["batch"]
    ):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, padding="same"
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
        )

        self.norm = (
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)

        return self.norm(x + res)


class SeperateBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, normalization: Literal["batch"]
    ):
        super(SeperateBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=(1, kernel_size), padding="same"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
            nn.Conv2d(
                out_channel, out_channel, kernel_size=(kernel_size, 1), padding="same"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)
