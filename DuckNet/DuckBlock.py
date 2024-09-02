from __future__ import annotations

import torch.nn as nn
from .customblock import *

from typing import Literal, List

__all__ = [
    "DuckBlock",
]


class DuckBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        normalization: Literal["batch"] = "batch",
    ):
        super(DuckBlock, self).__init__()
        self.before_norm = (
            nn.BatchNorm2d(in_channel) if normalization == "batch" else nn.Identity()
        )
        self.after_norm = (
            nn.BatchNorm2d(out_channel) if normalization == "batch" else nn.Identity()
        )

        self.wide_scope = WideScope(in_channel, out_channel, kernel_size, "batch")
        self.mid_scope = MidScope(in_channel, out_channel, kernel_size, "batch")
        self.one_depth_residual = ResidualBlock(
            in_channel, out_channel, kernel_size, "batch"
        )
        self.two_depth_residual = nn.Sequential(
            ResidualBlock(in_channel, out_channel, kernel_size, "batch"),
            ResidualBlock(out_channel, out_channel, kernel_size, "batch"),
        )
        self.three_depth_residual = nn.Sequential(
            ResidualBlock(in_channel, out_channel, kernel_size, "batch"),
            ResidualBlock(out_channel, out_channel, kernel_size, "batch"),
            ResidualBlock(out_channel, out_channel, kernel_size, "batch"),
        )
        self.seperate_block = SeperateBlock(
            in_channel, out_channel, kernel_size, "batch"
        )

    def forward(self, x):
        before_norm = self.before_norm(x)

        wide_out = self.wide_scope(before_norm)
        mid_out = self.mid_scope(before_norm)
        one_depth_out = self.one_depth_residual(before_norm)
        two_depth_out = self.two_depth_residual(before_norm)
        three_depth_out = self.three_depth_residual(before_norm)
        seperate_out = self.seperate_block(before_norm)

        after_norm = self.after_norm(
            wide_out
            + mid_out
            + one_depth_out
            + two_depth_out
            + three_depth_out
            + seperate_out
        )

        return after_norm
