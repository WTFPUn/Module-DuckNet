from __future__ import annotations

from .customblock import *
from .DuckBlock import *

import torch.nn as nn
import torch

__all__ = [
    "ResidualBlock",
    "MidScope",
    "WideScope",
    "SeperateBlock",
    "DuckNet",
]


class DuckNet(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_classes,
        kernel_size=3,
        normalization="batch",
    ):
        super(DuckNet, self).__init__()

        # Encoder part
        # first layer
        self.first_duck = DuckBlock(in_channel, out_channel, kernel_size, normalization)

        # second layer
        self.second_conv = nn.Conv2d(
            out_channel, out_channel * 2, kernel_size, stride=2, padding=1
        )
        self.second_conv_from_first = nn.Conv2d(
            out_channel, out_channel * 2, kernel_size, stride=2, padding=1
        )
        self.second_duck = DuckBlock(
            out_channel * 2, out_channel * 2, kernel_size, normalization
        )

        # third layer
        self.third_conv = nn.Conv2d(
            out_channel * 2, out_channel * 4, kernel_size, stride=2, padding=1
        )
        self.third_conv_from_second = nn.Conv2d(
            out_channel * 2, out_channel * 4, kernel_size, stride=2, padding=1
        )
        self.third_duck = DuckBlock(
            out_channel * 4, out_channel * 4, kernel_size, normalization
        )

        # forth layer
        self.forth_conv = nn.Conv2d(
            out_channel * 4, out_channel * 8, kernel_size, stride=2, padding=1
        )
        self.forth_conv_from_third = nn.Conv2d(
            out_channel * 4, out_channel * 8, kernel_size, stride=2, padding=1
        )
        self.forth_duck = DuckBlock(
            out_channel * 8, out_channel * 8, kernel_size, normalization
        )

        # fifth layer
        self.fifth_conv = nn.Conv2d(
            out_channel * 8, out_channel * 16, kernel_size, stride=2, padding=1
        )
        self.fifth_conv_from_forth = nn.Conv2d(
            out_channel * 8, out_channel * 16, kernel_size, stride=2, padding=1
        )
        self.fifth_duck = DuckBlock(
            out_channel * 16, out_channel * 16, kernel_size, normalization
        )

        # sixth layer
        self.sixth_conv = nn.Conv2d(
            out_channel * 16, out_channel * 32, kernel_size, stride=2, padding=1
        )
        self.sixth_conv_from_fifth = nn.Conv2d(
            out_channel * 16, out_channel * 32, kernel_size, stride=2, padding=1
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(
                out_channel * 32, out_channel * 32, kernel_size, normalization
            ),
            ResidualBlock(
                out_channel * 32, out_channel * 16, kernel_size, normalization
            ),
        )

        # Decoder part
        # sixth layer has only nearest upsampling

        # fifth layer
        self.fifth_duck_from_sixth = DuckBlock(
            out_channel * 16, out_channel * 8, kernel_size, normalization
        )

        # forth layer
        self.forth_duck_from_fifth = DuckBlock(
            out_channel * 8, out_channel * 4, kernel_size, normalization
        )

        # third layer
        self.third_duck_from_forth = DuckBlock(
            out_channel * 4, out_channel * 2, kernel_size, normalization
        )

        # second layer
        self.second_duck_from_third = DuckBlock(
            out_channel * 2, out_channel, kernel_size, normalization
        )

        # first layer
        # self.output = nn.Sequential(
        #     DuckBlock(out_channel, out_channel, kernel_size, normalization),
        #     nn.Conv2d(out_channel, 1, 1, padding="same"),
        #     nn.Sigmoid(),
        # )
        self.output = nn.ModuleList()
        self.output.append(
            DuckBlock(out_channel, out_channel, kernel_size, normalization)
        )
        self.output.append(nn.Conv2d(out_channel, num_classes, 1, padding="same"))
        self.output.append(nn.Sigmoid())

        if num_classes > 1:
            self.output.append(nn.Softmax(dim=1))

    def forward(self, x):

        # Encoder part
        # first layer
        first_duck = self.first_duck(x)

        # second layer
        second_conv = self.second_conv(first_duck)
        second_conv_from_first = second_conv + self.second_conv_from_first(first_duck)
        second_duck = self.second_duck(second_conv_from_first)

        # # third layer
        third_conv = self.third_conv(second_duck)
        third_conv_from_second = third_conv + self.third_conv_from_second(second_duck)
        third_duck = self.third_duck(third_conv_from_second)

        # # forth layer
        forth_conv = self.forth_conv(third_duck)
        forth_conv_from_third = forth_conv + self.forth_conv_from_third(third_duck)
        forth_duck = self.forth_duck(forth_conv_from_third)

        # # fifth layer
        fifth_conv = self.fifth_conv(forth_duck)
        fifth_conv_from_forth = fifth_conv + self.fifth_conv_from_forth(forth_duck)
        fifth_duck = self.fifth_duck(fifth_conv_from_forth)

        # # sixth layer
        sixth_conv = self.sixth_conv(fifth_duck)
        sixth_conv_from_fifth = sixth_conv + self.sixth_conv_from_fifth(fifth_duck)

        # # bottleneck
        bottleneck = self.bottleneck(sixth_conv_from_fifth)

        # # Decoder part
        # # sixth layer
        bottleneck = nn.functional.interpolate(
            bottleneck, scale_factor=2, mode="nearest"
        )

        # # fifth layer
        fifth_duck_from_sixth = self.fifth_duck_from_sixth(bottleneck + fifth_duck)
        fifth_duck_from_sixth = nn.functional.interpolate(
            fifth_duck_from_sixth, scale_factor=2, mode="nearest"
        )

        print("fifth_duck_from_sixth: ", fifth_duck_from_sixth.shape)
        print("forth_duck: ", forth_duck.shape)

        # # forth layer
        forth_duck_from_fifth = self.forth_duck_from_fifth(
            fifth_duck_from_sixth + forth_duck
        )
        forth_duck_from_fifth = nn.functional.interpolate(
            forth_duck_from_fifth, scale_factor=2, mode="nearest"
        )

        # # third layer
        third_duck_from_forth = self.third_duck_from_forth(
            forth_duck_from_fifth + third_duck
        )
        third_duck_from_forth = nn.functional.interpolate(
            third_duck_from_forth, scale_factor=2, mode="nearest"
        )

        # # second layer
        second_duck_from_third = self.second_duck_from_third(
            third_duck_from_forth + second_duck
        )
        second_duck_from_third = nn.functional.interpolate(
            second_duck_from_third, scale_factor=2, mode="nearest"
        )

        # # first layer
        # output = self.output(second_duck_from_third + first_duck)
        addition = second_duck_from_third + first_duck
        for layer in self.output:
            addition = layer(addition)

        return addition
