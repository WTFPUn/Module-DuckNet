import torch.nn as nn
import torch

from DuckNet import *

kernel_size = 3
in_channel = 3
out_channel = 3

# midScope = MidScope(in_channel, out_channel, kernel_size, "batch")
# wideScope = WideScope(in_channel, out_channel, kernel_size, "batch")
# residualBLock = ResidualBlock(in_channel, out_channel, kernel_size, "batch")
# seperateBlock = SeperateBlock(in_channel, out_channel, kernel_size, "batch")

sample = torch.randn(1, 3, 352, 352)

# wide_out = wideScope(sample)
# mid_out = midScope(sample)
# residual_out = residualBLock(sample)
# seperate_out = seperateBlock(sample)

# print("WideScope: ", wide_out.shape)
# print("MidScope: ", mid_out.shape)
# print("ResidualBlock: ", residual_out.shape)
# print("SeperateBlock: ", seperate_out.shape)

duckNet = DuckNet(in_channel, out_channel, 1, kernel_size, "batch")
# try:
output = duckNet(sample)
# except Exception as e:
#     print(e)
#     pass
print("DuckNet: ", output.shape)
