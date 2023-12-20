from typing import List
import torch
from torch_geometric.nn.norm import InstanceNorm


class PointMLPBlock(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            torch.nn.Linear(in_channels, out_channels, bias=False),
            # torch.nn.BatchNorm1d(out_channels),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
        )


class PointMLP(torch.nn.Sequential):
    def __init__(self, in_channels: int, channels: List):
        channels = [in_channels] + channels
        super().__init__(
            *[
                PointMLPBlock(in_c, out_c)
                for in_c, out_c in zip(channels[:-1], channels[1:])
            ],
            # torch.nn.BatchNorm1d(channels[-1])
        )
