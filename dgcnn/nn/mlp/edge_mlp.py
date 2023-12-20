from typing import List
import torch
from torch_geometric.nn.norm import InstanceNorm, HeteroBatchNorm
from torch_geometric.nn import MLP

from dgcnn.nn.util import Sequential, MaxPool, DebugPrintShape


class EdgeBatchNorm(torch.nn.Module):
    def __init__(self, k: int, channels: int):
        self.norm = HeteroBatchNorm(channels, num_types=64)
        super().__init__()

    def forward(self, x, batch):
        return self.norm(x, batch)


class EdgeMLPBlock(Sequential):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__(
            torch.nn.Linear(in_channels, out_channels, bias=False),
            # Transpose(),
            # torch.nn.BatchNorm1d(out_channels),
            # Transpose(),
            # EdgeLinear(in_channels, out_channels),

            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
        )


class EdgeMLP(Sequential):
    def __init__(self, in_channels: int, channels: List, k: int):
        channels = [in_channels] + channels
        super().__init__(
            # MLP(channel_list=[in_channels] + channels),

            *[EdgeMLPBlock(in_c, out_c, k) for in_c, out_c in zip(channels[:-1], channels[1:])],
            MaxPool(dim=-2),

            # torch.nn.BatchNorm1d(channels[-1])

        )
