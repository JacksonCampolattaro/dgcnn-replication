from typing import List
import torch
from torch_geometric.nn.norm import HeteroBatchNorm
from torch_geometric.nn import GCNConv

from dgcnn.nn.util import Sequential, MaxPool, DebugPrintShape


class EdgeBatchNorm(torch.nn.Module):
    def __init__(self, k: int, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = torch.nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x.reshape(-1, self.channels)).reshape(x.shape)


class EdgeMLPBlock(Sequential):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__(
            torch.nn.Linear(in_channels, out_channels, bias=False),
            EdgeBatchNorm(k=k, channels=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )


class EdgeMLP(Sequential):
    def __init__(self, in_channels: int, channels: List, k: int):
        channels = [in_channels] + channels
        super().__init__(
            *[EdgeMLPBlock(in_c, out_c, k) for in_c, out_c in zip(channels[:-1], channels[1:])],
            # EdgeDropout(p=0.5),
            MaxPool(dim=-2),
        )
