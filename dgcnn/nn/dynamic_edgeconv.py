from typing import List

import torch

from dgcnn.nn.mlp import EdgeMLP, PointMLP
from dgcnn.nn.util import Sequential, AppendNormals, Concatenate, SequentialWithConcatenatedResults, VNMaxMeanPool, \
    DebugPrintShape, VNMaxPool
from dgcnn.geometry import FindNearestNeighbors, CollectEdgeFeatures

from .centralize_edge_features import CentralizeEdgeFeatures


class DynamicEdgeConvBlock(Sequential):
    def __init__(self, in_channels: int, channels: List, k: int, d: int = None):
        super().__init__(
            FindNearestNeighbors(k=k, d=d),
            CollectEdgeFeatures(),
            Concatenate(
                torch.nn.Identity(),
                CentralizeEdgeFeatures(),
                dim=-1
            ),
            EdgeMLP(in_channels=in_channels * 2, channels=channels, k=k),
        )
        self.out_features = channels[-1]


class DynamicEdgeConv(Sequential):
    def __init__(self, in_channels: int, k: int, embedding_features: int, hidden_channels: List):
        super().__init__(
            AppendNormals(),
            SequentialWithConcatenatedResults(
                DynamicEdgeConvBlock(in_channels, [hidden_channels[0]], k=k, d=3),
                *[
                    DynamicEdgeConvBlock(in_c, [out_c], k=k)
                    for in_c, out_c in zip(hidden_channels[:-1], hidden_channels[1:])
                ],
            ),
            PointMLP(sum(hidden_channels), [embedding_features]),
            VNMaxPool(),
        )
        self.out_features = embedding_features

    @staticmethod
    def add_args(parser):
        parser.add_argument("--k", type=int, default=20)
        parser.add_argument("--embedding_features", type=int, default=1024)
        parser.add_argument("--hidden_channels", nargs='+', type=int, default=[64, 64, 128, 256])
        return parser
