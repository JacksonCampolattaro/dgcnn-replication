import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.conv import DynamicEdgeConv


class RandomPointDropout(BaseTransform):

    def __init__(self, max_dropout=0.875, min_points=20):
        super().__init__()
        self.max_dropout = max_dropout
        self.min_points = min_points

    def __call__(self, data: Data) -> Data:
        n, d = data.pos.shape
        dropout_ratio = torch.rand(1) * self.max_dropout
        num_kept_indices = max(int(n * (1 - dropout_ratio)), self.min_points)
        kept_indices = torch.randperm(n)[:num_kept_indices]
        data.pos = data.pos[kept_indices, :]
        if 'normal' in data.keys() and data.normal is not None:
            data.normal = data.normal[kept_indices, :]

        return data
