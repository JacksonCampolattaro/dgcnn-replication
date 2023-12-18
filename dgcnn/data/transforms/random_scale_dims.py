import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomScaleDims(BaseTransform):

    def __init__(self, scales):
        super().__init__()
        self.low, self.high = scales

    def __call__(self, data: Data) -> Data:
        scale = torch.rand(data.pos.shape[-1]) * (self.high - self.low) + self.low
        data.pos = data.pos * scale[None, :]
        return data
