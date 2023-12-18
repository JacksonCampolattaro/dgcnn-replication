import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomShift(BaseTransform):

    def __init__(self, max_offset=1.0):
        super().__init__()
        self.max_offset = max_offset

    def __call__(self, data: Data) -> Data:
        offset = torch.rand(data.pos.shape[-1]) * (2 * self.max_offset) - self.max_offset
        data.pos = data.pos + offset[None, :]
        return data
