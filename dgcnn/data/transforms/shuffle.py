import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Shuffle(BaseTransform):
    def __call__(self, data: Data) -> Data:
        n, d = data.pos.shape
        shuffled_indices = torch.randperm(n)
        data.pos = data.pos[shuffled_indices, :]
        if 'normal' in data.keys() and data.normal is not None:
            data.normal = data.normal[shuffled_indices, :]
        return data
