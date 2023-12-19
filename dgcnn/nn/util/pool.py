import torch
from torch_geometric.utils import scatter
from .concatenate import Concatenate


class MaxPool(torch.nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)


class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MaxMeanPool(Concatenate):
    def __init__(self, dim=-1):
        super().__init__(
            MaxPool(dim, keepdim=True),
            MeanPool(dim, keepdim=True),
            dim=dim
        )


class VNMaxPool(torch.nn.Module):

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # torch_geometric's implementation is marginally faster than the one provided by torch
        b = batch[-1] + 1
        return scatter(x, batch, dim=0, dim_size=b, reduce='max')


class VNMeanPool(torch.nn.Module):

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # torch_geometric's implementation is marginally faster than the one provided by torch
        b = batch[-1] + 1
        return scatter(x, batch, dim=0, dim_size=b, reduce='mean')


class VNMaxMeanPool(Concatenate):
    def __init__(self, dim=-1):
        super().__init__(
            VNMaxPool(),
            VNMeanPool(),
            dim=dim
        )
