import torch


class CentralizeEdgeFeatures(torch.nn.Module):

    def __init__(self, d=3):
        super().__init__()
        self.d = d

    def forward(self, x):
        # x.shape == b*n, k, d
        return x - x[:, 0, None, :]
