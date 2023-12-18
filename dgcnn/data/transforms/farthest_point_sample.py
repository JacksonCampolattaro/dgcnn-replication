from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.nn.pool import fps


class FarthestPointSample(BaseTransform):

    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def __call__(self, data: Data):
        indices = fps(
            data.pos, data.batch,
            ratio=self.ratio
        )
        data.pos = data.pos[indices, :]
        if 'normal' in data.keys() and data.normal is not None:
            data.normal = data.normal[indices, :]
        return data
