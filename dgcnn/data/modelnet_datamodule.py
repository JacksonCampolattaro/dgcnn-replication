import os.path
from argparse import BooleanOptionalAction

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints

from .modelnet_dataset import ModelNet40Dataset
from .transforms import FarthestPointSample, RandomScaleDims, RandomShift, Shuffle, RandomPointDropout


class ModelNet40DataModule(LightningDataset):

    def __init__(
            self,
            data_dir: str,
            num_points: int,
            sampling_factor: int,
            include_normals: bool,
            **kwargs
    ):
        data_dir = os.path.join(data_dir, f'ModelNet40-{num_points}')

        pre_transform = Compose([
            NormalizeScale(),
            SamplePoints(num_points * sampling_factor, include_normals=include_normals),
            FarthestPointSample(ratio=1 / sampling_factor)
        ])

        regularize = Compose([
            RandomScaleDims(scales=(2 / 3, 3 / 2)),
            RandomShift(max_offset=0.2),
            # todo: dropout might not be necessary
            RandomPointDropout(max_dropout=0.875),
        ])

        transform = Compose([
            Shuffle(),
            # todo: maybe more should go here?
        ])

        super().__init__(
            train_dataset=ModelNet40Dataset(
                root=data_dir,
                pre_transform=pre_transform,
                transform=Compose([transform, regularize]),
                train=True,
            ),
            val_dataset=ModelNet40Dataset(
                root=data_dir,
                pre_transform=pre_transform,
                transform=transform,
                train=False,
            ),
            num_workers=31,
            persistent_workers=True,
            pin_memory=True,
            **kwargs
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data_dir', type=str, default='data')
        parser.add_argument('--num_points', type=int, default=1024)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--sampling_factor', type=int, default=1)
        parser.add_argument('--include_normals', action=BooleanOptionalAction, default=False)
        #parser.add_argument('--point_dropout', type=float, default=0.875)
