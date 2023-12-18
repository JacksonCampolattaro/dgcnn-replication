import os.path

from torch_geometric.data.lightning import LightningDataset
from .modelnet_dataset import ModelNet40Dataset


class ModelNet40DataModule(LightningDataset):

    def __init__(
            self,
            data_dir: str,
            num_points: int,
            **kwargs
    ):

        data_dir = os.path.join(data_dir, f'ModelNet40-{num_points}')

        super().__init__(
            train_dataset=ModelNet40Dataset(
                root=data_dir,
                pre_transform=None,
                transform=None,
                train=True,
            ),
            val_dataset=ModelNet40Dataset(
                root=data_dir,
                pre_transform=None,
                transform=None,
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
        parser.add_argument('--point_dropout', type=float, default=0.875)
