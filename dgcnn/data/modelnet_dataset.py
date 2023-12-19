import os
import glob
import shutil
from typing import Callable

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import to_list
from torch_geometric.io import read_off

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


class ModelNet40Dataset(InMemoryDataset):

    def __init__(self, root, transform, pre_transform, train=True, n_per_class=None):
        self.n_per_class = n_per_class
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform
        )
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # if at least one of the class directories is here, we probably don't need to re-download the dataset
        return ['toilet']

    @property
    def processed_file_names(self):
        # If these files are both present, then the preprocessing doesn't need to be re-done
        return ['train.pt', 'test.pt']

    def download(self):
        # Download and extract the zip file
        zip_path = download_url('http://modelnet.cs.princeton.edu/ModelNet40.zip', self.root)
        extract_zip(zip_path, self.root)
        os.unlink(zip_path)
        unzipped_dir = os.path.join(self.root, 'ModelNet40')

        # Make sure the unzipped data is in the raw_dir directory (overwrite any existing data)
        shutil.rmtree(self.raw_dir)
        os.rename(unzipped_dir, self.raw_dir)

    def process(self):
        torch.save(self._process_set('train'), self.processed_paths[0])
        torch.save(self._process_set('test'), self.processed_paths[1])

    def _process_set(self, dataset):
        classes = glob.glob(os.path.join(self.raw_dir, '*', ''))
        classes = sorted([x.split(os.sep)[-2] for x in classes])

        # Prepare a list of all .off files and their associated labels
        labeled_paths = []
        for label_id, label in enumerate(classes):
            directory = os.path.join(self.raw_dir, label, dataset)
            paths = glob.glob(f'{directory}/{label}_*.off')
            if self.n_per_class is not None and len(paths) > self.n_per_class:
                paths = paths[:self.n_per_class]
            labeled_paths = labeled_paths + list(zip(paths, [label_id] * len(paths)))

        # Load all .off files from disc
        data_list = process_map(
            self._load_data, labeled_paths,
            max_workers=31, chunksize=1,
            desc="Loading data",
            total=len(labeled_paths),
            bar_format='{l_bar}{bar}{r_bar}',
            smoothing=0.1
        )

        # Perform preprocessing on all files
        data_list = process_map(
            self._preprocess_data, data_list,
            max_workers=31, chunksize=10,
            desc="Preprocessing data",
            total=len(labeled_paths),
            bar_format='{l_bar}{bar}{r_bar}',
            smoothing=0.1
        )

        return self.collate(data_list)

    def _load_data(self, args):
        path, target = args
        data = read_off(path)
        data.y = torch.tensor([target])
        return data

    def _preprocess_data(self, data):
        if self.pre_filter is not None:
            data = self.pre_filter(data)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data
