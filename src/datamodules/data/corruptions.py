from typing import Any, Callable, Optional
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import pathlib
import logging
import os
import imagecorruptions
from torchvision import transforms
from tqdm import tqdm


logger = logging.getLogger(__name__)


class SingleCurruptionDataloader(data.Dataset):
    """
        Corrupts the dataset with a specific corruption and severity and stores it.

        args:
            dataset (Dataset): An existing dataset e.g. torch.datasets.CIFAR10
            folder_name (string): the name of the folder where the corrupted dataset will be stored
            corruption (string): the type of corruption which should be applied
            severity (int): the severity of the corruption
    """

    def __init__(self, dataset: data.Dataset, corruption: str, severity: int,
                 folder_name: str, transform: Optional[Callable] = None):

        super().__init__()

        assert(corruption in imagecorruptions.get_corruption_names(), 'Corruption must be of type imagecorruptions')
        assert(severity >= 1 or severity <= 5, 'Severity must be between 1 and 5')
        self.corruption = corruption
        self.severity = severity

        base_folder = pathlib.Path(dataset.root) / folder_name
        folder = base_folder / 'corrupted' / corruption / str(severity)
        os.makedirs(folder, exist_ok=True)

        self.data_file_name = folder / 'data.npy'
        self.targets_file_name = folder / 'targets.npy'
        self.dataset = dataset
        self.transform = transform

        if not self.data_file_name.exists():
            logger.info('Generating corrupted dataset')
            self._setup()
        else:
            logger.info('Using pre generated currupted dataset')

        self.data = np.load(self.data_file_name)
        self.targets = np.load(self.targets_file_name)

    def _setup(self):
        data_c = []
        targets = []

        for d in tqdm(iter(self.dataset)):
            img, c = d
            img_array = np.asarray(img)

            corrupted = imagecorruptions.corrupt(img_array, corruption_name=self.corruption, severity=self.severity)
            data_c.append(corrupted)
            targets.append(c)

        # save data to disk
        np.save(self.data_file_name, np.array(data_c))
        np.save(self.targets_file_name, np.array(targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CorruptionDataloader(data.Dataset):
    """
        Corrupts the dataset using https://github.com/bethgelab/imagecorruptions and stores it.

        args:
            dataset (Dataset): An existing dataset e.g. torch.datasets.CIFAR10
            folder_name (string): the name of the folder where the corrupted dataset will be stored. Full path will be `dataset.root / folder_name`
    """

    def __init__(self, dataset: data.Dataset, folder_name: str, transform: Optional[Callable] = None):

        super().__init__()

        base_folder = pathlib.Path(dataset.root) / folder_name
        folder = base_folder / 'corrupted'
        os.makedirs(folder, exist_ok=True)

        self.data_file_name = folder / 'data.npy'
        self.targets_file_name = folder / 'targets.npy'

        self.dataset = dataset
        self.transform = transform

        if not self.data_file_name.exists():
            logger.info('Generating corrupted dataset')
            self._setup()
        else:
            logger.info('Using pre generated currupted dataset')

        self.data = np.load(self.data_file_name)
        self.targets = np.load(self.targets_file_name)

    def _setup(self):
        data_c = []
        targets = []

        for d in iter(self.dataset):
            img, c = d
            img_array = np.asarray(img)

            # generate corruptions
            for corruption in imagecorruptions.get_corruption_names():
                for severity in range(5):
                    corrupted = imagecorruptions.corrupt(
                        img_array, corruption_name=corruption, severity=severity + 1)
                    data_c.append(corrupted)
                    targets.append(c)

        # save data to disk
        np.save(self.data_file_name, np.array(data_c))
        np.save(self.targets_file_name, np.array(targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
