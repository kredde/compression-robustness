from src.datamodules.data.corruptions import CorruptionDataloader
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    """
      Data module for Cifar10 dataset.
    """

    def __init__(
        self,
        data_dir: str = './data',
        train_val_split: Tuple[int, int] = (45_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize
        ])

        self.test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

        self.dims = (3, 32, 32)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """
          Download data if needed
        """

        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
          Split data and set
        """

        trainset = CIFAR10(self.data_dir, train=True,
                           transform=self.train_transforms)
        testset = CIFAR10(self.data_dir, train=False,
                          transform=self.test_transforms)

        self.data_train, self.data_val, = random_split(
            trainset, self.train_val_split)
        self.data_test = testset

        self.data_c_test = CIFAR10(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
