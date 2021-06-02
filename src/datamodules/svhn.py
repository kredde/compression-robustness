from src.datamodules.data.corruptions import CorruptionDataloader
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import SVHN
from torchvision.transforms import transforms


class SVHNDataModule(LightningDataModule):
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

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.dims = (3, 32, 32)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """
          Download data if needed
        """

        SVHN(self.data_dir, split='test', download=True)

    def setup(self, stage: Optional[str] = None):
        self.data_test = SVHN(self.data_dir, split='test',
                              transform=self.transforms)

    def train_dataloader(self):
        return

    def val_dataloader(self):
        return

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
