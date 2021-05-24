from src.datamodules.data.corruptions import CorruptionDataloader
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, ImageFolder

from torchvision.transforms import transforms


class CINIC10DataModule(LightningDataModule):
    """
      Data module for CINIC10 dataset.
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

        normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                         std=[0.24205776, 0.23828046, 0.25874835])

        self.train_transforms = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ])

        self.test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

        self.dims = (3, 32, 32)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
          Split data and set
        """

        self.data_train = ImageFolder(root=f"{self.data_dir}/train", transform=self.train_transforms)
        self.data_val = ImageFolder(root=f"{self.data_dir}/valid", transform=self.train_transforms)
        self.data_test = ImageFolder(root=f"{self.data_dir}/test", transform=self.test_transforms)

        self.data_c_test = ImageFolder(root=f"{self.data_dir}/test")

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
