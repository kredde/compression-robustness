from src.datamodules.data.corruptions import CorruptionDataloader
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, ImageFolder

from torchvision.transforms import transforms


class GTSRBDataModule(LightningDataModule):
    """
      Data module for GTSRB dataset.
    """

    def __init__(
        self,
        data_dir: str = '/data/datasets/gtsrb/converted',
        train_val_split: Tuple[int, int] = (35_000, 4_209),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_dim: int = 48,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        normalize = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))

        self.train_transforms = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            normalize
        ])

        self.test_transforms = transforms.Compose(
            [transforms.Resize((img_dim, img_dim)), transforms.ToTensor(), normalize])

        self.dims = (3, img_dim, img_dim)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
          Split data and set
        """

        data_train = ImageFolder(root=f"{self.data_dir}/training", transform=self.train_transforms)
        self.data_train, self.data_val = random_split(data_train, self.train_val_split)

        self.data_test = ImageFolder(root=f"{self.data_dir}/test", transform=self.test_transforms)

        self.data_c_test = ImageFolder(root=f"{self.data_dir}/test",
                                       transform=transforms.Compose([transforms.Resize((48, 48))]))

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
