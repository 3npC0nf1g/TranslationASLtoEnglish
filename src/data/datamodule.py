import pytorch_lightning as pl
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


class ASLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../../data/raw",
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --------------------
        # Transforms
        # --------------------
        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage=None):
        # Load datasets with different transforms
        full_train = datasets.ImageFolder(
            root=self.hparams.data_dir,
            transform=self.train_transforms
        )

        full_val = datasets.ImageFolder(
            root=self.hparams.data_dir,
            transform=self.val_transforms
        )

        # Create deterministic split
        indices = torch.randperm(len(full_train))
        train_len = int(0.8 * len(indices))
        val_len = int(0.1 * len(indices))
        test_len = len(indices) - train_len - val_len

        self.train_set = Subset(
            full_train, indices[:train_len]
        )

        self.val_set = Subset(
            full_val, indices[train_len:train_len + val_len]
        )

        self.test_set = Subset(
            full_val, indices[train_len + val_len:]
        )

        # Needed by the model
        self.num_classes = len(full_train.classes)
        self.class_names = full_train.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
