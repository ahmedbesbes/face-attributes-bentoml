import os
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms.transforms import Compose
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.train.utils import train_transform, valid_transform


class CelebaDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        image_folder: str,
        augmentation: Compose,
    ):
        self.data = data
        self.column_labels = self.data.columns.tolist()[1:]
        self.column_image_id = self.data.columns.tolist()[0]
        self.image_folder = image_folder
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        labels = torch.FloatTensor(row[self.column_labels])
        image_id = row[self.column_image_id]
        image = Image.open(os.path.join(self.image_folder, image_id))
        tensors = self.augmentation(image)
        return dict(
            tensors=tensors,
            labels=labels,
        )


class CelebaDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size, image_folder, num_workers):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CelebaDataset(
            self.train_df,
            self.image_folder,
            train_transform,
        )
        self.test_dataset = CelebaDataset(
            self.test_df,
            self.image_folder,
            valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
