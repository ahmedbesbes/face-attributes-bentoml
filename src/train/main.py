import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train.model import AttributeClassifier
from src.train.datasets import CelebaDataModule
from src.train.config import TrainingConfig


def train(config: TrainingConfig):
    data = pd.read_csv(config.csv_data_path)
    data = data.sample(config.n_samples)
    data = data.replace(-1, 0)
    df_train, df_test = train_test_split(data, test_size=config.test_size)

    data_module = CelebaDataModule(
        df_train,
        df_test,
        batch_size=config.batch_size,
        image_folder=config.image_folder_path,
        num_workers=config.num_workers,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_auc",
        mode="max",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_auc",
        patience=2,
        mode="max",
    )
    model = AttributeClassifier(lr=config.lr)

    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=config.n_epochs,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--csv_data_path")
    argument_parser.add_argument("--image_folder_path")
    argument_parser.add_argument("--n_samples", default=5000)
    argument_parser.add_argument("--test_size", default=0.2)
    argument_parser.add_argument("--n_epochs", default=10)
    argument_parser.add_argument("--batch_size", default=8)
    argument_parser.add_argument("--num_workers", default=4)
    argument_parser.add_argument("--lr", default=1.5e-3)
    args = argument_parser.parse_args()
    config = TrainingConfig(**vars(args))
    train(config)
