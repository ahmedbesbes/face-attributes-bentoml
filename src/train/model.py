import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchvision.models import resnet34
from src.train.config import NUM_LABELS


class AttributeClassifier(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        self.model = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Linear(512, NUM_LABELS)
        self.auroc = AUROC(task="multilabel", num_labels=NUM_LABELS)
        self.lr = lr
        self.criterion = nn.BCELoss()

    def forward(self, tensors):
        output = self.model(tensors)
        output = output.view(output.shape[0], -1)
        output = self.classifier(output)
        output = torch.sigmoid(output)
        return output

    def training_step(self, batch, batch_idx):
        tensors = batch["tensors"]
        labels = batch["labels"]
        outputs = self(tensors)
        loss = self.criterion(outputs, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        score = self.auroc(outputs, labels.long())
        self.log(
            "train_auc",
            score,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        tensors = batch["tensors"]
        labels = batch["labels"]
        outputs = self(tensors)
        loss = self.criterion(outputs, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        score = self.auroc(outputs, labels.long())
        self.log(
            "val_auc",
            score,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return dict(optimizer=optimizer)
