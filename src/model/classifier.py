import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchvision.models import vit_b_16, ViT_B_16_Weights


class LitASLViT(pl.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 3e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features,
            num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )
