import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.datamodule import ASLDataModule
from src.model.classifier import LitASLViT


def main():
    datamodule = ASLDataModule(
        data_dir="data/raw",
        batch_size=32,
        img_size=224
    )
    datamodule.setup()

    model = LitASLViT(
        num_classes=datamodule.num_classes,
        lr=3e-4
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="asl-vit-{epoch:02d}-{val_acc:.2f}"
    )

    logger = TensorBoardLogger("logs", name="asl_vit")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb],
        logger=logger
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
