from logging import Logger
from dataclasses import dataclass
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from dcgan import DCGAN, DCGANConfig
from data import get_datasets, get_dataloaders


@dataclass
class Config:
    # Training
    batch_size: int = 128
    num_workers: int = 4
    max_epochs: int = 10
    gpus: int = 1

    # Dataset
    dataset_name: str = "MNIST"
    image_height: int = 64
    image_width: int = 64
    image_mean: float = 0.5
    image_stdev: float = 0.5
    data_dir: str = "./datasets"

    # Model
    DCGAN_config: DCGANConfig = DCGANConfig()

    def __post_init__(self):
        if self.dataset_name == "MNIST":
            assert self.DCGAN_config.image_channels == 1


def run_training(config):
    # Dataloaders
    train_dataset, val_dataset, image_channels = get_datasets(
        dataset_name=config.dataset_name,
        image_size=(config.image_height, config.image_width),
        mean=config.image_mean,
        stdev=config.image_stdev,
        data_dir=config.data_dir,
    )
    train_dataloader, val_dataloader = get_dataloaders(
        train_dataset,
        val_dataset,
        train_batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Model
    model = DCGAN(config.DCGAN_config)

    # Trainer
    trainer = pl.Trainer(
        gpus=config.gpus,
        max_epochs=config.max_epochs,
        profiler=None,
        resume_from_checkpoint=None,
        limit_val_batches=1,
        logger=TensorBoardLogger(save_dir="lightning_logs", default_hp_metric=False),
    )
    trainer.fit(model, train_dataloader, val_dataloader)
