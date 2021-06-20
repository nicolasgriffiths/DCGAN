from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import LSUN, MNIST


def get_MNIST(image_size: Tuple[int, int], mean, stdev, data_dir, train) -> Tuple[torch.utils.data.Dataset, int]:
    transforms = trans.Compose(
        [
            trans.Resize(image_size),
            trans.ToTensor(),
            trans.Normalize((mean,), (stdev,)),
        ]
    )
    image_channels = 1
    return MNIST(root=data_dir, download=True, transform=transforms, train=train), image_channels


def get_LSUN(image_size: Tuple[int, int], mean, stdev, data_dir, train) -> Tuple[torch.utils.data.Dataset, int]:
    transforms = trans.Compose(
        [
            trans.Resize(image_size),
            trans.CenterCrop(image_size),
            trans.ToTensor(),
            trans.Normalize((mean, mean, mean), (stdev, stdev, stdev)),
        ]
    )
    image_channels = 3
    return LSUN(root=data_dir, classes=["bedroom_train"], transform=transforms, train=train), image_channels


def get_dataset(dataset_name: str, **kwargs):
    if dataset_name == "MNIST":
        return get_MNIST(**kwargs)
    elif dataset_name == "LSUN":
        return get_LSUN(**kwargs)
    else:
        raise Exception(f"dataset {dataset_name} is not recognised")


def get_datasets(**kwargs):
    train_dataset, image_channels = get_dataset(train=True, **kwargs)
    val_dataset, image_channels = get_dataset(train=False, **kwargs)
    return train_dataset, val_dataset, image_channels


def get_dataloaders(train_dataset, val_dataset, train_batch_size, num_workers):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=9,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader
