from typing import Tuple
import torch
from torchvision import transforms as trans
from torchvision.datasets import LSUN, MNIST


def getMNIST(
    image_size: Tuple[int, int] = (64, 64), mean=0.5, stdev=0.5, data_dir="./datasets"
) -> Tuple[torch.utils.data.Dataset, int]:
    transforms = trans.Compose(
        [
            trans.Resize(image_size),
            trans.ToTensor(),
            trans.Normalize((mean,), (stdev,)),
        ]
    )
    image_channels = 1
    return MNIST(root=data_dir, download=True, transform=transforms), image_channels


def getLSUN(
    image_size: Tuple[int, int] = (64, 64), mean=0.5, stdev=0.5, data_dir="./datasets"
) -> Tuple[torch.utils.data.Dataset, int]:
    transforms = trans.Compose(
        [
            trans.Resize(image_size),
            trans.CenterCrop(image_size),
            trans.ToTensor(),
            trans.Normalize((mean, mean, mean), (stdev, stdev, stdev)),
        ]
    )
    image_channels = 3
    return LSUN(root=data_dir, classes=["bedroom_train"], transform=transforms), image_channels
