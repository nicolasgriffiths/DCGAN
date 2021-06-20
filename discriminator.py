from collections import OrderedDict
import torch
from torch import nn
from torch.nn.modules import batchnorm


class DiscriminatorBlock(torch.nn.Module):
    """
    input: [B, C, H, W]
    Deconv
    BN (skip on last layer)
    Activation: ReLU / Tanh (last layer)
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, batch_norm: bool, last: bool
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"conv_{in_ch}_{out_ch}",
                        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                    ),
                    (f"batch_norm_{out_ch}", nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()),
                    (f"LeakyReLU_{out_ch}", nn.LeakyReLU(0.2, inplace=True) if not last else nn.Sigmoid()),
                ]
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Discriminator(nn.Module):
    """
    images tensor:  [B, img_channels,     64,  64]
                    [B, feature_maps * 8,  4,   4]
                    [B, feature_maps * 4,  8,   8]
                    [B, feature_maps * 2, 16,  16]
                    [B, feature_maps,     32,  32]
    output:         [B, img_channels,     64,  64]
    """

    def __init__(self, feature_maps: int, img_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block_64x64_32x32",
                        DiscriminatorBlock(
                            in_ch=img_channels,
                            out_ch=feature_maps,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            batch_norm=False,
                            last=False,
                        ),
                    ),
                    (
                        "block_32x32_16x16",
                        DiscriminatorBlock(
                            in_ch=feature_maps,
                            out_ch=feature_maps * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            batch_norm=True,
                            last=False,
                        ),
                    ),
                    (
                        "block_16x16_8x8",
                        DiscriminatorBlock(
                            in_ch=feature_maps * 2,
                            out_ch=feature_maps * 4,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            batch_norm=True,
                            last=False,
                        ),
                    ),
                    (
                        "block_8x8_4x4",
                        DiscriminatorBlock(
                            in_ch=feature_maps * 4,
                            out_ch=feature_maps * 8,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            batch_norm=True,
                            last=False,
                        ),
                    ),
                    (
                        "block_4x4_1x1",
                        DiscriminatorBlock(
                            in_ch=feature_maps * 8,
                            out_ch=1,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            batch_norm=False,
                            last=True,
                        ),
                    ),
                ]
            )
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)
