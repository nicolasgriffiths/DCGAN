from collections import OrderedDict
import torch
from torch import nn


class GeneratorBlock(torch.nn.Module):
    """
    input: [B, C, H, W]
    Deconv
    BN (skip on last layer)
    Activation: ReLU / Tanh (last layer)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, last: bool) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"deconv_{in_ch}_{out_ch}",
                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                    ),
                    (f"batch_norm_{out_ch}", nn.BatchNorm2d(out_ch) if not last else nn.Identity()),
                    (f"ReLU_{out_ch}", nn.ReLU(True) if not last else nn.Tanh()),
                ]
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Generator(nn.Module):
    """
    noise tensor: [B, latent_dim, 1, 1]
    [B, feature_maps*8, 4, 4]
    [B, feature_maps*4, 8, 8]
    [B, feature_maps*2, 16, 16]
    [B, feature_maps, 32, 32]
    output: [B, img_channels, 64, 64]
    """

    def __init__(self, latent_dim: int, feature_maps: int, img_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "Block_1x1_4x4",
                        GeneratorBlock(
                            in_ch=latent_dim,
                            out_ch=feature_maps * 8,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            last=False,
                        ),
                    ),
                    (
                        "Block_4x4_8x8",
                        GeneratorBlock(
                            in_ch=feature_maps * 8,
                            out_ch=feature_maps * 4,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            last=False,
                        ),
                    ),
                    (
                        "Block_8x8_16x16",
                        GeneratorBlock(
                            in_ch=feature_maps * 4,
                            out_ch=feature_maps * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            last=False,
                        ),
                    ),
                    (
                        "Block_16x16_32x32",
                        GeneratorBlock(
                            in_ch=feature_maps * 2,
                            out_ch=feature_maps,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            last=False,
                        ),
                    ),
                    (
                        "Block_32x32_64x64",
                        GeneratorBlock(
                            in_ch=feature_maps,
                            out_ch=img_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            last=True,
                        ),
                    ),
                ]
            )
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.layers(noise)
