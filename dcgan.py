from typing import Optional, Tuple, List, Any
import torch
from torch import nn
import pytorch_lightning as pl

from generator import Generator
from discriminator import Discriminator


class DCGANConfig:
    beta1: float = 0.5
    beta2: float = 0.999
    feature_maps_gen: int = 64
    feature_maps_disc: int = 64
    image_channels: int = 1
    latent_dim: int = 100
    learning_rate: float = 0.0002


class DCGAN(pl.LightningModule):
    def __init__(self, config: DCGANConfig = DCGANConfig()) -> None:
        super().__init__()
        self.config = config
        self.generator = Generator(
            latent_dim=config.latent_dim, feature_maps=config.feature_maps_gen, img_channels=config.image_channels
        )
        self.generator.apply(self._weights_init)
        self.discriminator = Discriminator(feature_maps=config.feature_maps_gen, img_channels=config.image_channels)
        self.discriminator.apply(self._weights_init)
        self.loss = nn.BCELoss()

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2)
        )
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2)
        )
        lr_schedulers = []
        return [opt_disc, opt_gen], lr_schedulers

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates an image [B, C, H, W] given input noise [B, latent_dim]
        """
        noise = noise.view(*noise.shape, 1, 1)  # Reshape: Add image dims to noise tensor
        return self.generator(noise)

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        assert optimizer_idx <= 1
        real_img, _ = batch  # Label is not used
        return self._disc_step(real_img) if optimizer_idx == 0 else self._gen_step(real_img)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def _get_fake_pred(self, batch_size: int) -> torch.Tensor:
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_img = self.forward(noise)
        return self.discriminator(fake_img)

    ### Discriminator training ###

    def _disc_step(self, real_img: torch.Tensor) -> torch.Tensor:
        discriminator_loss = self._get_discriminator_loss(real_img)
        self.log("loss/discriminator", discriminator_loss, on_epoch=True)
        return discriminator_loss

    def _get_discriminator_loss(self, real_img: torch.Tensor) -> torch.Tensor:
        # Train with real_img
        real_pred = self.discriminator(real_img)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.loss(real_pred, real_gt)

        # Train with fake
        bs = real_img.size(0)
        fake_pred = self._get_fake_pred(bs)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.loss(fake_pred, fake_gt)

        return real_loss + fake_loss

    ### Generator training ###

    def _gen_step(self, real_img: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_generator_loss(real_img)
        self.log("loss/generator", gen_loss, on_epoch=True)
        return gen_loss

    def _get_generator_loss(self, real: torch.Tensor) -> torch.Tensor:
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)  # Generator tries to fool discriminator
        return self.loss(fake_pred, fake_gt)