"""A small KL-regularized autoencoder in the Stable Diffusion mold.

Images (padded to 32x32) are compressed 4x per side into a 8x8x4 latent —
the same 8x-per-side/4-channel *shape* of trick SD's VAE plays on 512x512
photos, scaled down to run in minutes. The encoder outputs a Gaussian
(mu, logvar) per latent position; the KL term is kept TINY. This is the
"VAE for diffusion" recipe: we want a well-behaved, smooth, roughly
unit-scale latent space, not a generative VAE — the diffusion model
(project 38) will do the generating.
"""

import torch
import torch.nn.functional as F
from torch import nn

LATENT_CHANNELS = 4


class Encoder(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, stride=2, padding=1), nn.SiLU(),       # 32 -> 16
            nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),  # 16 -> 8
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1), nn.SiLU(),
        )
        self.to_moments = nn.Conv2d(2 * ch, 2 * LATENT_CHANNELS, 1)

    def forward(self, x):
        mu, logvar = self.to_moments(self.net(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)


class Decoder(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 2 * ch, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                # 8 -> 16
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),                # 16 -> 32
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.encoder = Encoder(ch)
        self.decoder = Decoder(ch)

    def encode(self, x, sample: bool = True):
        mu, logvar = self.encoder(x)
        if not sample:
            return mu
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        recon = self.decoder(z)
        kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).mean()
        return recon, kl


def vae_loss(model: VAE, feature_extractor, x, kl_weight: float = 1e-4,
             perceptual_weight: float = 0.1):
    """Reconstruction + tiny KL + perceptual term.

    The perceptual term compares classifier features of input and
    reconstruction — a small stand-in for the LPIPS loss SD's VAE uses. The
    adversarial term of the full recipe is omitted here (it fights blur at
    photo scale; at MNIST scale the perceptual term is enough) — see the
    README for where it slots in.
    """
    recon, kl = model(x)
    mse = F.mse_loss(recon, x)
    # feature_extractor expects 28x28: crop the 2px pad back off.
    f_real = feature_extractor.features(x[:, :, 2:-2, 2:-2])
    f_fake = feature_extractor.features(recon[:, :, 2:-2, 2:-2])
    perceptual = F.mse_loss(f_fake, f_real)
    return mse + kl_weight * kl + perceptual_weight * perceptual, mse, kl
