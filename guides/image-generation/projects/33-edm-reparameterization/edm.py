"""EDM (Karras et al. 2022): diffusion in sigma-space with preconditioning.

The denoiser the rest of the world sees is D(x, sigma) ~ x0. Internally it
wraps a raw network F with four scalings chosen so F always works with
roughly unit-variance inputs and targets, at every noise level:

    D(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x, c_noise(sigma))

    c_in    = 1 / sqrt(sigma^2 + sigma_data^2)      unit-variance input
    c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)   how much to trust x itself
    c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)   unit-variance target
    c_noise = log(sigma) / 4                        a well-scaled conditioning value

At tiny sigma, c_skip ~ 1: D mostly passes x through and F only predicts a
small correction. At huge sigma, c_skip ~ 0: x is useless and F predicts the
image from scratch. The network never has to learn either rescaling itself.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from unet import UNet  # noqa: E402

SIGMA_DATA = 0.5  # RMS of [-1, 1]-scaled image data (EDM's default)


class EDMDenoiser(torch.nn.Module):
    def __init__(self, **unet_kwargs):
        super().__init__()
        self.net = UNet(**unet_kwargs)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(-1, 1, 1, 1)
        c_in = 1.0 / (sigma**2 + SIGMA_DATA**2).sqrt()
        c_skip = SIGMA_DATA**2 / (sigma**2 + SIGMA_DATA**2)
        c_out = sigma * SIGMA_DATA / (sigma**2 + SIGMA_DATA**2).sqrt()
        c_noise = sigma.log().flatten() / 4.0
        # The U-Net's sinusoidal time embedding happily takes a continuous
        # value — c_noise plays the role the integer timestep played in DDPM.
        F = self.net(c_in * x, c_noise)
        return c_skip * x + c_out * F


def edm_loss(model: EDMDenoiser, x0: torch.Tensor,
             p_mean: float = -1.2, p_std: float = 1.2):
    """Sample sigma from a log-normal, noise the batch, and regress D onto x0
    with the weight that makes every sigma contribute equally.

    Returns (loss, sigma, per_sample_loss) — the extras feed the
    loss-vs-sigma diagnostic figure.
    """
    B = x0.size(0)
    sigma = (p_mean + p_std * torch.randn(B, device=x0.device)).exp()
    noise = sigma.view(-1, 1, 1, 1) * torch.randn_like(x0)
    D = model(x0 + noise, sigma)
    weight = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2
    per_sample = ((D - x0) ** 2).mean(dim=(1, 2, 3))
    return (weight * per_sample).mean(), sigma.detach(), per_sample.detach()


def karras_sigmas(n: int, sigma_min: float = 0.002, sigma_max: float = 80.0,
                  rho: float = 7.0) -> torch.Tensor:
    """The EDM sampling schedule: dense near sigma_min, sparse near sigma_max."""
    i = torch.linspace(0, 1, n)
    return (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


@torch.no_grad()
def heun_sample(model: EDMDenoiser, shape: tuple, n_steps: int = 18,
                device: str = "cpu", x_init: torch.Tensor | None = None) -> torch.Tensor:
    """EDM's default sampler: Heun's 2nd-order method on dx/dsigma = (x - D)/sigma."""
    sigmas = karras_sigmas(n_steps).to(device)
    x = (sigmas[0] * torch.randn(shape, device=device)) if x_init is None else x_init.clone()
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        d = (x - model(x, s.expand(shape[0]))) / s
        x_euler = x + (s_next - s) * d
        if s_next > 0:
            d_next = (x_euler - model(x_euler, s_next.expand(shape[0]))) / s_next
            x = x + (s_next - s) * 0.5 * (d + d_next)
        else:
            x = x_euler
    # One final denoise to sigma = 0.
    return model(x, sigmas[-1].expand(shape[0]))
