"""VP and VE diffusion behind one interface, sharing one U-Net.

Both families define a ladder of noised versions of the data and train the
same eps-predicting network — they differ ONLY in what the forward process
does to the signal:

    VP (DDPM):  x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps
                signal shrinks as noise grows; total variance stays ~1
    VE (NCSN):  x_t = x0 + sigma_t * eps
                signal untouched; variance grows without bound (to sigma_max^2)

Each class exposes perturb() for training and ancestral_step() for sampling,
so the training loop and sampler in train_sde.py / compare.py are literally
shared.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402


class VPSDE:
    """Thin wrapper over project 24's DDPM math — DDPM *is* the discrete VP SDE."""

    name = "vp"

    def __init__(self, T: int = 300, device: str = "cpu"):
        self.T = T
        self.d = GaussianDiffusion(T=T, schedule="linear", device=device)

    def perturb(self, x0, t, eps):
        return self.d.q_sample(x0, t, eps)

    @torch.no_grad()
    def ancestral_step(self, model, x, t):
        return self.d.p_sample(model, x, t)

    def prior_sample(self, shape, device):
        return torch.randn(shape, device=device)


class VESDE:
    """Discrete VE: a geometric sigma ladder, NCSN-style ancestral sampling."""

    name = "ve"

    def __init__(self, T: int = 300, sigma_min: float = 0.01,
                 sigma_max: float = 30.0, device: str = "cpu"):
        self.T = T
        # sigmas[0] = sigma_min ... sigmas[T-1] = sigma_max, geometric.
        ratio = (sigma_max / sigma_min) ** (1.0 / (T - 1))
        self.sigmas = sigma_min * ratio ** torch.arange(T, dtype=torch.float32)
        self.sigmas = self.sigmas.to(device)
        self.sigma_max = sigma_max

    def perturb(self, x0, t, eps):
        return x0 + self.sigmas[t].view(-1, 1, 1, 1) * eps

    @torch.no_grad()
    def ancestral_step(self, model, x, t):
        """Song et al.'s reverse Markov step for VE:
        mean = x + (sigma_t^2 - sigma_{t-1}^2) * score,
        var  = sigma_{t-1}^2 * (sigma_t^2 - sigma_{t-1}^2) / sigma_t^2.
        The score comes from the eps model: score = -eps / sigma_t, and the
        implied x0 = x - sigma_t * eps is clamped exactly as in project 24.
        """
        B = x.size(0)
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
        sig = self.sigmas[t]
        eps = model(x, t_batch)
        x0 = (x - sig * eps).clamp(-1, 1)
        score = (x0 - x) / sig**2  # equals -eps_clamped / sigma
        sig_prev = self.sigmas[t - 1] if t > 0 else torch.zeros_like(sig)
        beta = sig**2 - sig_prev**2
        mean = x + beta * score
        if t == 0:
            return x0
        std = (sig_prev**2 * beta / sig**2).sqrt()
        return mean + std * torch.randn_like(x)

    def prior_sample(self, shape, device):
        return self.sigma_max * torch.randn(shape, device=device)


def make_sde(family: str, T: int = 300, device: str = "cpu"):
    return {"vp": VPSDE, "ve": VESDE}[family](T=T, device=device)


def sde_loss(sde, model, x0):
    """The shared objective: predict eps at a uniformly random ladder rung."""
    B = x0.size(0)
    t = torch.randint(0, sde.T, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    x_t = sde.perturb(x0, t, eps)
    return torch.mean((model(x_t, t) - eps) ** 2)


@torch.no_grad()
def sample(sde, model, shape, device="cpu"):
    x = sde.prior_sample(shape, device)
    for t in reversed(range(sde.T)):
        x = sde.ancestral_step(model, x, t)
    return x
