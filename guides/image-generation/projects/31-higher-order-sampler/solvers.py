"""Euler and Heun ODE solvers driving an ordinary trained DDPM.

The bridge (this is the whole trick): a DDPM's discrete ladder hides a
sigma-space denoiser. Divide the DDPM state by sqrt(a_bar_t) and it becomes

    x_hat = x0 + sigma * eps,    sigma(t) = sqrt(1 - a_bar_t) / sqrt(a_bar_t)

— exactly the VE form EDM works in. The probability-flow ODE in these
coordinates is simply

    dx_hat / dsigma = eps_hat(x_hat, sigma)

so any ODE solver can sample from a model that was trained as a plain DDPM.
Euler takes first-order steps along eps_hat; Heun re-evaluates at the
step's endpoint and averages the two slopes (second order, 2 evals/step).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402


class DDPMDenoiser:
    """Exposes eps_hat(x_hat, sigma) for a model trained on discrete t.

    Snapping a requested sigma to the NEAREST trained t makes eps piecewise
    constant in sigma, which puts a quantization floor under every solver
    (we measured it: Heun stopped beating Euler). Instead we exploit that
    the U-Net's sinusoidal time embedding is a perfectly smooth function of
    t — interpolate a FRACTIONAL t from the sigma table and feed the float
    straight in. The input scaling needs no table at all: from
    sigma^2 = (1 - a_bar)/a_bar it follows exactly that a_bar = 1/(1 + sigma^2).
    """

    def __init__(self, model, diffusion: GaussianDiffusion):
        self.model = model
        self.log_sigmas = ((1 - diffusion.alpha_bar) / diffusion.alpha_bar).log() / 2
        self.sigma_min = float(self.log_sigmas[0].exp())
        self.sigma_max = float(self.log_sigmas[-1].exp())

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Continuous timestep whose noise level matches sigma (log-interp)."""
        log_sig = sigma.log()
        idx = torch.searchsorted(self.log_sigmas, log_sig).clamp(1, len(self.log_sigmas) - 1)
        lo, hi = self.log_sigmas[idx - 1], self.log_sigmas[idx]
        return (idx - 1) + (log_sig - lo) / (hi - lo)

    def eps(self, x_hat: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t_cont = self.t_of_sigma(sigma.reshape(1)).expand(x_hat.size(0))
        a_bar_sqrt = (1.0 + sigma**2).rsqrt()  # exact, no table lookup
        return self.model(a_bar_sqrt * x_hat, t_cont)


def karras_grid(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0):
    i = torch.linspace(0, 1, n)
    sig = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sig, torch.zeros(1)])  # final step lands on sigma = 0


@torch.no_grad()
def ode_sample(denoiser: DDPMDenoiser, x_init: torch.Tensor, n_steps: int,
               method: str = "heun"):
    """Integrate the probability-flow ODE from sigma_max to 0.

    Returns (x0, nfe) where nfe counts model evaluations — the honest cost
    axis when comparing solvers (Heun pays 2 per step).
    """
    sigmas = karras_grid(n_steps, denoiser.sigma_min, denoiser.sigma_max).to(x_init.device)
    x, nfe = x_init.clone(), 0
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        d = denoiser.eps(x, s)
        nfe += 1
        x_next = x + (s_next - s) * d          # Euler proposal
        if method == "heun" and s_next > 0:
            d_next = denoiser.eps(x_next, s_next)
            nfe += 1
            x_next = x + (s_next - s) * 0.5 * (d + d_next)
        x = x_next
    return x, nfe
