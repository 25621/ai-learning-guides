"""The probability flow ODE for project 30's VE score model, plus exact
log-likelihood via the instantaneous change of variables.

For the VE process x_sigma = x0 + sigma * eps, the deterministic ODE that
shares every marginal with the stochastic sampler is

    dx / dsigma = -sigma * s(x, sigma)

Integrate it from sigma_max down to sigma_min and you have a sampler.
Integrate it UP (data -> noise) while accumulating the divergence and you
have the model's exact log-likelihood — the continuous normalizing flow
identity:

    log p(x_data) = log N(x(sigma_max); 0, sigma_max^2 I)
                  + integral over sigma of  div_x f(x(sigma), sigma)

In 2D the divergence is computed exactly with two vector-Jacobian products —
no Hutchinson estimator needed. This is the payoff of doing the project on
toy data: every quantity is exact and plottable.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "30-score-matching-from-scratch"))
from score_net import ScoreNet  # noqa: E402


def load_score_model(dataset: str = "eight_gaussians") -> ScoreNet:
    ckpt = torch.load(
        HERE.parent / f"30-score-matching-from-scratch/checkpoints/{dataset}.pt",
        weights_only=True,
    )
    model = ScoreNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def drift(model, x, sigma: float):
    """f(x, sigma) = -sigma * s(x, sigma) — the PF-ODE right-hand side."""
    sig = torch.full((x.size(0),), sigma)
    return -sigma * model(x, sig)


def sigma_grid(n, sigma_min=0.05, sigma_max=8.0):
    return torch.logspace(
        torch.log10(torch.tensor(sigma_max)), torch.log10(torch.tensor(sigma_min)), n
    )


@torch.no_grad()
def ode_sample(model, n=2000, n_steps=80, seed=7, return_trajectory=False):
    """Heun integration from noise to data. Deterministic given x_init."""
    torch.manual_seed(seed)
    sigmas = sigma_grid(n_steps)
    x = float(sigmas[0]) * torch.randn(n, 2)
    traj = [x.clone()]
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        d = drift(model, x, s)
        x_e = x + (s_next - s) * d
        d2 = drift(model, x_e, s_next)
        x = x + (s_next - s) * 0.5 * (d + d2)
        traj.append(x.clone())
    return (x, traj) if return_trajectory else (x, None)


@torch.no_grad()
def sde_sample(model, n=2000, n_steps=300, seed=7, return_trajectory=False):
    """Reverse-SDE (Euler–Maruyama) counterpart: same marginals, jagged paths.

    Reverse VE SDE in sigma time:  dx = -2 sigma s dsigma + sqrt(2 sigma |dsigma|) z
    """
    torch.manual_seed(seed)
    sigmas = sigma_grid(n_steps)
    x = float(sigmas[0]) * torch.randn(n, 2)
    traj = [x.clone()]
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        dsig = s_next - s  # negative: sigma is decreasing
        sig = torch.full((n,), s)
        drift_term = -2.0 * s * model(x, sig) * dsig  # moves along the score
        x = x + drift_term + (2 * s * abs(dsig)) ** 0.5 * torch.randn_like(x)
        traj.append(x.clone())
    return (x, traj) if return_trajectory else (x, None)


def divergence(model, x, sigma: float):
    """Exact div_x f in 2D: two rows of the Jacobian via autograd."""
    x = x.detach().requires_grad_(True)
    f = drift(model, x, sigma)
    div = torch.zeros(x.size(0))
    for k in range(2):
        grad_k = torch.autograd.grad(f[:, k].sum(), x, retain_graph=(k == 0))[0]
        div += grad_k[:, k]
    return f.detach(), div.detach()


def log_likelihood(model, x_data: torch.Tensor, n_steps=100,
                   sigma_min=0.05, sigma_max=8.0) -> torch.Tensor:
    """Exact log p(x) in nats, by integrating data -> noise with Heun on the
    augmented state (x, log p)."""
    sigmas = sigma_grid(n_steps, sigma_min, sigma_max).flip(0)  # ascending
    x = x_data.clone()
    delta_logp = torch.zeros(x.size(0))
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        h = s_next - s  # positive
        f1, div1 = divergence(model, x, s)
        x_e = x + h * f1
        f2, div2 = divergence(model, x_e, s_next)
        x = x + h * 0.5 * (f1 + f2)
        delta_logp = delta_logp + h * 0.5 * (div1 + div2)
    # Terminal prior: N(0, sigma_max^2 I) (data variance << sigma_max^2).
    prior = torch.distributions.Normal(0.0, sigma_max)
    log_prior = prior.log_prob(x).sum(dim=1)
    return log_prior + delta_logp
