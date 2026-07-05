"""Gaussian diffusion: the fixed forward process and the learned reverse process.

Everything here is a direct transcription of the DDPM equations (Ho et al. 2020):

    forward:   q(x_t | x_0)   = N( sqrt(a_bar_t) x_0, (1 - a_bar_t) I )
    loss:      L = E || eps - eps_theta(x_t, t) ||^2
    reverse:   x_{t-1} = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1-a_bar_t) eps_theta) + sigma_t z

The class holds only precomputed schedule tensors — the model is passed in.
"""

import math

import torch


def make_beta_schedule(schedule: str, T: int) -> torch.Tensor:
    if schedule == "linear":
        # Ho et al. 2020: linear in beta from 1e-4 to 0.02 over T = 1000 steps.
        # For other T the endpoints scale by 1000/T so the TOTAL noise added
        # over the whole schedule stays the same — otherwise a short schedule
        # would never reach pure static and sampling could not start from
        # plain Gaussian noise.
        scale = 1000.0 / T
        return torch.linspace(scale * 1e-4, scale * 0.02, T)
    if schedule == "cosine":
        # Nichol & Dhariwal 2021: define a_bar(t) directly with a squared
        # cosine, then recover per-step betas from consecutive ratios.
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999).float()
    raise ValueError(f"unknown schedule: {schedule}")


class GaussianDiffusion:
    def __init__(self, T: int = 1000, schedule: str = "linear", device: str = "cpu"):
        self.T = T
        self.schedule = schedule
        self.betas = make_beta_schedule(schedule, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        # a_bar_{t-1}, with a_bar_{-1} := 1 (a clean image at "step -1")
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        # Variance of the true reverse posterior q(x_{t-1} | x_t, x_0).
        # Using this ("beta tilde") instead of beta_t is the other of the two
        # variance choices in the DDPM paper; both work, this one is slightly
        # better at low step counts.
        self.posterior_var = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

    # ------------------------------------------------------------- forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Jump straight to noise level t: x_t = sqrt(a_bar) x_0 + sqrt(1-a_bar) eps."""
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    def loss(self, model, x0: torch.Tensor, model_kwargs: dict | None = None) -> torch.Tensor:
        """The whole DDPM training objective: MSE on the added noise."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, **(model_kwargs or {}))
        return torch.mean((pred - noise) ** 2)

    # ------------------------------------------------------------- reverse
    def posterior_mean(self, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        """Mean of q(x_{t-1} | x_t, x_0) with the model's eps standing in for
        the true noise.

        Implementation detail that matters: rather than the textbook
        eps-form update, reconstruct the implied clean image x0 and CLAMP it
        to the valid pixel range before re-deriving the mean. Algebraically
        identical when the prediction is perfect, but without the clamp any
        schedule with large late-step betas (cosine hits the 0.999 clip)
        multiplies prediction errors by 1/sqrt(alpha_t) every step and
        sampling visibly explodes. Every serious DDPM codebase clips here.
        """
        a_bar, a_bar_prev = self.alpha_bar[t], self.alpha_bar_prev[t]
        x0 = ((x_t - (1.0 - a_bar).sqrt() * eps) / a_bar.sqrt()).clamp(-1, 1)
        return (
            a_bar_prev.sqrt() * self.betas[t] / (1.0 - a_bar) * x0
            + self.alphas[t].sqrt() * (1.0 - a_bar_prev) / (1.0 - a_bar) * x_t
        )

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: int, model_kwargs: dict | None = None) -> torch.Tensor:
        """One reverse step: x_t -> x_{t-1}."""
        B = x_t.size(0)
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t_batch, **(model_kwargs or {}))
        mean = self.posterior_mean(x_t, eps, t)
        if t == 0:
            return mean  # no noise is added on the last step
        return mean + self.posterior_var[t].sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape: tuple,
        device: str = "cpu",
        model_kwargs: dict | None = None,
        snapshot_ts: tuple = (),
        x_init: torch.Tensor | None = None,
    ):
        """Full T-step ancestral sampling, starting from pure noise.

        Pass `x_init` to start from a chosen x_T (e.g. to compare samplers on
        the same starting noise). Returns (x_0, snapshots) where snapshots maps
        each t in snapshot_ts to the intermediate x_t — handy for visualizing
        the denoising trajectory.
        """
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()
        snapshots = {}
        for t in reversed(range(self.T)):
            if t in snapshot_ts:
                snapshots[t] = x.clone()
            x = self.p_sample(model, x, t, model_kwargs)
        return x, snapshots
