"""DDIM sampling (Song et al. 2021) for a model trained with the DDPM loss.

The key fact: the DDPM training objective never commits you to the 1000-step
stochastic sampler. Any model that predicts eps(x_t, t) defines a whole family
of samplers over any *subsequence* of timesteps:

    x0_pred = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)
    x_prev  = sqrt(a_bar_prev) * x0_pred
            + sqrt(1 - a_bar_prev - sigma^2) * eps      (direction toward x_t)
            + sigma * z                                  (fresh noise)

    sigma = eta * sqrt((1-a_bar_prev)/(1-a_bar_t)) * sqrt(1 - a_bar_t/a_bar_prev)

eta = 0 gives the deterministic DDIM update (no fresh noise, same x_T always
maps to the same image); eta = 1 recovers DDPM-style stochasticity on the
subsequence.
"""

import torch


class DDIMSampler:
    def __init__(self, diffusion, num_steps: int = 50, eta: float = 0.0):
        self.diffusion = diffusion
        self.eta = eta
        # Evenly spaced subsequence of the training timesteps, always
        # including t = T-1 (the pure-noise end).
        self.ts = torch.linspace(0, diffusion.T - 1, num_steps).round().long().tolist()

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple,
        device: str = "cpu",
        model_kwargs: dict | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a_bar = self.diffusion.alpha_bar
        x = torch.randn(shape, device=device) if x_init is None else x_init.clone()

        for i in reversed(range(len(self.ts))):
            t = self.ts[i]
            a_t = a_bar[t]
            a_prev = a_bar[self.ts[i - 1]] if i > 0 else torch.tensor(1.0, device=device)

            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(x, t_batch, **(model_kwargs or {}))

            x0_pred = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sigma = (
                self.eta
                * ((1 - a_prev) / (1 - a_t)).sqrt()
                * (1 - a_t / a_prev).sqrt()
            )
            dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0_pred + dir_xt
            if i > 0 and self.eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x
