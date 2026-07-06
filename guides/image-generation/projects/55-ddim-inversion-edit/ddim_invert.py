"""DDIM inversion: run the deterministic DDIM sampler *backwards* to recover the
noise x_T that would regenerate a given real image x_0.

Deterministic DDIM (eta = 0) is an ODE, so it is (approximately) invertible.
The sampling step goes from a high noise level to a lower one; inversion walks
the same subsequence the other way, from t to the next-higher t_next:

    eps      = model(x_t, t)
    x0_pred  = (x_t - sqrt(1 - a_t) eps) / sqrt(a_t)
    x_next   = sqrt(a_next) x0_pred + sqrt(1 - a_next) eps

Note we do NOT clamp x0_pred here (the sampler does, for stability). Clamping
would break invertibility — inversion needs the exact algebraic inverse.
"""

import torch


@torch.no_grad()
def ddim_invert(model, x0, ts, alpha_bar, model_kwargs=None, device="cpu"):
    """x0 (B,C,H,W) -> x_T along the increasing timestep subsequence `ts`."""
    x = x0.clone()
    for i in range(len(ts) - 1):
        t, t_next = ts[i], ts[i + 1]
        a_t = alpha_bar[t]
        a_next = alpha_bar[t_next]
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps = model(x, t_batch, **(model_kwargs or {}))
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * eps
    return x


@torch.no_grad()
def ddim_denoise(model, xT, ts, alpha_bar, model_kwargs=None, device="cpu"):
    """The exact partner of `ddim_invert`: deterministic DDIM (eta = 0) walking
    the same subsequence back down. Crucially it does NOT clamp the predicted
    x0 — clamping (which the plain sampler in project 27 does for stability)
    would break the round-trip, so reconstructing a real image needs this
    unclamped version."""
    x = xT.clone()
    for i in reversed(range(len(ts))):
        t = ts[i]
        a_t = alpha_bar[t]
        a_prev = alpha_bar[ts[i - 1]] if i > 0 else torch.tensor(1.0, device=device)
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps = model(x, t_batch, **(model_kwargs or {}))
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps
    return x
