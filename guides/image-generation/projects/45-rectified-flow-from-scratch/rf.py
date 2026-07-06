"""Rectified flow / flow matching, complete in ~40 lines.

    t ~ U(0, 1),  eps ~ N(0, I)
    x_t = (1 - t) * x0 + t * eps        straight line from data to noise
    v_true = eps - x0                    the (constant!) velocity along it
    loss = || v_theta(x_t, t) - v_true ||^2

Sampling solves dx/dt = v_theta from t = 1 (noise) to t = 0 (data) — plain
Euler. No beta schedule, no alpha-bar, no posterior variance: the entire
schedule zoo of phase 5 is replaced by a linear interpolation.

The `pairing` argument is the hook project 47 (re-flow) uses: by default
(x0, eps) are paired at random each batch; re-flow instead feeds FIXED
(sample, noise) couples produced by a trained model.
"""

import torch


def rf_loss(model, x0: torch.Tensor, eps: torch.Tensor | None = None,
            model_kwargs: dict | None = None) -> torch.Tensor:
    if eps is None:
        eps = torch.randn_like(x0)  # random pairing (vanilla flow matching)
    t = torch.rand(x0.size(0), device=x0.device)
    tb = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1 - tb) * x0 + tb * eps
    v_true = eps - x0
    v = model(x_t, t, **(model_kwargs or {}))
    return torch.mean((v - v_true) ** 2)


@torch.no_grad()
def euler_sample(model, x_init: torch.Tensor, n_steps: int,
                 model_kwargs: dict | None = None,
                 return_trajectory: bool = False):
    """Integrate dx/dt = v from t=1 to t=0 in n_steps equal Euler steps."""
    x = x_init.clone()
    traj = [x.clone()]
    ts = torch.linspace(1.0, 0.0, n_steps + 1)
    for i in range(n_steps):
        t = ts[i].expand(x.size(0)).to(x.device)
        v = model(x, t, **(model_kwargs or {}))
        x = x + (ts[i + 1] - ts[i]) * v  # negative dt: moving toward data
        traj.append(x.clone())
    return (x, traj) if return_trajectory else (x, None)
