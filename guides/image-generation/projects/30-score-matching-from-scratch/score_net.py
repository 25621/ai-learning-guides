"""A small MLP that predicts the score s(x, sigma) of noised 2D data.

The noise level enters through Gaussian Fourier features of log(sigma) — the
continuous-noise analogue of the sinusoidal timestep embedding in the DDPM
U-Net (project 24). The network outputs the score directly; the training loss
scales it by sigma so all noise levels contribute comparably.
"""

import torch
from torch import nn


class ScoreNet(nn.Module):
    def __init__(self, hidden: int = 128, fourier_dim: int = 16):
        super().__init__()
        # Fixed random projection for the log-sigma embedding (not trained).
        self.register_buffer("freqs", torch.randn(fourier_dim) * 2.0)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * fourier_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: (B, 2), sigma: (B,) -> score estimate (B, 2)."""
        proj = sigma.log()[:, None] * self.freqs[None, :]
        emb = torch.cat([proj.sin(), proj.cos()], dim=1)
        return self.net(torch.cat([x, emb], dim=1))
