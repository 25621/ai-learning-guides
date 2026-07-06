"""2D toy datasets for score matching, scaled to roughly [-4, 4]."""

import math

import torch


def eight_gaussians(n: int, generator=None) -> torch.Tensor:
    """Eight well-separated modes on a circle of radius 3."""
    centers = torch.tensor(
        [
            (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4))
            for k in range(8)
        ]
    ) * 3.0
    idx = torch.randint(0, 8, (n,), generator=generator)
    return centers[idx] + 0.25 * torch.randn(n, 2, generator=generator)


def swiss_roll(n: int, generator=None) -> torch.Tensor:
    """A 2D spiral: radius grows with angle."""
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=generator))
    x = torch.stack([t * t.cos(), t * t.sin()], dim=1) / 4.0
    return x + 0.15 * torch.randn(n, 2, generator=generator)


DATASETS = {"eight_gaussians": eight_gaussians, "swiss_roll": swiss_roll}
