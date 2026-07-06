"""Axial 2D rotary position embeddings for a DiT.

RoPE encodes position by ROTATING each query/key vector by an angle
proportional to its token's position — relative offsets then fall out of the
attention dot product automatically, and nothing is tied to a fixed table
size. The 2D "axial" version splits each attention head's channels in half:
the first half rotates with the token's ROW index, the second half with its
COLUMN index.

Because a rotation is defined for *any* index, a RoPE model can be asked to
attend over a 9x9 grid after training on 7x7 — the extrapolation property
this project measures against a learned position table.
"""

import torch


class RoPE2D:
    """Callable applied to q and k inside project 43's Attention."""

    def __init__(self, head_dim: int, grid_h: int, grid_w: int, base: float = 100.0):
        quarter = head_dim // 4  # half the channels per axis, pairs of 2
        freqs = base ** (-torch.arange(quarter) / quarter)  # (quarter,)
        rows = torch.arange(grid_h)[:, None] * freqs[None, :]  # (H, quarter)
        cols = torch.arange(grid_w)[:, None] * freqs[None, :]
        # Per-token angle vector: row angles for the first half, col for the second.
        ang = torch.cat(
            [
                rows[:, None, :].expand(grid_h, grid_w, quarter),
                cols[None, :, :].expand(grid_h, grid_w, quarter),
            ],
            dim=-1,
        ).reshape(grid_h * grid_w, -1)  # (N, head_dim//2)
        self.cos = ang.cos()[None, None]  # (1, 1, N, head_dim//2)
        self.sin = ang.sin()[None, None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, heads, N, head_dim) -> rotated pairs (x1, x2)."""
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * self.cos - x2 * self.sin
        out[..., 1::2] = x1 * self.sin + x2 * self.cos
        return out
