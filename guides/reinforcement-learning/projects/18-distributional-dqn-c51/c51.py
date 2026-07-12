"""C51: predict the whole distribution of returns, not just its mean.

DQN learns Q(s, a) = E[G | s, a] -- one number. C51 learns the distribution of G
itself, discretized onto a fixed grid of `n_atoms` values ("atoms") spanning
[v_min, v_max]. The network outputs a probability per atom per action, so the
answer to "what happens if I do this" stops being "about 1.0" and becomes "half
the time +3, half the time -1".

The only genuinely new mechanic is the PROJECTION. The Bellman update maps each
atom z_i to r + gamma * z_i, which almost never lands on another atom -- so the
shifted distribution has to be redistributed onto the fixed grid before it can be
compared with a prediction. That is `project_distribution` below, and it is the
whole trick; the rest is a cross-entropy loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn_lib import DQNAgent

V_MIN, V_MAX, N_ATOMS = -2.0, 4.0, 51


class C51Net(nn.Module):
    """Same body as an ordinary Q-net; the head is (n_actions x n_atoms) logits."""

    def __init__(self, obs_dim, n_actions, n_atoms=N_ATOMS, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions * n_atoms),
        )

    def forward(self, x):
        """-> log-probabilities, shape (batch, n_actions, n_atoms)."""
        logits = self.net(x).view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(logits, dim=-1)


def project_distribution(next_probs, rewards, dones, support, gamma):
    """The categorical projection, a.k.a. the only hard part of C51.

    Args:
        next_probs: (B, n_atoms) -- the target net's distribution for the chosen a*
        rewards:    (B,)
        dones:      (B,)  1.0 if the episode ended (then the return IS just r)
        support:    (n_atoms,) the fixed atom values z_0 .. z_{N-1}
        gamma:      discount, already raised to the n-step power if using n-step

    Returns:
        (B, n_atoms) -- the target distribution, re-landed on the fixed grid.

    Mechanically: shift every atom to Tz = clamp(r + gamma * z, v_min, v_max), find
    where it falls in the grid (a fractional index b), and split that atom's
    probability mass between the two neighbouring atoms floor(b) and ceil(b) in
    proportion to how close it landed to each. Mass is conserved, and a shifted
    distribution that lands exactly on a grid point degenerates to a plain copy.
    """
    batch, n_atoms = next_probs.shape
    v_min, v_max = float(support[0]), float(support[-1])
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Where does each atom move to? A terminal transition's return is exactly r,
    # so the whole distribution collapses onto a single point -- that is what
    # (1 - dones) does here.
    tz = rewards[:, None] + gamma * support[None, :] * (1.0 - dones[:, None])
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / delta_z          # fractional position on the grid
    lower = b.floor().long()
    upper = b.ceil().long()

    # An atom landing exactly on a grid point has lower == upper, and the two
    # weights below would both be zero, silently deleting its mass. Nudge the
    # bounds apart in that case (guarding the two edges separately).
    lower = torch.where((upper > 0) & (lower == upper), lower - 1, lower)
    upper = torch.where((lower < n_atoms - 1) & (lower == upper), upper + 1, upper)

    m = torch.zeros(batch, n_atoms, dtype=next_probs.dtype)
    w_lower = next_probs * (upper.float() - b)     # closer to the lower atom -> more mass
    w_upper = next_probs * (b - lower.float())
    m.scatter_add_(1, lower, w_lower)
    m.scatter_add_(1, upper, w_upper)
    return m


class C51Agent(DQNAgent):
    """DQN with a distributional head. Everything else -- replay, target network,
    epsilon-greedy, double action selection -- is inherited unchanged."""

    def __init__(self, net_fn, n_actions, cfg, v_min=V_MIN, v_max=V_MAX,
                 n_atoms=N_ATOMS):
        super().__init__(net_fn, n_actions, cfg)
        self.n_atoms = n_atoms
        self.support = torch.linspace(v_min, v_max, n_atoms)

    @torch.no_grad()
    def q_values(self, obs_batch):
        """Collapse the distribution to its mean, which is all epsilon-greedy needs."""
        log_p = self.net(torch.as_tensor(obs_batch, dtype=torch.float32))
        return (log_p.exp() * self.support).sum(dim=-1)

    @torch.no_grad()
    def distribution(self, obs_batch):
        """(batch, n_actions, n_atoms) probabilities -- for the figures."""
        return self.net(torch.as_tensor(obs_batch, dtype=torch.float32)).exp()

    def update(self, batch):
        cfg = self.cfg
        gamma_n = cfg.gamma ** cfg.n_step

        with torch.no_grad():
            # pick a* at s' -- by the mean, exactly as DQN would
            next_log_p = self.target(batch.s2)
            next_q = (next_log_p.exp() * self.support).sum(-1)
            if cfg.double:
                online_q = (self.net(batch.s2).exp() * self.support).sum(-1)
                a_star = online_q.argmax(-1)
            else:
                a_star = next_q.argmax(-1)
            next_probs = next_log_p.exp()[torch.arange(len(a_star)), a_star]
            target_probs = project_distribution(next_probs, batch.r, batch.d,
                                                self.support, gamma_n)

        log_p = self.net(batch.s)[torch.arange(len(batch.a)), batch.a]
        # cross-entropy between the projected target and the prediction. This
        # replaces the squared TD error entirely -- there is no "TD error" left to
        # square, only two distributions to compare.
        per_sample = -(target_probs * log_p).sum(-1)
        loss = (batch.w * per_sample).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optim.step()
        self.updates += 1
        self.sync_target()

        # PER wants a per-sample "how surprising was this"; the cross-entropy is
        # the distributional stand-in for |TD error|.
        priorities = per_sample.detach().abs().numpy()
        with torch.no_grad():
            q_mean = float((log_p.exp() * self.support).sum(-1).mean())
        return float(loss.item()), priorities, q_mean
