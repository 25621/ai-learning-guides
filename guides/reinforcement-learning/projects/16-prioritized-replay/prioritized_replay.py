"""Does prioritizing by TD error actually buy sample efficiency? Measure it.

The testbed is Schaul et al.'s own: the BLIND CLIFFWALK. N states in a row, two
actions. The 'right' action advances one state; the 'wrong' action ends the
episode with nothing. Only the single all-right sequence of length N pays out
(+1). A random policy therefore succeeds with probability 2^-N -- the reward
transition is not rare by accident, it is rare by construction, and its rarity is
a dial we can turn.

The experiment is deliberately OFFLINE, because that is the only way to isolate
the thing under test. Both replay strategies are handed the SAME fixed dataset
and the SAME network initialization, and differ in exactly one respect: which
rows of that dataset they look at. Any difference in learning speed is therefore
attributable to the sampler, not to luckier exploration.

Ground truth is known in closed form -- Q(s, right) = gamma^(N-1-s), Q(s, wrong) = 0
-- so "how wrong is the network" needs no reference implementation.

Runtime: ~3 min on 12 CPU cores.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))

import matplotlib.pyplot as plt  # noqa: E402
from dqn_lib import ReplayBuffer  # noqa: E402
from per import PrioritizedReplayBuffer, SumTree  # noqa: E402
from plot_style import (INK, INK_MUTED, INK_SECONDARY, SERIES,  # noqa: E402
                        SURFACE, finish, new_axes, style_axes)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

GAMMA = 0.99
SIZES = [8, 10, 12, 14]
SEEDS = 5
MAX_UPDATES = 25_000   # generous enough that even N = 14 finishes rather than being censored
BATCH = 32
LR = 1e-3
TARGET_SYNC = 200      # offline Q-learning oscillates badly without this (see project 13)
SUCCESSES = 3          # collect until the random policy has stumbled into this many payouts
EVAL_EVERY = 50
SOLVED_RMS = 0.05      # "learned it": RMS error against the exact Q* drops below this


# --------------------------------------------------------------------------
# The environment, its exact solution, and a dataset from a random policy
# --------------------------------------------------------------------------

def true_q(n):
    """Q*(s, right) = gamma^(N-1-s); Q*(s, wrong) = 0. Column 0 = wrong, 1 = right."""
    q = np.zeros((n, 2), dtype=np.float32)
    q[:, 1] = GAMMA ** (n - 1 - np.arange(n))
    return q


def collect_dataset(n, seed):
    """Roll a uniformly random policy until it has succeeded SUCCESSES times."""
    rng = np.random.default_rng(seed)
    s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
    successes, episodes = 0, 0

    while successes < SUCCESSES:
        s = 0
        episodes += 1
        while True:
            a = int(rng.integers(2))                 # 0 = wrong, 1 = right
            if a == 0:                               # step off the cliff: over, nothing
                s_list.append(s); a_list.append(a); r_list.append(0.0)
                s2_list.append(s); d_list.append(1.0)
                break
            if s == n - 1:                           # right, from the last state: payout
                s_list.append(s); a_list.append(a); r_list.append(1.0)
                s2_list.append(s); d_list.append(1.0)
                successes += 1
                break
            s_list.append(s); a_list.append(a); r_list.append(0.0)
            s2_list.append(s + 1); d_list.append(0.0)
            s = s + 1

    onehot = np.eye(n, dtype=np.float32)
    data = dict(
        s=onehot[np.array(s_list)], a=np.array(a_list, dtype=np.int64),
        r=np.array(r_list, dtype=np.float32), s2=onehot[np.array(s2_list)],
        d=np.array(d_list, dtype=np.float32),
        s_idx=np.array(s_list), episodes=episodes,
    )
    data["is_reward"] = data["r"] > 0
    return data


class TinyQ(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class CountingReplayBuffer(ReplayBuffer):
    """Uniform replay that also records how often each row was drawn, so the two
    samplers can be compared on where they spend their attention."""

    def __init__(self, capacity, obs_shape, **kw):
        super().__init__(capacity, obs_shape, **kw)
        self.sampled_count = np.zeros(self.capacity, dtype=np.int64)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self), size=batch_size)
        self.sampled_count[idx] += 1
        return self._to_batch(idx)


def fill(buffer, data):
    for i in range(len(data["a"])):
        buffer.add(data["s"][i], data["a"][i], data["r"][i], data["s2"][i], data["d"][i])


def train_offline(n, seed, method, data):
    """Train on the fixed dataset; return the error curve and updates-to-solve."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)          # identical initialization for both methods
    rng = np.random.default_rng(seed + 99)

    net = TinyQ(n)
    target_net = TinyQ(n)
    target_net.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=LR)
    size = len(data["a"])

    if method == "PER":
        buffer = PrioritizedReplayBuffer(size, (n,), alpha=0.6, beta0=0.4,
                                         beta_steps=MAX_UPDATES)
    else:
        buffer = CountingReplayBuffer(size, (n,))
    fill(buffer, data)

    q_star = torch.as_tensor(true_q(n))
    states = torch.as_tensor(np.eye(n, dtype=np.float32))
    steps, errors = [], []
    solved_at = None

    for update in range(1, MAX_UPDATES + 1):
        batch = buffer.sample(BATCH, rng)
        with torch.no_grad():
            target = batch.r + GAMMA * target_net(batch.s2).max(-1).values * (1 - batch.d)
        pred = net(batch.s).gather(-1, batch.a[:, None]).squeeze(-1)
        td = pred - target
        loss = (batch.w * F.smooth_l1_loss(pred, target, reduction="none")).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        buffer.update_priorities(batch.idx, td.detach().abs().numpy())
        if update % TARGET_SYNC == 0:
            target_net.load_state_dict(net.state_dict())

        if update % EVAL_EVERY == 0:
            with torch.no_grad():
                q = net(states)
                rms = float(torch.sqrt(((q - q_star) ** 2).mean()))
            steps.append(update)
            errors.append(rms)
            if rms <= SOLVED_RMS and solved_at is None:
                solved_at = update

    counts = buffer.sampled_count
    return {
        "n": n, "seed": seed, "method": method,
        "steps": np.array(steps), "errors": np.array(errors),
        "solved_at": solved_at if solved_at is not None else MAX_UPDATES,
        "solved": solved_at is not None,
        "reward_samples": int(counts[data["is_reward"]].sum()),
        "total_samples": int(counts.sum()),
        "n_transitions": size,
    }


def run(job):
    n, seed = job
    data = collect_dataset(n, seed)       # ONE dataset, shared by both methods
    out = []
    for method in ("uniform", "PER"):
        out.append(train_offline(n, seed, method, data))
        out[-1]["dataset_episodes"] = data["episodes"]
    return out


# --------------------------------------------------------------------------
# Figure 1: is the sum-tree actually sampling proportionally?
# --------------------------------------------------------------------------

def verify_sum_tree():
    """Give the tree known priorities, draw a lot, and compare with the theory."""
    rng = np.random.default_rng(0)
    capacity = 64
    tree = SumTree(capacity)
    priorities = rng.exponential(1.0, size=capacity) ** 2   # deliberately skewed
    for i, p in enumerate(priorities):
        tree.update(i, float(p))

    draws = 200_000
    idx = tree.find_many(rng.uniform(0, tree.total, size=draws))
    empirical = np.bincount(idx, minlength=capacity) / draws
    expected = priorities / priorities.sum()

    fig, ax = new_axes(6.6, 4.4)
    ax.scatter(expected, empirical, s=26, color=SERIES[0], alpha=0.85,
               edgecolor="none", zorder=3)
    lim = max(expected.max(), empirical.max()) * 1.06
    ax.plot([0, lim], [0, lim], color=INK_MUTED, linestyle="--", linewidth=1.0,
            zorder=2, label="perfect proportionality")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    max_err = float(np.abs(empirical - expected).max())
    finish(fig, ax,
           f"The sum-tree samples exactly in proportion to priority "
           f"(max error {max_err:.4f} over {draws:,} draws)",
           "intended probability  p_i / Σp", "empirical frequency",
           OUT / "sum_tree_check.png")
    return max_err


# --------------------------------------------------------------------------
# Figures 2 and 3: the payoff
# --------------------------------------------------------------------------

def pick(results, n, method):
    return [r for r in results if r["n"] == n and r["method"] == method]


def plot_learning(results, n=12):
    fig, ax = new_axes(7.6, 4.4)
    for i, method in enumerate(("uniform", "PER")):
        runs = pick(results, n, method)
        xs = runs[0]["steps"]
        ys = np.stack([r["errors"] for r in runs])
        color = SERIES[2] if method == "uniform" else SERIES[1]
        ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, np.median(ys, 0), color=color, linewidth=1.9,
                label=f"{method} replay")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    finish(fig, ax,
           f"Same data, same network, different sampler (N = {n}, 5 seeds)",
           "gradient update", "RMS error against the exact Q*  [log scale]",
           OUT / "learning_curves.png")


def plot_scaling(results):
    """The headline: how the gap grows as the reward gets rarer."""
    fig, ax = new_axes(7.2, 4.4)
    for method, color in (("uniform", SERIES[2]), ("PER", SERIES[1])):
        med, lo, hi = [], [], []
        for n in SIZES:
            solved = [r["solved_at"] for r in pick(results, n, method)]
            med.append(np.median(solved))
            lo.append(np.min(solved))
            hi.append(np.max(solved))
        ax.plot(SIZES, med, color=color, linewidth=1.9, marker="o", markersize=5,
                label=f"{method} replay")
        ax.fill_between(SIZES, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.axhline(MAX_UPDATES, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(SIZES[0], MAX_UPDATES * 0.88, "budget exhausted (never got there)",
            color=INK_MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    finish(fig, ax, "The rarer the reward, the more uniform replay wastes",
           "chain length N   (a random policy pays out with probability 2⁻ᴺ)",
           f"gradient updates to reach RMS error ≤ {SOLVED_RMS}  [log]",
           OUT / "updates_to_solve.png")


def plot_attention(results, n=12):
    """Where does each sampler spend its batches?

    The dashed line is the share of the dataset that pays out at all. Uniform
    replay sits exactly on it -- that is what uniform MEANS. PER climbs far above
    it, which is the entire mechanism in one bar chart.
    """
    fig, ax = new_axes(7.2, 4.2)
    values = []
    for method in ("uniform", "PER"):
        runs = pick(results, n, method)
        values.append(np.mean([r["reward_samples"] / max(1, r["total_samples"])
                               for r in runs]))
    dataset_share = np.mean([SUCCESSES / r["n_transitions"]
                             for r in pick(results, n, "uniform")])

    ax.bar(["uniform", "PER"], [v * 100 for v in values],
           color=[SERIES[2], SERIES[1]], width=0.5)
    for i, v in enumerate(values):
        ax.text(i, v * 100 + max(values) * 100 * 0.02, f"{v * 100:.3f}%", ha="center",
                color=INK_SECONDARY, fontsize=10)
    ax.axhline(dataset_share * 100, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(1.45, dataset_share * 100 + max(values) * 100 * 0.03,
            f"share of the dataset that pays out at all ({dataset_share * 100:.3f}%)",
            ha="right", color=INK_MUTED, fontsize=9)
    finish(fig, ax,
           f"PER spends its batches where the information is (N = {n})",
           "", "share of all sampled transitions that carry the reward (%)",
           OUT / "sampling_attention.png")


def main():
    max_err = verify_sum_tree()
    print(f"sum-tree check: max |empirical - intended| = {max_err:.5f}\n")

    jobs = [(n, seed) for n in SIZES for seed in range(SEEDS)]
    print(f"{len(jobs)} datasets x 2 samplers x {MAX_UPDATES:,} updates, in parallel...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = [r for pair in pool.map(run, jobs) for r in pair]

    print(f"\n{'N':>3}  {'transitions':>11}  {'reward rows':>11}  "
          f"{'uniform: updates':>17}  {'PER: updates':>13}  {'speedup':>8}")
    for n in SIZES:
        u = pick(results, n, "uniform")
        p = pick(results, n, "PER")
        u_med = np.median([r["solved_at"] for r in u])
        p_med = np.median([r["solved_at"] for r in p])
        u_solved = sum(r["solved"] for r in u)
        p_solved = sum(r["solved"] for r in p)
        size = int(np.mean([r["n_transitions"] for r in u]))
        print(f"{n:>3}  {size:>11,}  {SUCCESSES:>11}  "
              f"{u_med:>10,.0f} ({u_solved}/{SEEDS})  {p_med:>7,.0f} ({p_solved}/{SEEDS})  "
              f"{u_med / max(1, p_med):>7.1f}x")

    for n in SIZES:
        u = np.mean([r["reward_samples"] / max(1, r["total_samples"])
                     for r in pick(results, n, "uniform")])
        p = np.mean([r["reward_samples"] / max(1, r["total_samples"])
                     for r in pick(results, n, "PER")])
        print(f"N={n:>2}: share of sampled rows carrying reward — "
              f"uniform {u * 100:.3f}%, PER {p * 100:.3f}%  ({p / max(u, 1e-9):.0f}x)")

    plot_learning(results, n=12)
    plot_scaling(results)
    plot_attention(results, n=12)


if __name__ == "__main__":
    main()
