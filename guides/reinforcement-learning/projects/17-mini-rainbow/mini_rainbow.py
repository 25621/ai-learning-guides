"""Rainbow-lite: add Double, Dueling, PER, and n-step one rung at a time.

Rainbow's contribution was not any single idea -- it was the observation that six
independent patches to DQN stack. This is that experiment, shrunk to four rungs
and a Pong you can train in a few minutes:

    DQN
    + Double            (project 15)
    + Dueling           (project 15)
    + PER               (project 16)
    + 3-step returns    (new here)

Each rung KEEPS everything below it, so the last one is all four at once. The
ablation is cumulative rather than leave-one-out because that is the claim being
tested: not "is PER good in isolation" but "does PER still help once you already
have Double and Dueling".

The one genuinely new piece of machinery is `NStepBuffer`, which sits in front of
any buffer and converts single transitions into n-step ones on the way in.

Runtime: ~7 min on 12 CPU cores (15 runs in parallel, 1 thread each).
"""

import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "13-add-a-replay-buffer"))
sys.path.insert(0, str(HERE.parent / "14-atari-pong"))
sys.path.insert(0, str(HERE.parent / "16-prioritized-replay"))

import matplotlib.pyplot as plt  # noqa: E402
from dqn_lib import Config, DQNAgent, ReplayBuffer, train_dqn  # noqa: E402
from per import PrioritizedReplayBuffer  # noqa: E402
from plot_style import (INK, INK_MUTED, INK_SECONDARY, SERIES,  # noqa: E402
                        SURFACE, finish, new_axes, style_axes)
from pong_lib import MAX_RALLIES, ConvQNet, make_minipong  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# 5 rungs x 3 seeds = 15 runs, which is one full wave of 12 cores plus a short
# second one. At 50k steps that lands at ~13 minutes; 40k is enough for every rung
# to reach competence (project 14's agent clears 7 rallies by ~25k) and brings the
# whole ablation inside the ten-minute budget.
TOTAL_STEPS = 40_000
SEEDS = 3
N_STEP = 3

BASE = Config(
    total_steps=TOTAL_STEPS, gamma=0.99, lr=1e-3, batch_size=32,
    buffer_size=20_000, learning_starts=1_000, train_freq=2, target_freq=500,
    eps_start=1.0, eps_end=0.05, eps_decay_frac=0.3,
    eval_every=5_000, eval_episodes=8,
)

# Cumulative: each rung inherits everything above it.
LADDER = {
    "DQN":         dict(double=False, dueling=False, per=False, n_step=1),
    "+ Double":    dict(double=True,  dueling=False, per=False, n_step=1),
    "+ Dueling":   dict(double=True,  dueling=True,  per=False, n_step=1),
    "+ PER":       dict(double=True,  dueling=True,  per=True,  n_step=1),
    "+ 3-step":    dict(double=True,  dueling=True,  per=True,  n_step=N_STEP),
}


class NStepBuffer:
    """Turns 1-step transitions into n-step ones, in front of any buffer.

    A one-step target only moves value one state backwards per update, so a reward
    at the end of a rally needs ~20 updates to reach the action that earned it. An
    n-step target carries the real reward n states back in a single update:

        r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n * max_a Q(s_{t+n}, a)

    which trades a little bias (the intermediate rewards came from the OLD policy;
    strictly, off-policy n-step returns want importance sampling, and everyone
    quietly skips it for small n) for a lot less waiting.

    On episode end the partial windows are flushed. Each carries done = 1, so its
    bootstrap term is multiplied by zero, and the fact that it is a k-step return
    with k < n never gets a chance to matter.
    """

    def __init__(self, base, n, gamma):
        self.base = base
        self.n = n
        self.gamma = gamma
        self.window = deque(maxlen=n)

    def _emit(self, k):
        """Collapse the oldest k entries of the window into one k-step transition."""
        s, a = self.window[0][0], self.window[0][1]
        ret, done = 0.0, 0.0
        s2 = self.window[k - 1][3]
        for i in range(k):
            _, _, r, nxt, d = self.window[i]
            ret += (self.gamma ** i) * r
            s2 = nxt
            done = d
            if d:                     # the episode ended inside the window
                break
        self.base.add(s, a, ret, s2, done)

    def add(self, s, a, r, s2, d):
        self.window.append((s, a, r, s2, d))
        if len(self.window) == self.n:
            self._emit(self.n)
        if d:                          # flush what is left, shortest window last
            while len(self.window) > 1:
                self.window.popleft()
                self._emit(len(self.window))
            self.window.clear()

    def sample(self, batch_size, rng):
        return self.base.sample(batch_size, rng)

    def update_priorities(self, idx, td):
        self.base.update_priorities(idx, td)

    def __len__(self):
        return len(self.base)


def build_buffer(cfg, flags, obs_shape):
    if flags["per"]:
        base = PrioritizedReplayBuffer(cfg.buffer_size, obs_shape, obs_dtype=np.uint8,
                                       alpha=0.6, beta0=0.4,
                                       beta_steps=cfg.total_steps // cfg.train_freq)
    else:
        base = ReplayBuffer(cfg.buffer_size, obs_shape, obs_dtype=np.uint8)
    if flags["n_step"] > 1:
        return NStepBuffer(base, flags["n_step"], cfg.gamma)
    return base


def run(job):
    name, seed = job
    torch.set_num_threads(1)
    flags = LADDER[name]
    cfg = replace(BASE, seed=seed, double=flags["double"], n_step=flags["n_step"])

    env = make_minipong(seed=seed)
    eval_env = make_minipong(seed=seed + 1000)
    agent = DQNAgent(
        lambda: ConvQNet(n_actions=env.n_actions, dueling=flags["dueling"]),
        env.n_actions, cfg)
    buffer = build_buffer(cfg, flags, env.obs_shape)

    hist = train_dqn(env, agent, buffer, cfg, eval_env=eval_env)
    evals = np.array(hist["eval_return"])
    steps = np.array(hist["eval_step"])

    # "Steps to competence" -- the first checkpoint at which the agent returns at
    # least 7 of 10 rallies. A blunt instrument, but the same blunt instrument for
    # every rung, and less seed-sensitive than final score alone.
    reached = np.where(evals >= 7.0)[0]
    return {
        "name": name, "seed": seed,
        "eval_step": steps, "eval_return": evals,
        "steps_to_7": int(steps[reached[0]]) if len(reached) else TOTAL_STEPS,
        "reached_7": bool(len(reached)),
        "auc": float(np.trapezoid(evals, steps) / (steps[-1] - steps[0])),
    }


def group(results, name):
    return [r for r in results if r["name"] == name]


def plot_ladder(results):
    fig, ax = new_axes(8.0, 4.8)
    # The seed spread is shown for the two rungs the write-up makes claims about --
    # drawing all five bands would be an unreadable smear, and hiding all of them
    # would let four curves inside one noise band look like a ranking.
    banded = {"DQN", "+ 3-step"}
    for i, name in enumerate(LADDER):
        runs = group(results, name)
        xs = runs[0]["eval_step"]
        ys = np.stack([r["eval_return"] for r in runs])
        if name in banded:
            ax.fill_between(xs, ys.min(0), ys.max(0), color=SERIES[i], alpha=0.13,
                            linewidth=0)
        ax.plot(xs, np.median(ys, 0), color=SERIES[i], linewidth=1.9, marker="o",
                markersize=3.5, label=name)
    ax.axhline(MAX_RALLIES, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, MAX_RALLIES - 0.75, "perfect play", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.axhline(7.0, color=INK_MUTED, linestyle=":", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, 7.15, "competence bar (7 rallies)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(-1.6, 11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, ax, "Rainbow's claim is that the patches stack. On this task, they do not.",
           "environment step", "rallies returned (median of 3 seeds)",
           OUT / "ladder.png")


def plot_summary(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        style_axes(ax)

    names = list(LADDER)
    aucs = [np.mean([r["auc"] for r in group(results, n)]) for n in names]
    steps = [np.median([r["steps_to_7"] for r in group(results, n)]) for n in names]
    colors = [SERIES[i] for i in range(len(names))]

    ax1.bar(range(len(names)), aucs, color=colors, width=0.6)
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.08, f"{v:.2f}", ha="center", color=INK_SECONDARY, fontsize=9)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_title("Area under the learning curve", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax1.set_ylabel("mean rallies over training", color=INK_SECONDARY, fontsize=10)

    ax2.bar(range(len(names)), steps, color=colors, width=0.6)
    for i, v in enumerate(steps):
        ax2.text(i, v + TOTAL_STEPS * 0.012, f"{int(v / 1000)}k", ha="center",
                 color=INK_SECONDARY, fontsize=9)
    ax2.axhline(TOTAL_STEPS, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax2.text(len(names) - 0.5, TOTAL_STEPS * 0.955, "never got there", ha="right",
             color=INK_MUTED, fontsize=9)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_title("Steps to 7 rallies (lower is better)", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_ylabel("environment steps", color=INK_SECONDARY, fontsize=10)

    fig.suptitle("Which rung carries the win?", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "ladder_summary.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    jobs = [(name, seed) for name in LADDER for seed in range(SEEDS)]
    print(f"{len(jobs)} MiniPong runs x {TOTAL_STEPS:,} steps, in parallel...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, jobs))

    print(f"\n{'rung':<12}  {'final':>7}  {'best':>6}  {'AUC':>6}  "
          f"{'steps to 7 rallies':>19}")
    for name in LADDER:
        runs = group(results, name)
        final = np.mean([r["eval_return"][-1] for r in runs])
        best = np.mean([r["eval_return"].max() for r in runs])
        auc = np.mean([r["auc"] for r in runs])
        s7 = np.median([r["steps_to_7"] for r in runs])
        hit = sum(r["reached_7"] for r in runs)
        print(f"{name:<12}  {final:7.2f}  {best:6.2f}  {auc:6.2f}  "
              f"{s7:>12,.0f} ({hit}/{SEEDS})")

    plot_ladder(results)
    plot_summary(results)


if __name__ == "__main__":
    main()
