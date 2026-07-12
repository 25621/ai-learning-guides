"""Naive DQN on CartPole-v1: a neural net, TD(0), and nothing else.

The agent is deliberately missing both of DQN's stabilizers:
  * no experience replay -- every gradient step uses only the newest transition,
    so consecutive updates see almost identical, highly correlated states;
  * no target network -- the bootstrap target r + gamma * max Q(s') is computed
    with the very network being updated, so the target moves the instant the
    prediction does.

That is the deadly triad with the brakes cut. Three experiments:

  1. eight seeds of the naive agent               -> does it ever learn?
  2. a learning-rate sweep, still no target net   -> is divergence a tuning bug?
  3. the same agent with a target network bolted  -> which missing piece matters?
     on (still no replay)

Each run is a few tens of thousands of tiny gradient steps, so wall time is
dominated by per-step Python/torch overhead rather than by arithmetic. Every run
is therefore given one torch thread and the runs are spread across CPU cores.

Runtime: ~3 min on 12 CPU cores.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import matplotlib.pyplot as plt  # noqa: E402
from plot_style import INK_MUTED, SERIES, finish, new_axes  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TOTAL_STEPS = 30_000
GAMMA = 0.99
BASE_LR = 1e-3
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 10_000
TARGET_SYNC = 500       # only used by experiment 3
SEEDS = 8
LR_GRID = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
LR_SEEDS = 3
SOLVED = 475.0          # CartPole-v1's official bar: 475 averaged over 100 episodes
LOG_EVERY = 100

# The largest Q-value CartPole can honestly support: reward is +1 per step and the
# episode is capped at 500 steps, so sum_{t<500} gamma^t is a hard ceiling. Any
# prediction above this line is the network hallucinating return that cannot exist.
Q_CEILING = (1 - GAMMA ** 500) / (1 - GAMMA)


class QNet(nn.Module):
    """4 numbers in (cart position/velocity, pole angle/angular velocity), 2 Q-values out."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def run(job):
    """One naive-DQN run. `use_target` is the only stabilizer that can be switched on."""
    lr, seed, use_target = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make("CartPole-v1")
    q = QNet()
    optim = torch.optim.Adam(q.parameters(), lr=lr)
    if use_target:
        q_target = QNet()
        q_target.load_state_dict(q.state_dict())

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_steps, ep_returns, log_steps, q_maxes = [], [], [], []

    for step in range(1, TOTAL_STEPS + 1):
        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * step / EPS_DECAY_STEPS)

        s = torch.as_tensor(obs, dtype=torch.float32)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(q(s).argmax().item())

        nxt, r, term, trunc, _ = env.step(a)
        ep_return += r
        s2 = torch.as_tensor(nxt, dtype=torch.float32)

        # --- the whole algorithm: one TD(0) step on the single newest transition
        with torch.no_grad():
            bootstrap_net = q_target if use_target else q
            # `term` = the pole actually fell, so the future really is worth 0.
            # `trunc` = we hit the 500-step time limit; the episode's value did not
            # end, we just stopped watching, so that case still bootstraps.
            bootstrap = 0.0 if term else GAMMA * bootstrap_net(s2).max().item()
            target = torch.as_tensor(r + bootstrap, dtype=torch.float32)
        pred = q(s)[a]
        loss = F.smooth_l1_loss(pred, target)          # Huber
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        # --- no replay buffer. Batch size one, and the batch is whatever just happened.

        if use_target and step % TARGET_SYNC == 0:
            q_target.load_state_dict(q.state_dict())

        obs = nxt
        if term or trunc:
            ep_returns.append(ep_return)
            ep_steps.append(step)
            ep_return = 0.0
            obs, _ = env.reset()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                log_steps.append(step)
                q_maxes.append(float(q(s).max().item()))

    env.close()
    return {
        "lr": lr, "seed": seed, "use_target": use_target,
        "ep_steps": np.array(ep_steps), "ep_returns": np.array(ep_returns),
        "log_steps": np.array(log_steps), "q_maxes": np.array(q_maxes),
    }


def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else np.asarray(x)


def best_ma(run_, k=10):
    sm = smooth(run_["ep_returns"], k)
    return float(sm.max()) if len(sm) else 0.0


def final_return(run_, k=10):
    ys = run_["ep_returns"]
    return float(ys[-k:].mean()) if len(ys) else 0.0


def plot_learning_curves(runs):
    fig, ax = new_axes(7.6, 4.4)
    for i, r in enumerate(runs):
        ys = smooth(r["ep_returns"])
        if len(ys):
            ax.plot(r["ep_steps"][9:], ys, color=SERIES[i % len(SERIES)],
                    linewidth=1.3, alpha=0.85, label=f"seed {r['seed']}")
    ax.axhline(SOLVED, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, SOLVED + 10, "solved (475)", ha="right",
            color=INK_MUTED, fontsize=9)
    ax.set_ylim(0, 520)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    finish(fig, ax, "Naive DQN never gets off the floor — and it is not close",
           "environment step", "episode return (10-episode moving average)",
           OUT / "naive_learning_curves.png")


def plot_q_divergence(runs):
    fig, ax = new_axes(7.6, 4.4)
    for i, r in enumerate(runs):
        ax.plot(r["log_steps"], np.maximum(r["q_maxes"], 1e-2),
                color=SERIES[i % len(SERIES)], linewidth=1.3, alpha=0.85,
                label=f"seed {r['seed']}")
    ax.set_yscale("log")
    ax.axhline(Q_CEILING, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(TOTAL_STEPS * 0.99, Q_CEILING * 1.6,
            f"true ceiling ≈ {Q_CEILING:.0f}", ha="right", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    finish(fig, ax, "The Q-values give it away first: eight orders of magnitude of fiction",
           "environment step", "max predicted Q(s, ·)  [log scale]",
           OUT / "q_value_divergence.png")


def plot_lr_sweep(sweep):
    """Peak predicted Q and best return, per learning rate, no target network."""
    by_lr = {lr: [r for r in sweep if r["lr"] == lr] for lr in LR_GRID}
    peak_q = [np.mean([r["q_maxes"].max() for r in by_lr[lr]]) for lr in LR_GRID]
    best_r = [np.mean([best_ma(r) for r in by_lr[lr]]) for lr in LR_GRID]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    from plot_style import style_axes
    for ax in (ax1, ax2):
        style_axes(ax)

    ax1.plot(LR_GRID, peak_q, marker="o", color=SERIES[2], linewidth=1.6)
    ax1.axhline(Q_CEILING, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax1.text(LR_GRID[-1], Q_CEILING * 2.2, f"true ceiling ≈ {Q_CEILING:.0f}",
             color=INK_MUTED, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Peak predicted Q", color="#0b0b0b", fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("learning rate", color="#52514e", fontsize=10)
    ax1.set_ylabel("max Q ever predicted  [log]", color="#52514e", fontsize=10)

    ax2.plot(LR_GRID, best_r, marker="o", color=SERIES[0], linewidth=1.6)
    ax2.axhline(SOLVED, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax2.text(LR_GRID[-1], SOLVED - 45, "solved (475)", color=INK_MUTED, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 520)
    ax2.set_title("Best return reached", color="#0b0b0b", fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("learning rate", color="#52514e", fontsize=10)
    ax2.set_ylabel("best 10-episode average", color="#52514e", fontsize=10)

    fig.suptitle("No learning rate rescues it: the step size only sets the fuse length",
                 color="#0b0b0b", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT / "lr_sweep.png"
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_target_teaser(naive, with_target):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    from plot_style import style_axes
    for ax in (ax1, ax2):
        style_axes(ax)

    for label, runs, color in (("naive", naive, SERIES[2]),
                               ("+ target network", with_target, SERIES[1])):
        for j, r in enumerate(runs):
            ys = smooth(r["ep_returns"])
            if len(ys):
                ax1.plot(r["ep_steps"][9:], ys, color=color, linewidth=1.3, alpha=0.8,
                         label=label if j == 0 else None)
            ax2.plot(r["log_steps"], np.maximum(r["q_maxes"], 1e-2), color=color,
                     linewidth=1.3, alpha=0.8, label=label if j == 0 else None)

    ax1.set_ylim(0, 520)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_title("Episode return", color="#0b0b0b", fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("environment step", color="#52514e", fontsize=10)
    ax1.set_ylabel("10-episode moving average", color="#52514e", fontsize=10)

    ax2.set_yscale("log")
    ax2.axhline(Q_CEILING, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax2.text(TOTAL_STEPS * 0.98, Q_CEILING * 2.0, f"true ceiling ≈ {Q_CEILING:.0f}",
             ha="right", color=INK_MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_title("Max predicted Q(s, ·)", color="#0b0b0b", fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("environment step", color="#52514e", fontsize=10)
    ax2.set_ylabel("max Q  [log scale]", color="#52514e", fontsize=10)

    fig.suptitle("One frozen copy of the network is the difference between fiction and physics",
                 color="#0b0b0b", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT / "target_network_teaser.png"
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    naive_jobs = [(BASE_LR, s, False) for s in range(SEEDS)]
    sweep_jobs = [(lr, s, False) for lr in LR_GRID for s in range(LR_SEEDS)
                  if not (lr == BASE_LR and s < SEEDS)]  # reuse the naive runs below
    target_jobs = [(BASE_LR, s, True) for s in range(3)]

    all_jobs = naive_jobs + sweep_jobs + target_jobs
    print(f"{len(all_jobs)} runs x {TOTAL_STEPS:,} steps, in parallel across cores...")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, all_jobs))

    naive = [r for r in results if r["lr"] == BASE_LR and not r["use_target"]]
    with_target = [r for r in results if r["use_target"]]
    sweep = [r for r in results if not r["use_target"]]  # includes the naive runs

    print("\n--- naive DQN (no replay, no target), lr = 1e-3 ---")
    print("seed   best-10ep   final-10ep       peak Q")
    for r in naive:
        print(f"{r['seed']:>4}   {best_ma(r):9.1f}   {final_return(r):10.1f}   "
              f"{r['q_maxes'].max():10.3g}")
    print(f"\nsolved by {sum(best_ma(r) >= SOLVED for r in naive)}/{len(naive)} seeds. "
          f"True Q ceiling is {Q_CEILING:.1f}; the network predicted up to "
          f"{max(r['q_maxes'].max() for r in naive):.3g}.")

    print("\n--- learning-rate sweep (still no target network) ---")
    print("      lr     mean peak Q   mean best return")
    for lr in LR_GRID:
        rs = [r for r in sweep if r["lr"] == lr]
        print(f"{lr:>8.0e}   {np.mean([r['q_maxes'].max() for r in rs]):11.3g}   "
              f"{np.mean([best_ma(r) for r in rs]):16.1f}")

    print("\n--- same agent + a target network (still no replay) ---")
    print("seed   best-10ep   final-10ep       peak Q")
    for r in with_target:
        print(f"{r['seed']:>4}   {best_ma(r):9.1f}   {final_return(r):10.1f}   "
              f"{r['q_maxes'].max():10.3g}")

    plot_learning_curves(naive)
    plot_q_divergence(naive)
    plot_lr_sweep(sweep)
    plot_target_teaser(naive[:3], with_target)


if __name__ == "__main__":
    main()
