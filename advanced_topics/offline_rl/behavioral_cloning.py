"""
Behavioral Cloning (BC) on the D4RL-style CartPole datasets.

Behavioral cloning is the most direct offline-RL baseline:

    Treat the dataset as a supervised classification problem.
    Features = state.   Label = action.
    Train a network with cross-entropy.

That's it.  No Bellman update, no value function, no reward signal — BC
ignores the reward column entirely.  It just clones the behaviour policy
that produced the data.

Why care about BC?
------------------
1. It is the trivial baseline every offline-RL paper must beat.
2. On *expert* data it is shockingly hard to beat: if the data is perfect,
   cloning is enough.
3. On *random* or *medium-replay* data it falls apart, because BC mimics
   the bad behaviours just as faithfully as the good ones.  An algorithm
   that uses the reward signal (like CQL) can extract a policy *better*
   than the average data quality.

So BC vs CQL on the SAME dataset is the cleanest demo of "why we need
reward-aware offline RL at all".

This script
-----------
For each of the four datasets (random / medium / expert / medium-replay):
  1. Train a BC classifier (state -> action distribution) for N gradient steps.
  2. Evaluate greedily in the real environment.

The plot at the end shows BC's evaluation return as a function of dataset
quality — the classic "BC tracks the data" curve.

You can later overlay CQL's numbers from `cql.py` to see where CQL wins.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENV_ID = "CartPole-v1"
OBS_DIM = 4
N_ACTIONS = 2


# ---------------------------------------------------------------------------
class BCPolicy(nn.Module):
    """Tiny MLP outputting action logits."""
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(name):
    path = os.path.join(OUTPUT_DIR, f"cartpole_{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset: {path}\nRun `python d4rl_dataset.py` first.")
    d = np.load(path)
    return {k: d[k] for k in ["obs", "action", "reward", "next_obs", "terminal"]}


def evaluate(policy, n_episodes=20, seed=999):
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    policy.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done, ep_r = False, 0.0
        while not done:
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = int(logits.argmax(1).item())
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        returns.append(ep_r)
    env.close()
    policy.train()
    return float(np.mean(returns)), float(np.std(returns))


def train_bc(data, n_steps=10_000, batch_size=256, lr=3e-4,
             eval_every=2_500, label="BC"):
    rng = np.random.default_rng(SEED)
    obs = torch.tensor(data["obs"])
    act = torch.tensor(data["action"])
    N = obs.shape[0]

    policy = BCPolicy()
    opt = optim.Adam(policy.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    eval_curve = []
    loss_history = []

    print(f"\n  [{label}]  training on {N} transitions for {n_steps} steps ...")
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, N, size=batch_size)
        logits = policy(obs[idx])
        loss = ce(logits, act[idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(float(loss.item()))

        if step % eval_every == 0 or step == n_steps:
            mu, sd = evaluate(policy)
            eval_curve.append((step, mu, sd))
            print(f"    step {step:>5}  |  ce {loss.item():.4f}  "
                  f"|  eval return {mu:.1f} ± {sd:.1f}")

    return {"policy": policy, "eval_curve": eval_curve,
            "loss_history": loss_history}


# ---------------------------------------------------------------------------
def main():
    print("=== Behavioral Cloning on the D4RL-style CartPole datasets ===")

    datasets = {name: load_dataset(name)
                for name in ["random", "medium", "expert", "medium-replay"]}

    # Quick sanity print
    print("\nDataset sizes:")
    for n, d in datasets.items():
        # average return implicit in the data: sum of rewards / number of
        # episodes (= number of terminals + 1)
        n_eps = max(int(d["terminal"].sum()), 1)
        avg_ret = float(d["reward"].sum()) / n_eps
        print(f"  {n:<14}  N={d['obs'].shape[0]:>5}   avg episode return ~ {avg_ret:.1f}")

    runs = {}
    for name in ["random", "medium", "expert", "medium-replay"]:
        runs[name] = train_bc(datasets[name], n_steps=10_000,
                              label=f"BC on {name}")

    print("\nFinal evaluation returns:")
    summary_rows = []
    for name, run in runs.items():
        _, mu, sd = run["eval_curve"][-1]
        summary_rows.append((name, mu, sd))
        print(f"  BC on {name:<15} ->  {mu:6.1f} ± {sd:5.1f}")

    # ----- Plot 1: BC return vs dataset quality -----------------------------
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(13, 4.6))

    names = [r[0] for r in summary_rows]
    means = [r[1] for r in summary_rows]
    stds = [r[2] for r in summary_rows]
    colors = ["#95a5a6", "#3498db", "#27ae60", "#e67e22"]
    bars = ax_bar.bar(names, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black")
    ax_bar.axhline(500, color="#7f8c8d", linestyle="--", lw=1, label="env max")
    ax_bar.set_ylabel("Evaluation return (avg of 20 episodes)")
    ax_bar.set_title("Behavioral Cloning: return tracks dataset quality")
    ax_bar.set_ylim(0, 540)
    for bar, m in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, m + 12,
                    f"{m:.0f}", ha="center", fontsize=10)
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.legend()

    # ----- Plot 2: BC learning curve per dataset ----------------------------
    for (name, run), c in zip(runs.items(), colors):
        xs = [x for x, _, _ in run["eval_curve"]]
        ys = [y for _, y, _ in run["eval_curve"]]
        ax_curve.plot(xs, ys, marker="o", color=c, linewidth=2, label=name)
    ax_curve.axhline(500, color="#7f8c8d", linestyle="--", lw=1)
    ax_curve.set_xlabel("Gradient step")
    ax_curve.set_ylabel("Evaluation return")
    ax_curve.set_title("BC learning curves")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "bc.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")

    # ----- BC vs CQL comparison ---------------------------------------------
    print("\n--- BC vs CQL on the SAME dataset ('medium-replay') ---")
    print("BC's final return on medium-replay:  "
          f"{dict((n, m) for n, m, _ in summary_rows)['medium-replay']:.1f}")
    print("(Run `python cql.py` to see CQL's number on the same dataset; "
          "CQL typically reaches 400+ while BC stays around the average "
          "data quality.)")


if __name__ == "__main__":
    main()
