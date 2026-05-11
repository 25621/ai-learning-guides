"""
Conservative Q-Learning (CQL) for offline RL on CartPole-v1.

The big problem with naive offline Q-learning
---------------------------------------------
In the standard Q-learning Bellman target

    target(s, a) = r + gamma * max_{a'} Q(s', a')

the `max_{a'}` can pick an action `a'` that the *dataset never recorded*.
For that out-of-distribution action the network's Q-value is whatever
random number the MLP happens to extrapolate to — usually too high.  The
target inherits that lie, and the next iteration of Q-learning believes
even harder in fictional good actions.  This is **distribution shift**, the
core failure mode of offline RL.

CQL's fix (Kumar et al., 2020)
------------------------------
Add a penalty to the loss that *pushes down* Q for actions NOT in the data
and *pushes up* Q for actions that ARE in the data:

    cql_loss = logsumexp_a Q(s, a)  -  Q(s, a_dataset)

The logsumexp is a soft `max` over all actions — penalising it shrinks Q on
out-of-distribution actions.  Subtracting Q on the dataset action protects
in-distribution Q-values.  The full objective is

    L = Bellman_loss  +  alpha * cql_loss

With alpha large enough, the learned Q-function is provably a *lower bound*
on the true Q-function, so `argmax Q` no longer chases hallucinated optima.

This script
-----------
1. Loads the four datasets built by `d4rl_dataset.py`.
2. Trains three Q-networks PURELY OFFLINE (no environment steps) on the
   `medium-replay` dataset:
       (a) naive offline DQN     (alpha = 0)
       (b) CQL with alpha = 1.0
       (c) CQL with alpha = 5.0
3. Evaluates each in the real environment every 5k gradient steps.
4. Plots the three learning curves side by side.

You should see (b) and (c) climb to near 500 reward while (a) stays stuck
or oscillates — the textbook distribution-shift collapse.
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
DEVICE = torch.device("cpu")

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENV_ID = "CartPole-v1"
OBS_DIM = 4
N_ACTIONS = 2


# ---------------------------------------------------------------------------
class QNet(nn.Module):
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


# ---------------------------------------------------------------------------
def evaluate(qnet, n_episodes=10, seed=999):
    """Greedy rollout in the real env.  This is the ONLY env interaction we
    do — for evaluation only, never for training."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    qnet.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done, ep_r = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(qnet(torch.tensor(obs, dtype=torch.float32)
                             .unsqueeze(0)).argmax(1).item())
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        returns.append(ep_r)
    env.close()
    qnet.train()
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
def train_cql(data, alpha_cql, n_steps=30_000, batch_size=256,
              lr=3e-4, gamma=0.99, target_sync_every=200,
              eval_every=2_500, label="CQL"):
    """Offline Q-learning.  `alpha_cql=0` recovers naive offline DQN."""
    rng = np.random.default_rng(SEED)

    q = QNet()
    q_tgt = QNet()
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)

    obs = torch.tensor(data["obs"])
    act = torch.tensor(data["action"])
    rew = torch.tensor(data["reward"])
    nobs = torch.tensor(data["next_obs"])
    term = torch.tensor(data["terminal"], dtype=torch.float32)
    N = obs.shape[0]

    eval_returns = []
    cql_losses = []
    bellman_losses = []

    print(f"\n  Training [{label}] for {n_steps} gradient steps on "
          f"{N} transitions ...")
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, N, size=batch_size)
        s, a, r, ns, te = obs[idx], act[idx], rew[idx], nobs[idx], term[idx]

        with torch.no_grad():
            q_next = q_tgt(ns).max(1).values
            target = r + gamma * (1.0 - te) * q_next

        q_all = q(s)                                  # (B, A)
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        bellman_loss = nn.functional.mse_loss(q_taken, target)

        # ----- CQL penalty ---------------------------------------------------
        # log sum_a exp Q(s, a)  -  Q(s, a_dataset)
        # Pushes Q down on un-taken (out-of-distribution) actions,
        # pulls   Q up   on taken (in-distribution) actions.
        cql_term = torch.logsumexp(q_all, dim=1) - q_taken
        cql_loss = cql_term.mean()

        loss = bellman_loss + alpha_cql * cql_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()

        if step % target_sync_every == 0:
            q_tgt.load_state_dict(q.state_dict())

        bellman_losses.append(float(bellman_loss.item()))
        cql_losses.append(float(cql_loss.item()))

        if step % eval_every == 0 or step == n_steps:
            avg_ret = evaluate(q)
            eval_returns.append((step, avg_ret))
            print(f"    step {step:>5}  |  bellman {bellman_loss.item():.3f}  "
                  f"|  cql {cql_loss.item():.3f}  |  eval return {avg_ret:.1f}")

    return {
        "q": q,
        "eval_returns": eval_returns,
        "bellman_losses": bellman_losses,
        "cql_losses": cql_losses,
    }


# ---------------------------------------------------------------------------
def main():
    print("=== Conservative Q-Learning on CartPole (offline) ===")

    print("\nLoading datasets built by d4rl_dataset.py ...")
    datasets = {name: load_dataset(name)
                for name in ["random", "medium", "expert", "medium-replay"]}
    for name, d in datasets.items():
        print(f"  {name:<14}  {d['obs'].shape[0]} transitions")

    # We focus on medium-replay because it is the most interesting:
    # mixed-quality data where naive Q-learning is expected to fail.
    train_data = datasets["medium-replay"]
    print(f"\nUsing 'medium-replay' as the training dataset "
          f"({train_data['obs'].shape[0]} transitions).")

    runs = {}
    for alpha, label in [(0.0, "naive offline DQN (alpha=0)"),
                         (1.0, "CQL (alpha=1.0)"),
                         (5.0, "CQL (alpha=5.0)")]:
        runs[label] = train_cql(train_data, alpha_cql=alpha,
                                n_steps=30_000, label=label)

    print("\nFinal evaluation returns (avg over 10 episodes, greedy):")
    final_table = []
    for label, run in runs.items():
        _, last_ret = run["eval_returns"][-1]
        final_table.append((label, last_ret))
        print(f"  {label:<35}  ->  {last_ret:6.1f}")

    # ----- Plot --------------------------------------------------------------
    fig, (ax_ret, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.6))

    colors = {
        "naive offline DQN (alpha=0)": "#e74c3c",
        "CQL (alpha=1.0)": "#f39c12",
        "CQL (alpha=5.0)": "#27ae60",
    }
    for label, run in runs.items():
        xs, ys = zip(*run["eval_returns"])
        ax_ret.plot(xs, ys, marker="o", linewidth=2, color=colors[label],
                    label=label)
    ax_ret.axhline(500, color="#7f8c8d", linestyle="--", linewidth=1,
                   label="env max (500)")
    ax_ret.set_xlabel("Gradient step (offline)")
    ax_ret.set_ylabel("Evaluation return (greedy, 10 eps)")
    ax_ret.set_title("CQL vs naive offline DQN on medium-replay")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend(loc="lower right", fontsize=9)

    for label, run in runs.items():
        ax_loss.plot(np.convolve(run["bellman_losses"],
                                 np.ones(200) / 200, mode="valid"),
                     color=colors[label], linewidth=1.6, label=label)
    ax_loss.set_xlabel("Gradient step")
    ax_loss.set_ylabel("Bellman MSE (smoothed)")
    ax_loss.set_title("Bellman loss during training")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.3, which="both")
    ax_loss.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cql.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
