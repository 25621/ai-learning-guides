"""
Model-Based Planning: using a learned world model for control on CartPole-v1.

We load the world model trained by world_model.py and use it for online
Model Predictive Control (MPC) via the *random shooting* method (a.k.a.
the simplest form of the Cross-Entropy Method):

  At every real step:
    1. Sample K random action sequences of length H from the action space.
    2. Use the world model to imagine each rollout from the current state.
    3. Score each sequence by its predicted (discounted) return.
    4. Execute only the FIRST action of the best sequence.
    5. Observe the real transition and repeat.

This is the same idea behind classic MPC and the planning loop inside
MuZero / Dreamer — only those use search trees and latent-space models.

We compare three controllers on CartPole-v1:
  - Random policy            (baseline)
  - Greedy on world-model reward, horizon = 1 (myopic)
  - MPC with horizon = 15 over the learned model
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

from world_model import WorldModel   # reuse the architecture

DEVICE = torch.device("cpu")
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CKPT = os.path.join(OUTPUT_DIR, "world_model.pt")


# ----------------------------------------------------------------------
# Planner: random-shooting MPC over the learned model
# ----------------------------------------------------------------------
@torch.no_grad()
def plan_action(model, state, horizon=15, n_samples=200, gamma=0.99, rng=None):
    """Return the best first action by imagining `n_samples` rollouts of length
    `horizon` from `state` through the learned world model."""
    rng = rng or np.random.default_rng()
    n_actions = model.n_actions

    # Sample random action sequences: shape (n_samples, horizon)
    seqs = rng.integers(0, n_actions, size=(n_samples, horizon))

    # Batched imagination
    s = torch.tensor(np.tile(state, (n_samples, 1)), dtype=torch.float32, device=DEVICE)
    alive = torch.ones(n_samples, device=DEVICE)
    returns = torch.zeros(n_samples, device=DEVICE)
    discount = 1.0

    for t in range(horizon):
        a = torch.tensor(seqs[:, t], dtype=torch.long, device=DEVICE)
        delta, r, d_logit = model(s, a)
        s = s + delta
        # In CartPole reward is always 1.0 while alive; trust model's reward
        returns = returns + alive * discount * r
        # Update alive mask: stop accumulating after predicted termination
        not_done = (torch.sigmoid(d_logit) < 0.5).float()
        alive = alive * not_done
        discount *= gamma
        if alive.sum() == 0:
            break

    best = int(torch.argmax(returns).item())
    return int(seqs[best, 0])


# ----------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------
def run_random(env, seed):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    return total


def run_mpc(env, model, seed, horizon, n_samples, rng):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        a = plan_action(model, obs, horizon=horizon, n_samples=n_samples, rng=rng)
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    return total


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def main():
    print("=== Model-Based Planning on CartPole-v1 ===\n")
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"Could not find {CKPT}. Please run world_model.py first.")

    model = WorldModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f"Loaded world model from {CKPT}\n")

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(SEED)

    n_eval = 10
    seeds = [100 + i for i in range(n_eval)]

    configs = [
        ("Random policy",        dict(controller="random")),
        ("MPC  horizon=1   K=50",  dict(controller="mpc", horizon=1,  n_samples=50)),
        ("MPC  horizon=15  K=200", dict(controller="mpc", horizon=15, n_samples=200)),
    ]

    results = {}
    for name, cfg in configs:
        print(f"Evaluating: {name}")
        rewards = []
        for s in seeds:
            if cfg["controller"] == "random":
                r = run_random(env, s)
            else:
                r = run_mpc(env, model, s, cfg["horizon"], cfg["n_samples"], rng)
            rewards.append(r)
            print(f"   seed {s:3d}: reward = {r:6.1f}")
        results[name] = rewards
        print(f"   → mean ± std: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}\n")

    env.close()

    # ---- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    means = [np.mean(results[n]) for n in names]
    stds = [np.std(results[n]) for n in names]
    colors = ["#95a5a6", "#f1c40f", "#2ecc71"]
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, edgecolor="black")
    ax.axhline(500, color="green", linestyle="--", linewidth=1, label="Max reward (500)")
    ax.set_ylabel("Episode reward (avg over 10 seeds)")
    ax.set_title("Planning with a learned world model beats reactive policies")
    ax.set_ylim(0, 520)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "model_based_planning.png")
    plt.savefig(out, dpi=120)
    print(f"Plot saved to {out}")

    print("\nWhat to notice:")
    print("  • A random policy keeps the pole up only briefly (~20 steps).")
    print("  • Horizon=1 MPC is myopic — it only looks 1 step ahead, so it")
    print("    can't recover from a tilt that takes several steps to fix.")
    print("  • Horizon=15 MPC plans through the LEARNED model and can")
    print("    balance the pole far longer — using zero new training and")
    print("    zero real-environment trial-and-error.")


if __name__ == "__main__":
    main()
