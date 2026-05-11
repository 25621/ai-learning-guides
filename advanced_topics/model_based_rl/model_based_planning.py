"""
Model-Based Planning with MPC (Random Shooting) on CartPole-v1.

We reuse the world model trained in world_model.py.

Random-Shooting Model Predictive Control
----------------------------------------
At every real step:
    1. Sample N candidate action sequences of horizon H (uniformly at random).
    2. Use the learned world model to simulate each sequence forward from the
       current state, scoring it by a shaped reward (upright pole, centred cart).
    3. Execute the FIRST action of the best-scoring sequence in the real env.
    4. Observe the real next state and repeat (receding-horizon).

Why a shaped reward?  The world model only predicts (next_state, reward) and
in CartPole the reward is +1 every step until the episode terminates.  A flat
+1 signal gives the planner no gradient between sequences, so we replace it
with a smooth proxy that rewards small pole angles and small cart drift.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
import torch

# Reuse the WorldModel + data utilities defined in world_model.py
sys.path.append(os.path.dirname(__file__))
from world_model import WorldModel, collect_data, train_world_model  # noqa: E402


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def reward_proxy(states):
    """Higher when the pole is upright and the cart is centred.

    CartPole terminates at |pole_angle| > 12 deg ~= 0.21 rad, or |cart_pos| > 2.4.
    We give 1.0 when perfectly upright and centred, falling toward 0 at failure.
    """
    cart_pos = states[..., 0]
    pole_angle = states[..., 2]
    return 1.0 - (pole_angle.abs() / 0.21) - 0.1 * (cart_pos.abs() / 2.4)


def mpc_action(model, state, n_samples=200, horizon=15, n_actions=2, rng=None):
    """Random-shooting MPC: return the first action of the best sampled plan."""
    if rng is None:
        rng = np.random.default_rng()
    acts = rng.integers(0, n_actions, size=(n_samples, horizon))  # (N, H)

    s = (torch.tensor(state, dtype=torch.float32)
         .unsqueeze(0).expand(n_samples, -1).contiguous())
    total_reward = torch.zeros(n_samples)
    with torch.no_grad():
        for t in range(horizon):
            a_t = torch.tensor(acts[:, t], dtype=torch.int64)
            s, _ = model(s, a_t)
            total_reward += reward_proxy(s)
    best = int(torch.argmax(total_reward).item())
    return int(acts[best, 0])


def run_mpc_episode(model, env, n_samples=200, horizon=15, seed=0, max_steps=500):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(max_steps):
        a = mpc_action(model, obs, n_samples=n_samples, horizon=horizon, rng=rng)
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    return total


def run_random_episode(env, seed=0, max_steps=500):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(max_steps):
        a = int(rng.integers(2))
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    return total


# ---------------------------------------------------------------------------
def main():
    print("=== Model-Based Planning (MPC w/ Random Shooting) on CartPole ===\n")

    model = WorldModel()
    ckpt = os.path.join(OUTPUT_DIR, "world_model.pt")
    if os.path.exists(ckpt):
        print(f"Loading world model from {ckpt}")
        model.load_state_dict(torch.load(ckpt))
    else:
        print("No saved world model found - training one now...")
        data = collect_data(n_transitions=20_000)
        train_world_model(model, data, n_epochs=30, verbose=False)
        torch.save(model.state_dict(), ckpt)
        print(f"Saved trained model to {ckpt}")
    model.eval()

    env = gym.make("CartPole-v1")
    n_eps = 20
    print(f"\nRunning {n_eps} episodes: MPC (N=200, H=15) vs Random policy...\n")

    mpc_rewards, rand_rewards = [], []
    for ep in range(n_eps):
        mr = run_mpc_episode(model, env, n_samples=200, horizon=15, seed=ep)
        rr = run_random_episode(env, seed=ep)
        mpc_rewards.append(mr)
        rand_rewards.append(rr)
        print(f"  ep {ep+1:2d}:   MPC = {mr:5.0f}    Random = {rr:5.0f}")
    env.close()

    print(f"\nMPC    avg reward : {np.mean(mpc_rewards):6.1f}  "
          f"(std {np.std(mpc_rewards):5.1f})")
    print(f"Random avg reward : {np.mean(rand_rewards):6.1f}  "
          f"(std {np.std(rand_rewards):5.1f})")

    # Plot ---------------------------------------------------------------
    plt.figure(figsize=(11, 5))
    x = np.arange(n_eps)
    plt.bar(x - 0.2, mpc_rewards, 0.4,
            color="#27ae60", label=f"MPC  (avg {np.mean(mpc_rewards):.0f})")
    plt.bar(x + 0.2, rand_rewards, 0.4,
            color="#95a5a6", label=f"Random  (avg {np.mean(rand_rewards):.0f})")
    plt.axhline(500, color="gold", linestyle="--", lw=1, label="Max possible (500)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps survived)")
    plt.title("Planning with a learned world model beats a random policy")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "model_based_planning.png")
    plt.savefig(out, dpi=120)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
