"""
Train a world model on CartPole-v1.

A "world model" is a neural network that learns to predict the environment's
dynamics:
        (state, action)  ->  (next_state, reward)

Once trained, the model lets us simulate experience without touching the real
environment — useful for planning (see model_based_planning.py) or for
Dyna-style data augmentation.

Architecture
------------
  Input  : 4-d state  ++  2-d one-hot action       =>  6-d
  Hidden : 2 x 128 units, ReLU
  Output : 4-d delta state  ++  1-d reward         =>  5-d

We predict the *delta* (next_state - state) rather than next_state directly
— this is a standard trick that keeps the targets near zero and stabilises
training when state magnitudes vary.

Data collection
---------------
We roll out a uniform random policy for ~20k transitions.  Random data
covers a wide region of state space, which is what we want for an
unbiased one-step dynamics model.

Evaluation
----------
1. Hold-out validation MSE  (single-step accuracy)
2. k-step rollout error     — feed the model's own prediction back in for
                              k steps and measure how the error compounds.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    """MLP dynamics model: predicts delta-state and reward."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),  # delta_state (4) + reward (1)
        )

    def forward(self, state, action):
        """state: (B, obs_dim) float ;  action: (B,) int64."""
        a_oh = torch.zeros(state.shape[0], self.n_actions, device=state.device)
        a_oh.scatter_(1, action.long().unsqueeze(1), 1.0)
        x = torch.cat([state, a_oh], dim=-1)
        out = self.net(x)
        delta = out[:, : self.obs_dim]
        reward = out[:, self.obs_dim]
        return state + delta, reward


# ---------------------------------------------------------------------------
def collect_data(n_transitions=20_000, seed=0):
    """Roll out a random policy and store transitions."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    states, actions, rewards, next_states = [], [], [], []
    for _ in range(n_transitions):
        a = int(rng.integers(2))
        nobs, r, terminated, truncated, _ = env.step(a)
        states.append(obs)
        actions.append(a)
        rewards.append(r)
        next_states.append(nobs)
        obs = nobs
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
    )


def train_world_model(model, data, n_epochs=30, batch_size=256, lr=1e-3,
                      val_frac=0.1, verbose=True):
    s, a, r, ns = data
    n = len(s)
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    s_t = torch.tensor(s)
    a_t = torch.tensor(a)
    r_t = torch.tensor(r)
    ns_t = torch.tensor(ns)

    opt = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_idx)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[i:i + batch_size]
            pred_ns, pred_r = model(s_t[idx], a_t[idx])
            loss = (nn.functional.mse_loss(pred_ns, ns_t[idx])
                    + nn.functional.mse_loss(pred_r, r_t[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        train_loss = ep_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pred_ns, pred_r = model(s_t[val_idx], a_t[val_idx])
            val_loss = (nn.functional.mse_loss(pred_ns, ns_t[val_idx])
                        + nn.functional.mse_loss(pred_r, r_t[val_idx])).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"Epoch {epoch+1:2d} | train MSE {train_loss:.5f} "
                  f"| val MSE {val_loss:.5f}")
    return history


def evaluate_rollout(model, horizon=20, n_rollouts=50, seed=99):
    """Compare k-step model rollouts to the real environment under random actions."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    errs = np.zeros(horizon)
    counts = np.zeros(horizon)
    model.eval()
    for _ in range(n_rollouts):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        sim = obs.copy()
        for t in range(horizon):
            a = int(rng.integers(2))
            real_next, _, term, trunc, _ = env.step(a)
            with torch.no_grad():
                pred_next, _ = model(
                    torch.tensor(sim, dtype=torch.float32).unsqueeze(0),
                    torch.tensor([a], dtype=torch.int64),
                )
            pred_next = pred_next.squeeze(0).numpy()
            errs[t] += float(np.linalg.norm(real_next - pred_next))
            counts[t] += 1
            sim = pred_next
            if term or trunc:
                break
    env.close()
    return errs / np.maximum(counts, 1)


# ---------------------------------------------------------------------------
def main():
    print("=== Training a World Model on CartPole-v1 ===\n")

    print("Step 1: Collecting 20,000 transitions with a random policy...")
    data = collect_data(n_transitions=20_000)
    print(f"  collected {len(data[0])} transitions")

    print("\nStep 2: Training MLP world model (30 epochs)...")
    model = WorldModel()
    history = train_world_model(model, data, n_epochs=30)

    print("\nStep 3: k-step prediction error (random actions, horizon=20)...")
    err = evaluate_rollout(model, horizon=20, n_rollouts=50)
    for t in [0, 4, 9, 19]:
        print(f"  step {t+1:2d}: avg L2 state error = {err[t]:.4f}")

    ckpt = os.path.join(OUTPUT_DIR, "world_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\nWorld-model weights saved to {ckpt}")

    # Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="train MSE", color="#3498db", lw=2)
    axes[0].plot(history["val_loss"], label="val MSE", color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (delta + reward)")
    axes[0].set_title("World-Model Training Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, len(err) + 1), err,
                 color="#9b59b6", lw=2, marker="o")
    axes[1].set_xlabel("Rollout step (k)")
    axes[1].set_ylabel("Avg ||real - predicted||  (L2)")
    axes[1].set_title("Compounding Error of k-step Rollouts")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "world_model.png")
    plt.savefig(out, dpi=120)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
