"""
World Model for CartPole-v1

A "world model" is a neural network that learns to predict the environment's dynamics:
    given (state, action), output (next_state, reward, done?)

Once trained, we can roll the model forward to *imagine* trajectories without
ever calling the real simulator — the foundation of model-based RL methods
like Dyna, MPC, MuZero, and Dreamer.

This script:
  1. Collects a buffer of random transitions from CartPole-v1.
  2. Trains an MLP to predict delta_state, reward, and done-probability.
  3. Evaluates open-loop rollout error vs the real env at horizons 1, 5, 20.
  4. Saves the trained model to outputs/world_model.pt so model_based_planning.py
     can load it.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. The world model
# ----------------------------------------------------------------------
class WorldModel(nn.Module):
    """Predicts (Δstate, reward, done_logit) from (state, action_one_hot)."""

    def __init__(self, n_obs=4, n_actions=2, hidden=128):
        super().__init__()
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(n_obs + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, n_obs)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)

    def forward(self, state, action):
        """state: (B, n_obs), action: (B,) int64 — returns predictions."""
        a_onehot = torch.nn.functional.one_hot(action, self.n_actions).float()
        x = torch.cat([state, a_onehot], dim=-1)
        h = self.trunk(x)
        return self.delta_head(h), self.reward_head(h).squeeze(-1), self.done_head(h).squeeze(-1)

    @torch.no_grad()
    def step(self, state, action):
        """Single-step imagination. state: (n_obs,) np or tensor, action: int."""
        s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = torch.tensor([action], dtype=torch.long, device=DEVICE)
        delta, r, d_logit = self.forward(s, a)
        next_state = (s + delta).squeeze(0).cpu().numpy()
        reward = r.item()
        done = torch.sigmoid(d_logit).item() > 0.5
        return next_state, reward, done


# ----------------------------------------------------------------------
# 2. Data collection
# ----------------------------------------------------------------------
def collect_transitions(n_steps=20_000, seed=SEED):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)

    s_buf, a_buf, sn_buf, r_buf, d_buf = [], [], [], [], []
    for _ in range(n_steps):
        a = env.action_space.sample()
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        s_buf.append(obs)
        a_buf.append(a)
        sn_buf.append(next_obs)
        r_buf.append(r)
        d_buf.append(float(term))   # only "real" terminations, not time limits
        obs = next_obs
        if done:
            obs, _ = env.reset()

    env.close()
    return (
        np.array(s_buf, dtype=np.float32),
        np.array(a_buf, dtype=np.int64),
        np.array(sn_buf, dtype=np.float32),
        np.array(r_buf, dtype=np.float32),
        np.array(d_buf, dtype=np.float32),
    )


# ----------------------------------------------------------------------
# 3. Training
# ----------------------------------------------------------------------
def train_world_model(model, data, n_epochs=30, batch=256, lr=1e-3):
    s, a, sn, r, d = data
    delta = sn - s

    s_t = torch.tensor(s, device=DEVICE)
    a_t = torch.tensor(a, device=DEVICE)
    delta_t = torch.tensor(delta, device=DEVICE)
    r_t = torch.tensor(r, device=DEVICE)
    d_t = torch.tensor(d, device=DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = len(s_t)

    history = []
    for epoch in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        losses = []
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            pd, pr, pdone = model(s_t[b], a_t[b])
            loss_d = ((pd - delta_t[b]) ** 2).mean()
            loss_r = ((pr - r_t[b]) ** 2).mean()
            loss_done = bce(pdone, d_t[b])
            loss = loss_d + loss_r + 0.5 * loss_done
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = float(np.mean(losses))
        history.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d} | train loss: {avg:.5f}")
    return history


# ----------------------------------------------------------------------
# 4. Evaluation: open-loop rollout error vs the real env
# ----------------------------------------------------------------------
def rollout_error(model, horizons=(1, 5, 20), n_trials=50, seed=999):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    errors = {h: [] for h in horizons}

    for trial in range(n_trials):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        actions = [int(rng.integers(2)) for _ in range(max(horizons))]

        # Real rollout
        real_states = [obs.copy()]
        cur = obs.copy()
        for a in actions:
            cur, _, term, trunc, _ = env.step(a)
            real_states.append(cur.copy())
            if term or trunc:
                break

        # Imagined rollout from the world model
        sim = obs.copy()
        sim_states = [sim.copy()]
        for a in actions:
            sim, _, sim_done = model.step(sim, a)
            sim_states.append(sim.copy())
            if sim_done:
                break

        # Compare at each horizon (use min length so we don't index beyond done)
        usable = min(len(real_states), len(sim_states))
        for h in horizons:
            if h < usable:
                err = float(np.linalg.norm(real_states[h] - sim_states[h]))
                errors[h].append(err)

    env.close()
    return {h: (np.mean(e) if e else float("nan")) for h, e in errors.items()}


def plot_history(history, eval_errors):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(history, color="#3498db", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE + BCE training loss")
    axes[0].set_title("World model training loss")
    axes[0].grid(alpha=0.3)

    hs = list(eval_errors.keys())
    vs = [eval_errors[h] for h in hs]
    axes[1].bar([str(h) for h in hs], vs, color=["#2ecc71", "#f1c40f", "#e74c3c"])
    axes[1].set_xlabel("Rollout horizon (steps)")
    axes[1].set_ylabel("Mean ‖s_real − s_imagined‖")
    axes[1].set_title("Open-loop prediction error vs horizon")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "world_model.png")
    plt.savefig(out, dpi=120)
    print(f"Plot saved to {out}")


def main():
    print("=== World Model for CartPole-v1 ===\n")
    print("1) Collecting 20,000 random transitions...")
    data = collect_transitions(n_steps=20_000)
    print(f"   collected {len(data[0])} transitions, "
          f"termination rate: {data[4].mean()*100:.1f}%")

    print("\n2) Training world model (MLP)...")
    model = WorldModel().to(DEVICE)
    history = train_world_model(model, data, n_epochs=30)

    print("\n3) Evaluating open-loop rollout error...")
    eval_errors = rollout_error(model)
    for h, e in eval_errors.items():
        print(f"   horizon {h:3d} steps | mean L2 error: {e:.4f}")

    plot_history(history, eval_errors)

    ckpt_path = os.path.join(OUTPUT_DIR, "world_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved trained world model to {ckpt_path}")
    print("→ Run model_based_planning.py next to USE this model for control.")


if __name__ == "__main__":
    main()
