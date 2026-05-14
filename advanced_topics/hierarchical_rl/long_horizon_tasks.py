"""
Long-Horizon Tasks: Comparing flat vs hierarchical agents.

A flat DQN must discover a multi-step structure from a single sparse
reward.  A hierarchical agent uses a high-level manager (picks a subgoal)
and a low-level worker (navigates to it), with intrinsic rewards at each
subgoal — dramatically accelerating learning.

Environment: open 9x9 grid where the agent must:
  1. Collect a KEY (position K)
  2. Then reach the DOOR (position D)
The episode only gives +1 at the very end (after both steps).

The flat DQN must discover the joint sequence from this one reward.
The hierarchical agent gets +1 for reaching K, and +1 for reaching D
separately — two short tasks instead of one long compound task.
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = [None] * capacity
        self.cap = capacity
        self.idx = 0
        self.size = 0

    def add(self, *t):
        self.buf[self.idx] = t
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, k):
        indices = random.sample(range(self.size), k)
        return zip(*[self.buf[i] for i in indices])

    def __len__(self):
        return self.size

# ─── Environment ─────────────────────────────────────────────────────────────

SIZE = 9
# Only outer walls (open interior — navigation is easy)
WALLS = set()
for i in range(SIZE):
    WALLS.add((0, i)); WALLS.add((SIZE - 1, i))
    WALLS.add((i, 0)); WALLS.add((i, SIZE - 1))

FREE = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)
KEY_POS  = (2, 2)
DOOR_POS = (6, 6)


def enc(r, c):
    return r * SIZE + c


N_BASE = SIZE * SIZE
N_STATES_FLAT = N_BASE * 2   # (position, has_key)
SUBGOALS = [KEY_POS, DOOR_POS]
N_STATES_WORKER = N_BASE * len(SUBGOALS)


class KeyDoorEnv:
    def reset(self):
        # Start in a random free cell away from key and door
        candidates = [c for c in FREE if c not in (KEY_POS, DOOR_POS)]
        self.pos = random.choice(candidates)
        self.has_key = False
        return self._state()

    def _state(self):
        return enc(*self.pos) + (N_BASE if self.has_key else 0)

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if (nr, nc) not in WALLS and 0 <= nr < SIZE and 0 <= nc < SIZE:
            self.pos = (nr, nc)
        if not self.has_key and self.pos == KEY_POS:
            self.has_key = True
        done = self.has_key and (self.pos == DOOR_POS)
        reward = 1.0 if done else 0.0
        return self._state(), reward, done

# ─── DQN helpers ──────────────────────────────────────────────────────────────

class DQNNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_states, 32)
        self.net = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(self.emb(x))


def make_agent(n_states, n_actions, lr=5e-4):
    net = DQNNet(n_states, n_actions)
    tgt = DQNNet(n_states, n_actions)
    tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=lr)
    buf = ReplayBuffer(8_000)
    return net, tgt, opt, buf


def act(net, s, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        return net(torch.tensor([s])).argmax().item()


def update(net, tgt, opt, buf, batch=64, gamma=0.99):
    if len(buf) < batch:
        return
    s, a, r, ns, d = buf.sample(batch)
    s  = torch.tensor(list(s),  dtype=torch.long)
    a  = torch.tensor(list(a),  dtype=torch.long)
    r  = torch.tensor(list(r),  dtype=torch.float)
    ns = torch.tensor(list(ns), dtype=torch.long)
    d  = torch.tensor(list(d),  dtype=torch.float)
    with torch.no_grad():
        q_tgt = r + gamma * tgt(ns).max(1).values * (1 - d)
    q = net(s).gather(1, a.unsqueeze(1)).squeeze()
    loss = nn.functional.mse_loss(q, q_tgt)
    opt.zero_grad(); loss.backward(); opt.step()

# ─── Flat DQN ─────────────────────────────────────────────────────────────────

def train_flat(n_episodes=3000, max_steps=100):
    env = KeyDoorEnv()
    net, tgt, opt, buf = make_agent(N_STATES_FLAT, N_ACTIONS)
    history = []

    for ep in range(n_episodes):
        s = env.reset()
        eps = max(0.05, 0.6 * (1 - ep / n_episodes))
        done = False; step = 0; success = False

        while not done and step < max_steps:
            a = act(net, s, eps, N_ACTIONS)
            ns, r, done = env.step(a)
            buf.add(s, a, r, ns, done)
            update(net, tgt, opt, buf)
            s = ns; step += 1
            if done: success = True

        if (ep + 1) % 250 == 0:
            tgt.load_state_dict(net.state_dict())

        history.append(float(success))
        if (ep + 1) % 500 == 0:
            print(f"[Flat]  ep {ep+1:4d} | "
                  f"success={np.mean(history[-200:]):.3f}", flush=True)

    return history

# ─── Hierarchical ─────────────────────────────────────────────────────────────

def worker_s(pos, sg_idx):
    return enc(*pos) + N_BASE * sg_idx


def train_hierarchical(n_episodes=3000, max_steps=100):
    env = KeyDoorEnv()
    wnet, wtgt, wopt, wbuf = make_agent(N_STATES_WORKER, N_ACTIONS)
    history = []

    for ep in range(n_episodes):
        env.reset()
        pos = env.pos
        eps = max(0.05, 0.6 * (1 - ep / n_episodes))
        done = False; step = 0

        while not done and step < max_steps:
            sg_idx = 0 if not env.has_key else 1
            sg_pos = SUBGOALS[sg_idx]
            ws  = worker_s(pos, sg_idx)
            a   = act(wnet, ws, eps, N_ACTIONS)
            _, env_r, done = env.step(a)
            npos = env.pos
            nws  = worker_s(npos, sg_idx)
            # intrinsic: +1 if reached current subgoal
            intr_r = 1.0 if npos == sg_pos else 0.0
            wdone  = (npos == sg_pos) or done
            wbuf.add(ws, a, intr_r, nws, wdone)
            update(wnet, wtgt, wopt, wbuf)
            pos = npos; step += 1

        if (ep + 1) % 250 == 0:
            wtgt.load_state_dict(wnet.state_dict())

        success = env.has_key and (env.pos == DOOR_POS)
        history.append(float(success))
        if (ep + 1) % 500 == 0:
            print(f"[Hier] ep {ep+1:4d} | "
                  f"success={np.mean(history[-200:]):.3f}", flush=True)

    return history

# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(flat_h, hier_h, out_dir="outputs"):
    window = 100
    flat_sm = np.convolve(flat_h, np.ones(window) / window, mode="valid")
    hier_sm = np.convolve(hier_h, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(flat_sm, label="Flat DQN",      color="tomato")
    axes[0].plot(hier_sm, label="Hierarchical",  color="steelblue")
    axes[0].set_title("Long-Horizon Task: Success Rate (100-ep avg)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(-0.05, 1.05); axes[0].legend()

    f500 = np.mean(flat_h[-500:])
    h500 = np.mean(hier_h[-500:])
    axes[1].bar(["Flat DQN", "Hierarchical"], [f500, h500],
                color=["tomato", "steelblue"])
    axes[1].set_title("Final 500-Episode Success Rate")
    axes[1].set_ylabel("Success Rate"); axes[1].set_ylim(0, 1.05)
    for i, v in enumerate([f500, h500]):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    path = f"{out_dir}/long_horizon_tasks.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}", flush=True)
    print(f"Flat DQN:     {f500:.3f}")
    print(f"Hierarchical: {h500:.3f}")


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)

    print("=== Flat DQN ===", flush=True)
    flat_h = train_flat(n_episodes=3000)
    print("\n=== Hierarchical ===", flush=True)
    hier_h = train_hierarchical(n_episodes=3000)
    plot_results(flat_h, hier_h, out_dir=out_dir)
