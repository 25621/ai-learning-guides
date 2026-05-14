"""
Goal-Conditioned Policy with Hindsight Experience Replay (HER).

The agent learns a universal policy π(a | s, g) that can reach any goal
cell in a 7x7 maze.  HER replays failed trajectories as if the agent
had been aiming for the state it actually reached — dramatically
accelerating learning in sparse-reward settings.
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ─── Environment ────────────────────────────────────────────────────────────

MAZE = 7
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)


def _make_walls(size):
    walls = set()
    for i in range(size):
        walls.add((0, i)); walls.add((size - 1, i))
        walls.add((i, 0)); walls.add((i, size - 1))
    return walls


WALLS = _make_walls(MAZE)
FREE = [(r, c) for r in range(MAZE) for c in range(MAZE) if (r, c) not in WALLS]


def _encode(r, c, size=MAZE):
    return r * size + c


N_STATES = MAZE * MAZE


class GoalMaze:
    """Sparse-reward maze: +1 only when reaching the goal, 0 otherwise."""

    def __init__(self):
        self.pos = None
        self.goal = None
        self.goal_enc = None

    def reset(self, goal=None):
        self.goal = goal if goal is not None else random.choice(FREE)
        self.pos = random.choice([c for c in FREE if c != self.goal])
        self.goal_enc = _encode(*self.goal)
        return _encode(*self.pos), self.goal_enc

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if (nr, nc) not in WALLS and 0 <= nr < MAZE and 0 <= nc < MAZE:
            self.pos = (nr, nc)
        done = (self.pos == self.goal)
        reward = 1.0 if done else 0.0
        return _encode(*self.pos), self.goal_enc, reward, done


# ─── Network ────────────────────────────────────────────────────────────────

class GoalConditionedNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.state_emb = nn.Embedding(n_states, 32)
        self.goal_emb = nn.Embedding(n_states, 32)
        self.net = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state, goal):
        h = torch.cat([self.state_emb(state), self.goal_emb(goal)], dim=-1)
        return self.net(h)


# ─── HER Replay Buffer ───────────────────────────────────────────────────────

class HERBuffer:
    """Flat replay buffer with hindsight relabeling at sample time."""

    def __init__(self, capacity=50_000, her_k=4):
        self.transitions = deque(maxlen=capacity)
        self.her_k = her_k

    def store_episode(self, trajectory):
        """trajectory: list of (s_enc, g_enc, a, ns_enc, done) tuples."""
        T = len(trajectory)
        for t, (s, g, a, ns, done) in enumerate(trajectory):
            # real transition
            self.transitions.append((s, g, a, ns, done))
            # HER: relabel with k future achieved states
            for _ in range(self.her_k):
                future_t = random.randint(t, T - 1)
                her_goal = trajectory[future_t][3]  # ns_enc at future step
                her_done = (ns == her_goal)
                her_r = 1.0 if her_done else 0.0
                self.transitions.append((s, her_goal, a, ns, her_done))

    def sample(self, batch_size):
        batch = random.sample(self.transitions, min(batch_size, len(self.transitions)))
        s, g, a, r_or_done, ns, done = [], [], [], [], [], []
        for item in batch:
            _s, _g, _a, _ns, _done = item
            s.append(_s); g.append(_g); a.append(_a)
            r_or_done.append(1.0 if _done else 0.0)
            ns.append(_ns); done.append(float(_done))
        return (
            torch.tensor(s, dtype=torch.long),
            torch.tensor(g, dtype=torch.long),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r_or_done, dtype=torch.float),
            torch.tensor(ns, dtype=torch.long),
            torch.tensor(done, dtype=torch.float),
        )

    def __len__(self):
        return len(self.transitions)


# ─── Agent ──────────────────────────────────────────────────────────────────

class GCPAgent:
    def __init__(self, n_states, n_actions, lr=5e-4, gamma=0.98):
        self.gamma = gamma
        self.n_actions = n_actions
        self.net = GoalConditionedNet(n_states, n_actions)
        self.target = GoalConditionedNet(n_states, n_actions)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.buf = HERBuffer()

    def act(self, state, goal, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        s = torch.tensor([state], dtype=torch.long)
        g = torch.tensor([goal], dtype=torch.long)
        with torch.no_grad():
            q = self.net(s, g)
        return q.argmax().item()

    def update(self, batch_size=256):
        if len(self.buf) < batch_size:
            return
        s, g, a, r, ns, done = self.buf.sample(batch_size)
        with torch.no_grad():
            q_next = self.target(ns, g).max(dim=1).values
            target_q = r + self.gamma * q_next * (1 - done)
        q_pred = self.net(s, g).gather(1, a.unsqueeze(1)).squeeze()
        loss = nn.functional.mse_loss(q_pred, target_q)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def sync_target(self):
        self.target.load_state_dict(self.net.state_dict())


# ─── Training ───────────────────────────────────────────────────────────────

def train(n_episodes=3000, max_steps=40, target_update=100):
    env = GoalMaze()
    agent = GCPAgent(N_STATES, N_ACTIONS)
    success_history = []

    for ep in range(n_episodes):
        epsilon = max(0.05, 0.5 * (1 - ep / n_episodes))
        (s, g) = env.reset()
        trajectory = []
        done = False
        step = 0
        success = False

        while not done and step < max_steps:
            a = agent.act(s, g, epsilon=epsilon)
            ns, _g, r, done = env.step(a)
            trajectory.append((s, g, a, ns, done))
            s = ns
            step += 1
            if done:
                success = True

        agent.buf.store_episode(trajectory)
        for _ in range(2):
            agent.update(batch_size=256)

        if (ep + 1) % target_update == 0:
            agent.sync_target()

        success_history.append(float(success))

        if (ep + 1) % 500 == 0:
            rate = np.mean(success_history[-200:])
            print(f"Episode {ep+1:4d} | epsilon={epsilon:.3f} | success_rate={rate:.3f}")

    return agent, success_history


def evaluate_heatmap(agent, n_eval=30):
    env = GoalMaze()
    grid = np.full((MAZE, MAZE), np.nan)
    for r in range(1, MAZE - 1):
        for c in range(1, MAZE - 1):
            if (r, c) in WALLS:
                continue
            g_pos = (r, c)
            successes = 0
            for _ in range(n_eval):
                (s, g) = env.reset(goal=g_pos)
                done = False
                step = 0
                while not done and step < 40:
                    a = agent.act(s, g, epsilon=0.0)
                    ns, _g, r_val, done = env.step(a)
                    s = ns
                    step += 1
                if done:
                    successes += 1
            grid[r, c] = successes / n_eval
    return grid


def plot_results(agent, success_history, out_dir="outputs"):
    window = 100
    smoothed = np.convolve(success_history, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(smoothed, color="mediumseagreen")
    axes[0].set_title("Goal-Conditioned Policy: Success Rate (with HER)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% line")
    axes[0].legend()

    grid = evaluate_heatmap(agent)
    im = axes[1].imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", origin="upper")
    axes[1].set_title("Goal Success Rate Heatmap\n(can agent reach each cell?)")
    axes[1].set_xlabel("Column"); axes[1].set_ylabel("Row")
    plt.colorbar(im, ax=axes[1], label="Success Rate")

    plt.tight_layout()
    path = f"{out_dir}/goal_conditioned_policy.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    agent, success_history = train(n_episodes=3000)
    plot_results(agent, success_history, out_dir=out_dir)
    print(f"Done. Final 200-ep success rate: {np.mean(success_history[-200:]):.3f}")
