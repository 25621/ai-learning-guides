"""
Option-Critic Architecture on a simple 7x7 GridWorld.

The agent learns N options (sub-policies), each with its own termination
condition.  A meta-policy (policy-over-options) selects which option to
execute.  The option runs until it chooses to terminate, then the
meta-policy picks again.

Environment: 7x7 grid, random start, fixed goal at (5,5).
Potential-based shaping is added to make the reward less sparse.
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
        idx = random.sample(range(self.size), k)
        return zip(*[self.buf[i] for i in idx])

    def __len__(self):
        return self.size

# ─── Environment ─────────────────────────────────────────────────────────────

GRID = 7
GOAL_POS = (5, 5)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)
WALLS = set()
for i in range(GRID):
    WALLS.add((0, i)); WALLS.add((GRID - 1, i))
    WALLS.add((i, 0)); WALLS.add((i, GRID - 1))
FREE = [(r, c) for r in range(GRID) for c in range(GRID)
        if (r, c) not in WALLS and (r, c) != GOAL_POS]
N_STATES = GRID * GRID


def enc(r, c):
    return r * GRID + c


def potential(r, c):
    return -abs(r - GOAL_POS[0]) - abs(c - GOAL_POS[1])


class GridWorld:
    def reset(self):
        self.pos = random.choice(FREE)
        return enc(*self.pos)

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        old_phi = potential(*self.pos)
        if (nr, nc) not in WALLS and 0 <= nr < GRID and 0 <= nc < GRID:
            self.pos = (nr, nc)
        done = (self.pos == GOAL_POS)
        new_phi = potential(*self.pos)
        shaped_r = (1.0 if done else 0.0) + 0.99 * new_phi - old_phi
        return enc(*self.pos), shaped_r, done

# ─── Networks ─────────────────────────────────────────────────────────────────

N_OPTIONS = 4


class OptionCriticNet(nn.Module):
    def __init__(self, n_states, n_actions, n_options, hidden=64):
        super().__init__()
        self.n_options = n_options
        self.n_actions = n_actions
        self.emb = nn.Embedding(n_states, hidden)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.q_head = nn.Linear(hidden, n_options)
        self.pi_head = nn.Linear(hidden, n_options * n_actions)
        self.term_head = nn.Linear(hidden, n_options)

    def body(self, s):
        return self.fc(self.emb(s))

    def q(self, h):
        return self.q_head(h)

    def intra_pi(self, h):
        B = h.shape[0]
        return self.pi_head(h).view(B, self.n_options, self.n_actions).softmax(-1)

    def termination(self, h):
        return self.term_head(h).sigmoid()

# ─── Agent ────────────────────────────────────────────────────────────────────

class OptionCriticAgent:
    def __init__(self, n_states, n_actions, n_options=N_OPTIONS,
                 lr=3e-4, gamma=0.99, buffer_size=8_000, batch=64):
        self.gamma = gamma
        self.n_options = n_options
        self.batch = batch
        self.net    = OptionCriticNet(n_states, n_actions, n_options)
        self.target = OptionCriticNet(n_states, n_actions, n_options)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer_size)

    def _t(self, s):
        return torch.tensor([s], dtype=torch.long)

    def select_option(self, s, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_options)
        with torch.no_grad():
            h = self.net.body(self._t(s))
            return self.net.q(h).argmax().item()

    def should_terminate(self, s, option):
        with torch.no_grad():
            h = self.net.body(self._t(s))
            beta = self.net.termination(h)[0, option].item()
        return random.random() < beta

    def select_action(self, s, option, epsilon):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            h = self.net.body(self._t(s))
            probs = self.net.intra_pi(h)[0, option]
        return torch.multinomial(probs, 1).item()

    def update(self):
        if len(self.buf) < self.batch:
            return
        s, o, a, r, ns, done = self.buf.sample(self.batch)
        s  = torch.tensor(list(s),    dtype=torch.long)
        o  = torch.tensor(list(o),    dtype=torch.long)
        a  = torch.tensor(list(a),    dtype=torch.long)
        r  = torch.tensor(list(r),    dtype=torch.float)
        ns = torch.tensor(list(ns),   dtype=torch.long)
        dn = torch.tensor(list(done), dtype=torch.float)
        B  = self.batch

        with torch.no_grad():
            nh     = self.target.body(ns)
            q_next = self.target.q(nh).max(1).values
            tgt    = r + self.gamma * q_next * (1 - dn)

        h = self.net.body(s)

        # critic
        q_pred = self.net.q(h).gather(1, o.unsqueeze(1)).squeeze()
        loss_q = nn.functional.mse_loss(q_pred, tgt)

        # intra-option policy
        pi_all = self.net.intra_pi(h)
        pi_o   = pi_all[torch.arange(B), o]
        log_pi = torch.log(pi_o.gather(1, a.unsqueeze(1)).squeeze() + 1e-8)
        adv    = (tgt - q_pred).detach()
        loss_pi = -(log_pi * adv).mean()

        # termination
        nh2   = self.net.body(ns)
        beta  = self.net.termination(nh2)[torch.arange(B), o]
        q_ns  = self.net.q(nh2)
        adv_t = (q_ns[torch.arange(B), o] - q_ns.max(1).values).detach()
        loss_term = (beta * adv_t).mean()

        loss = loss_q + loss_pi + 0.01 * loss_term
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

    def sync_target(self):
        self.target.load_state_dict(self.net.state_dict())

# ─── Training ─────────────────────────────────────────────────────────────────

def train(n_episodes=2500, max_steps=100, target_update=150):
    env   = GridWorld()
    agent = OptionCriticAgent(N_STATES, N_ACTIONS)
    returns, lengths = [], []

    for ep in range(n_episodes):
        epsilon = max(0.05, 0.5 * (1 - ep / n_episodes))
        s = env.reset()
        option = agent.select_option(s, epsilon)
        ep_ret, ep_len, done = 0.0, 0, False

        while not done and ep_len < max_steps:
            a = agent.select_action(s, option, epsilon)
            ns, r, done = env.step(a)
            agent.buf.add(s, option, a, r, ns, done)
            agent.update()
            ep_ret += r; ep_len += 1; s = ns
            if not done and agent.should_terminate(s, option):
                option = agent.select_option(s, epsilon)

        if (ep + 1) % target_update == 0:
            agent.sync_target()

        returns.append(ep_ret)
        lengths.append(ep_len)

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1:4d} | "
                  f"avg_return={np.mean(returns[-200:]):.3f} | "
                  f"avg_steps={np.mean(lengths[-200:]):.1f}", flush=True)

    return returns, lengths

# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(returns, lengths, out_dir="outputs"):
    window = 100
    r_sm = np.convolve(returns, np.ones(window) / window, mode="valid")
    l_sm = np.convolve(lengths, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(r_sm, color="steelblue")
    axes[0].set_title("Option-Critic: Shaped Return (100-ep avg)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Shaped Return")

    axes[1].plot(l_sm, color="darkorange")
    axes[1].set_title("Option-Critic: Steps to Goal (100-ep avg)")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Steps")
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5, label="random walk baseline")
    axes[1].legend()

    plt.tight_layout()
    path = f"{out_dir}/option_critic.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    returns, lengths = train(n_episodes=2500)
    plot_results(returns, lengths, out_dir=out_dir)
    print(f"Done. Final 200-ep avg return: {np.mean(returns[-200:]):.3f}", flush=True)
