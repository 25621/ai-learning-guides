"""
DQN with Experience Replay for CartPole-v1

Compares:
  (A) Naive DQN   — online update, one sample at a time (high variance, correlated)
  (B) DQN + replay — random mini-batch from a replay buffer (lower variance, decorrelated)

Experience replay breaks temporal correlations and allows each transition to be
reused multiple times, greatly stabilizing training.
"""

import os
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cpu")


# ─── Shared building blocks ──────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, n_obs=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Agent variants ───────────────────────────────────────────────────────────

class DQNBase:
    """Shared interface for both agent variants."""

    def __init__(self, lr=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.997):
        self.n_actions = 2
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.qnet = QNetwork().to(DEVICE)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.qnet(s).argmax(dim=1).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class NaiveDQNAgent(DQNBase):
    """Online update with a single transition — no replay buffer."""

    def update(self, state, action, reward, next_state, terminated):
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_next = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if terminated:
                target = torch.tensor([[reward]], dtype=torch.float32, device=DEVICE)
            else:
                q_next = self.qnet(s_next).max(dim=1, keepdim=True).values
                target = reward + self.gamma * q_next

        q_pred = self.qnet(s)[:, action].unsqueeze(1)
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class ReplayDQNAgent(DQNBase):
    """Mini-batch updates sampled from a replay buffer."""

    def __init__(self, buffer_capacity=10_000, batch_size=64,
                 warmup_steps=200, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.steps = 0

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1

    def update(self):
        if len(self.buffer) < self.warmup_steps:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s = torch.tensor(states, device=DEVICE)
        a = torch.tensor(actions, device=DEVICE).unsqueeze(1)
        r = torch.tensor(rewards, device=DEVICE).unsqueeze(1)
        s_next = torch.tensor(next_states, device=DEVICE)
        d = torch.tensor(dones, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            q_next = self.qnet(s_next).max(dim=1, keepdim=True).values
            target = r + self.gamma * q_next * (1 - d)

        q_pred = self.qnet(s).gather(1, a)
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()


# ─── Training loops ───────────────────────────────────────────────────────────

def train_naive(n_episodes=600, seed=42):
    env = gym.make("CartPole-v1")
    agent = NaiveDQNAgent()
    rng = np.random.default_rng(seed)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done, total = False, 0

        while not done:
            action = agent.act(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total += reward

        agent.decay_epsilon()
        rewards.append(total)

    env.close()
    return rewards


def train_with_replay(n_episodes=600, seed=42):
    env = gym.make("CartPole-v1")
    agent = ReplayDQNAgent()
    rng = np.random.default_rng(seed)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done, total = False, 0

        while not done:
            action = agent.act(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            total += reward

        agent.decay_epsilon()
        rewards.append(total)

    env.close()
    return rewards


def smooth(rewards, window=50):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


def plot_comparison(naive_rewards, replay_rewards):
    os.makedirs("outputs", exist_ok=True)
    window = 50
    fig, ax = plt.subplots(figsize=(11, 6))

    n = np.array(naive_rewards)
    r = np.array(replay_rewards)
    x_n = range(window - 1, len(n))
    x_r = range(window - 1, len(r))

    ax.plot(n, alpha=0.15, color='steelblue', linewidth=0.7)
    ax.plot(r, alpha=0.15, color='coral', linewidth=0.7)
    ax.plot(x_n, smooth(n, window), color='steelblue', linewidth=2.5,
            label='Naive DQN (no replay)')
    ax.plot(x_r, smooth(r, window), color='crimson', linewidth=2.5,
            label='DQN + Experience Replay')
    ax.axhline(195, color='green', linestyle='--', linewidth=1.5, label='Solved (195)')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Experience Replay: Stability Improvement on CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = "outputs/dqn_experience_replay.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Comparison plot saved to {path}")


def main():
    print(f"=== DQN Experience Replay Comparison on CartPole-v1 ===")
    print(f"Device: {DEVICE}\n")
    n_episodes = 800

    print("(A) Training Naive DQN (no replay buffer)...")
    naive_rewards = train_naive(n_episodes=n_episodes, seed=42)
    avg_naive = float(np.mean(naive_rewards[-100:]))
    print(f"    Final 100-ep avg: {avg_naive:.1f}")

    print("(B) Training DQN with Experience Replay...")
    replay_rewards = train_with_replay(n_episodes=n_episodes, seed=42)
    avg_replay = float(np.mean(replay_rewards[-100:]))
    print(f"    Final 100-ep avg: {avg_replay:.1f}")

    naive_var = float(np.std(naive_rewards[-100:]))
    replay_var = float(np.std(replay_rewards[-100:]))
    print(f"\nStability comparison (last 100 eps std): {naive_var:.1f} → {replay_var:.1f}")
    if replay_var < naive_var:
        print(f"✓ Experience replay reduces variance by {(1 - replay_var/naive_var)*100:.0f}%")
        print(f"  → Training is {naive_var/max(replay_var,1):.1f}× more stable with replay")
    print(f"\nNote: Replay DQN may have lower mean early on, but variance (stability)")
    print(f"is the primary metric — unstable training eventually collapses on harder tasks.")

    plot_comparison(naive_rewards, replay_rewards)


if __name__ == "__main__":
    main()
