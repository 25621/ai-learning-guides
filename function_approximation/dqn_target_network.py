"""
DQN with Target Network for CartPole-v1

Compares:
  (A) DQN + replay (no target network) — targets shift every step, creating a
      "moving goalpost" that destabilizes learning
  (B) Full DQN (replay + target network) — a frozen copy of the network is used
      for computing targets; synced every `target_update` steps

The target network breaks the feedback loop between predictions and targets,
greatly stabilizing training.
"""

import os
import copy
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

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s_, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN with replay buffer; optional frozen target network."""

    def __init__(self, use_target_network=True, target_update=100,
                 lr=1e-3, gamma=0.99, buffer_capacity=10_000,
                 batch_size=64, warmup=500,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997):
        self.n_actions = 2
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.use_target = use_target_network
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        self.qnet = QNetwork().to(DEVICE)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

        if use_target_network:
            self.target_net = copy.deepcopy(self.qnet)
            self.target_net.eval()

    def act(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.qnet(s).argmax(dim=1).item())

    def _get_target_net(self):
        return self.target_net if self.use_target else self.qnet

    def update(self):
        if len(self.buffer) < self.warmup:
            return
        self.steps += 1

        s, a, r, s_, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, device=DEVICE)
        a = torch.tensor(a, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, device=DEVICE).unsqueeze(1)
        s_ = torch.tensor(s_, device=DEVICE)
        d = torch.tensor(d, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            q_next = self._get_target_net()(s_).max(dim=1, keepdim=True).values
            target = r + self.gamma * q_next * (1 - d)

        q_pred = self.qnet(s).gather(1, a)
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        if self.use_target and self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_training(use_target_network, n_episodes=600, seed=42, label=""):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(use_target_network=use_target_network)
    rng = np.random.default_rng(seed)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done, total = False, 0
        while not done:
            action = agent.act(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            total += reward

        agent.decay_epsilon()
        rewards.append(total)

        if label and (ep + 1) % 100 == 0:
            avg = float(np.mean(rewards[-min(50, ep + 1):]))
            print(f"  [{label}] Ep {ep + 1:4d} | avg50 = {avg:6.1f} | ε = {agent.epsilon:.3f}")

    env.close()
    return rewards


def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_comparison(no_target_rewards, with_target_rewards):
    os.makedirs("outputs", exist_ok=True)
    window = 50
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rewards, label, color in [
        (axes[0], no_target_rewards, "DQN + Replay (no target)", 'steelblue'),
        (axes[1], with_target_rewards, "Full DQN (replay + target network)", 'crimson'),
    ]:
        r = np.array(rewards)
        ax.plot(r, alpha=0.2, color=color, linewidth=0.7)
        ax.plot(range(window - 1, len(r)), smooth(r, window), color=color,
                linewidth=2.5, label=f'{window}-ep avg')
        ax.axhline(195, color='green', linestyle='--', linewidth=1.5, label='Solved (195)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 520)

    plt.suptitle("Target Network: Stabilizing DQN on CartPole-v1", fontsize=14, y=1.02)
    plt.tight_layout()

    path = "outputs/dqn_target_network.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {path}")


def main():
    print(f"=== DQN Target Network Comparison on CartPole-v1 ===")
    print(f"Device: {DEVICE}\n")
    n_episodes = 600

    print("(A) Training DQN + Replay (no target network)...")
    no_target = run_training(use_target_network=False, n_episodes=n_episodes,
                             seed=42, label="no-target")
    avg_no = float(np.mean(no_target[-100:]))
    print(f"    Final 100-ep avg: {avg_no:.1f}")

    print("(B) Training Full DQN (replay + target network)...")
    with_target = run_training(use_target_network=True, n_episodes=n_episodes,
                               seed=42, label="with-target")
    avg_with = float(np.mean(with_target[-100:]))
    print(f"    Final 100-ep avg: {avg_with:.1f}")

    print(f"\nImprovement from target network: {avg_with - avg_no:+.1f} reward")
    print(f"Variance: {np.std(no_target[-100:]):.1f} → {np.std(with_target[-100:]):.1f}")

    plot_comparison(no_target, with_target)


if __name__ == "__main__":
    main()
