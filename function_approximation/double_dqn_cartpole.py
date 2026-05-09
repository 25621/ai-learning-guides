"""
Double DQN vs Vanilla DQN on CartPole-v1

Vanilla DQN overestimates Q-values because action selection and value evaluation
use the same network:
    target = r + γ max_a Q_target(s', a)   ← biased upward

Double DQN decouples these by using the online network to SELECT the action and
the target network to EVALUATE it:
    target = r + γ Q_target(s', argmax_a Q_online(s', a))   ← less biased

Compares both variants on CartPole, showing:
  - Learning curves
  - Q-value overestimation (true value vs predicted value)
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
    """DQN or Double DQN depending on `double` flag."""

    def __init__(self, double=False, lr=1e-3, gamma=0.99, target_update=100,
                 buffer_capacity=10_000, batch_size=64, warmup=500,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997):
        self.n_actions = 2
        self.gamma = gamma
        self.double = double
        self.target_update = target_update
        self.batch_size = batch_size
        self.warmup = warmup
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        self.qnet = QNetwork().to(DEVICE)
        self.target_net = copy.deepcopy(self.qnet)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

        # Track Q-value estimates for overestimation analysis
        self.q_estimates = []

    def act(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.qnet(s).argmax(dim=1).item())

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
            if self.double:
                # Double DQN: online selects action, target evaluates
                best_actions = self.qnet(s_).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s_).gather(1, best_actions)
            else:
                # Vanilla DQN: target selects and evaluates (biased)
                q_next = self.target_net(s_).max(dim=1, keepdim=True).values
            target = r + self.gamma * q_next * (1 - d)

        q_pred = self.qnet(s).gather(1, a)
        # Track mean Q estimate for overestimation analysis
        self.q_estimates.append(float(q_pred.mean().item()))

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_training(double, n_episodes=800, seed=42, label=""):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(double=double)
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
    return rewards, agent.q_estimates


def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_comparison(vanilla_rewards, double_rewards, vanilla_q, double_q):
    os.makedirs("outputs", exist_ok=True)
    window = 50

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    for rewards, label, color in [
        (vanilla_rewards, 'Vanilla DQN', 'steelblue'),
        (double_rewards, 'Double DQN', 'crimson'),
    ]:
        r = np.array(rewards)
        ax.plot(r, alpha=0.2, color=color, linewidth=0.7)
        ax.plot(range(window - 1, len(r)), smooth(r, window), color=color,
                linewidth=2.5, label=label)
    ax.axhline(195, color='green', linestyle='--', linewidth=1.5, label='Solved (195)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Double DQN vs Vanilla DQN — Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 520)

    # Q-value estimates (overestimation comparison)
    ax = axes[1]
    for q_est, label, color in [
        (vanilla_q, 'Vanilla DQN Q-estimates', 'steelblue'),
        (double_q, 'Double DQN Q-estimates', 'crimson'),
    ]:
        if len(q_est) > 100:
            q_smooth = np.convolve(q_est, np.ones(100) / 100, mode='valid')
            ax.plot(q_smooth, color=color, linewidth=2, label=label)
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Mean Q-value Estimate")
    ax.set_title("Q-value Overestimation: Vanilla vs Double DQN")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/double_dqn_cartpole.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Comparison plot saved to {path}")


def main():
    print(f"=== Double DQN vs Vanilla DQN on CartPole-v1 ===")
    print(f"Device: {DEVICE}\n")
    n_episodes = 800

    print("(A) Training Vanilla DQN...")
    vanilla_rewards, vanilla_q = run_training(double=False, n_episodes=n_episodes,
                                              seed=42, label="vanilla")
    avg_v = float(np.mean(vanilla_rewards[-100:]))
    print(f"    Final 100-ep avg: {avg_v:.1f}")
    if vanilla_q:
        print(f"    Mean Q-estimate (last 1000 updates): {np.mean(vanilla_q[-1000:]):.3f}")

    print("(B) Training Double DQN...")
    double_rewards, double_q = run_training(double=True, n_episodes=n_episodes,
                                            seed=42, label="double")
    avg_d = float(np.mean(double_rewards[-100:]))
    print(f"    Final 100-ep avg: {avg_d:.1f}")
    if double_q:
        print(f"    Mean Q-estimate (last 1000 updates): {np.mean(double_q[-1000:]):.3f}")

    print(f"\nDouble DQN performance delta: {avg_d - avg_v:+.1f}")
    if vanilla_q and double_q:
        overest_v = float(np.mean(vanilla_q[-1000:]))
        overest_d = float(np.mean(double_q[-1000:]))
        print(f"Q-value overestimation: {overest_v:.3f} (vanilla) vs {overest_d:.3f} (double)")
        if overest_v > overest_d:
            print("✓ Double DQN reduces Q-value overestimation as expected.")

    plot_comparison(vanilla_rewards, double_rewards, vanilla_q, double_q)


if __name__ == "__main__":
    main()
