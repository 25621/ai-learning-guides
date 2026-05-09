"""
DQN from Scratch for CartPole-v1 (baseline: no replay buffer, no target network)

A neural network replaces the Q-table, approximating Q(s, a) for all actions.
This "naive" version updates online (one experience at a time) to show instability
before improvements are added in subsequent scripts.

Network: state(4) → FC(128) → ReLU → FC(128) → ReLU → Q-values(2)
"""

import os
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
            nn.Linear(n_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class NaiveDQNAgent:
    """DQN without replay buffer or target network — naive online updates."""

    def __init__(self, n_obs=4, n_actions=2, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.qnet = QNetwork(n_obs, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.qnet(s)
        return int(q.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, terminated):
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_next = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            if terminated:
                target = torch.tensor([[reward]], dtype=torch.float32, device=DEVICE)
            else:
                q_next = self.qnet(s_next).max(dim=1, keepdim=True).values
                target = torch.tensor([[reward]], dtype=torch.float32, device=DEVICE) \
                         + self.gamma * q_next

        q_pred = self.qnet(s)[:, action].unsqueeze(1)
        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(n_episodes=600, seed=42):
    env = gym.make("CartPole-v1")
    agent = NaiveDQNAgent()
    rng = np.random.default_rng(seed)
    episode_rewards = []
    episode_losses = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        total_reward = 0
        step_losses = []

        while not done:
            action = agent.act(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            loss = agent.update(obs, action, reward, next_obs, terminated)
            step_losses.append(loss)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(step_losses)))

        if (ep + 1) % 100 == 0:
            window = min(50, ep + 1)
            avg = np.mean(episode_rewards[-window:])
            print(f"  Episode {ep + 1:4d} | avg({window}) = {avg:6.1f} | ε = {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards, episode_losses


def evaluate(agent, n_episodes=200, seed=99):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = int(agent.qnet(s).argmax(dim=1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def plot_results(episode_rewards, episode_losses):
    os.makedirs("outputs", exist_ok=True)
    window = 50
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    rewards = np.array(episode_rewards)
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    axes[0].plot(rewards, alpha=0.25, color='steelblue', linewidth=0.8)
    axes[0].plot(range(window - 1, len(rewards)), rolling, color='navy',
                 linewidth=2, label=f'{window}-ep moving avg')
    axes[0].axhline(195, color='red', linestyle='--', linewidth=1.5, label='Solved (195)')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Naive DQN (no replay, no target network) on CartPole-v1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    losses = np.array(episode_losses)
    rolling_loss = np.convolve(losses, np.ones(window) / window, mode='valid')
    axes[1].plot(losses, alpha=0.25, color='orange', linewidth=0.8)
    axes[1].plot(range(window - 1, len(losses)), rolling_loss, color='darkorange',
                 linewidth=2, label=f'{window}-ep moving avg')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Loss")
    axes[1].set_title("Training Loss (note instability — this is expected without improvements)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/dqn_cartpole_naive.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Plot saved to {path}")


def main():
    print(f"=== Naive DQN (no replay, no target network) on CartPole-v1 ===")
    print(f"Device: {DEVICE}\n")
    agent, rewards, losses = train(n_episodes=600)

    window = 100
    final_avg = float(np.mean(rewards[-window:]))
    print(f"\nFinal {window}-episode avg reward (training): {final_avg:.1f}")

    mean_eval, std_eval = evaluate(agent)
    print(f"Greedy evaluation (200 eps): {mean_eval:.1f} ± {std_eval:.1f}")
    print(f"\nNote: Instability is expected without experience replay and target network.")
    print(f"See dqn_experience_replay.py and dqn_target_network.py for improvements.")

    plot_results(rewards, losses)


if __name__ == "__main__":
    main()
