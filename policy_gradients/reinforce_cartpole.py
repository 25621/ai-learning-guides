"""
REINFORCE for CartPole-v1

Monte Carlo policy gradient: collect full episodes, compute discounted returns,
then update the policy to increase probability of actions that led to high returns.

∇J(θ) = E[∑_t ∇log π(a_t|s_t) · G_t]
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
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PolicyNetwork(nn.Module):
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
        return torch.softmax(self.net(x), dim=-1)


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE)


def train_reinforce(n_episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)

    policy = PolicyNetwork().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        episode_rewards.append(sum(rewards))

        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1:4d} | Avg reward (last 100): {avg:.1f}")

    env.close()
    return episode_rewards


def main():
    print("Training REINFORCE on CartPole-v1...")
    rewards = train_reinforce(n_episodes=1000)

    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep Moving Avg")
    plt.axhline(y=475, color='g', linestyle='--', label="Solved (475)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reinforce_cartpole.png"), dpi=100)
    print(f"Plot saved to outputs/reinforce_cartpole.png")
    print(f"Final 100-ep avg: {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    main()
