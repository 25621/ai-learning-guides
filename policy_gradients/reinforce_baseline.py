"""
REINFORCE with Baseline for CartPole-v1

Subtracting a learned value baseline V(s) from returns reduces gradient variance
without introducing bias:

∇J(θ) = E[∇log π(a_t|s_t) · (G_t - V(s_t))]

Trains two agents (with/without baseline) side-by-side and measures variance reduction.
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


class ValueNetwork(nn.Module):
    def __init__(self, n_obs=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE)


def train(use_baseline=True, n_episodes=1000, gamma=0.99, lr_policy=1e-3, lr_value=1e-3):
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)

    policy = PolicyNetwork().to(DEVICE)
    optimizer_policy = optim.Adam(policy.parameters(), lr=lr_policy)

    value_net = None
    optimizer_value = None
    if use_baseline:
        value_net = ValueNetwork().to(DEVICE)
        optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value)

    episode_rewards = []
    gradient_variances = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        log_probs, rewards, states = [], [], []

        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            states.append(state_t)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        episode_rewards.append(sum(rewards))

        returns = compute_returns(rewards, gamma)
        states_t = torch.stack(states)

        if use_baseline:
            with torch.no_grad():
                baseline = value_net(states_t)
            advantages = returns - baseline

            value_preds = value_net(states_t)
            value_loss = nn.functional.mse_loss(value_preds, returns)
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()
        else:
            advantages = returns

        gradient_variances.append(advantages.var().item())

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(torch.stack(log_probs) * advantages).sum()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            avg_var = np.mean(gradient_variances[-100:])
            label = "w/ baseline" if use_baseline else "no baseline"
            print(f"Episode {ep+1:4d} [{label}] | Avg reward: {avg:.1f} | Grad var: {avg_var:.1f}")

    env.close()
    return episode_rewards, gradient_variances


def main():
    print("Training REINFORCE without baseline...")
    rewards_no_base, vars_no_base = train(use_baseline=False, n_episodes=1000)

    print("\nTraining REINFORCE with baseline...")
    rewards_base, vars_base = train(use_baseline=True, n_episodes=1000)

    window = 50
    smooth = lambda r: np.convolve(r, np.ones(window) / window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(smooth(rewards_no_base), label="No Baseline", color='orange')
    ax.plot(smooth(rewards_base), label="With Baseline", color='blue')
    ax.axhline(y=475, color='g', linestyle='--', label="Solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-ep avg)")
    ax.set_title("REINFORCE: Reward Comparison")
    ax.legend()

    ax = axes[1]
    ax.plot(smooth(vars_no_base), label="No Baseline", color='orange')
    ax.plot(smooth(vars_base), label="With Baseline", color='blue')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gradient Signal Variance")
    ax.set_title("Variance Reduction with Baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reinforce_baseline.png"), dpi=100)
    print(f"\nPlot saved to outputs/reinforce_baseline.png")
    print(f"No baseline  — Final 100-ep avg: {np.mean(rewards_no_base[-100:]):.1f}, "
          f"grad var: {np.mean(vars_no_base[-100:]):.1f}")
    print(f"With baseline — Final 100-ep avg: {np.mean(rewards_base[-100:]):.1f}, "
          f"grad var: {np.mean(vars_base[-100:]):.1f}")


if __name__ == "__main__":
    main()
