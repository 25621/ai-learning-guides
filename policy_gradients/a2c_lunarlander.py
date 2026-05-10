"""
A2C (Advantage Actor-Critic) for LunarLander-v3

Synchronous n-step actor-critic with multiple parallel environments (SyncVectorEnv).
Uses GAE (Generalized Advantage Estimation) for stable advantage computation.

Rollout collected without gradients; update re-evaluates for clean computation graph.
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


class ActorCritic(nn.Module):
    def __init__(self, n_obs=8, n_actions=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, state_t):
        with torch.no_grad():
            logits, value = self(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.numpy(), value.numpy()

    def evaluate(self, states, actions):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def compute_gae_vectorized(rewards, values, dones, next_values,
                            gamma=0.99, gae_lambda=0.95):
    """GAE over (n_steps, n_envs) arrays."""
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv = np.zeros(n_envs)
    values_ext = np.vstack([values, next_values[np.newaxis, :]])

    for t in reversed(range(n_steps)):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def train_a2c(
    n_envs=8,
    n_steps=128,
    n_updates=3000,
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
):
    envs = gym.vector.SyncVectorEnv([
        (lambda i=i: gym.make("LunarLander-v3")) for i in range(n_envs)
    ])
    envs.reset(seed=SEED)

    net = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    episode_rewards = []
    running_reward = np.zeros(n_envs)
    states, _ = envs.reset()

    for update in range(n_updates):
        buf_states = np.zeros((n_steps, n_envs, 8), dtype=np.float32)
        buf_actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        buf_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_values = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            state_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions, values = net.act(state_t)

            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            dones = (terminated | truncated).astype(np.float32)

            buf_states[step] = states
            buf_actions[step] = actions
            buf_rewards[step] = rewards
            buf_dones[step] = dones
            buf_values[step] = values

            running_reward += rewards
            for env_idx in range(n_envs):
                if dones[env_idx]:
                    episode_rewards.append(running_reward[env_idx])
                    running_reward[env_idx] = 0.0

            states = next_states

        next_state_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, next_values = net(next_state_t)
        next_values = next_values.numpy()

        advantages_np, returns_np = compute_gae_vectorized(
            buf_rewards, buf_values, buf_dones, next_values, gamma, gae_lambda
        )

        # Flatten
        flat_states = torch.tensor(buf_states.reshape(-1, 8), dtype=torch.float32, device=DEVICE)
        flat_actions = torch.tensor(buf_actions.reshape(-1), dtype=torch.long, device=DEVICE)
        flat_returns = torch.tensor(returns_np.reshape(-1), dtype=torch.float32, device=DEVICE)
        flat_adv = torch.tensor(advantages_np.reshape(-1), dtype=torch.float32, device=DEVICE)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        log_probs, entropy, values_new = net.evaluate(flat_states, flat_actions)

        actor_loss = -(log_probs * flat_adv).mean()
        critic_loss = value_coef * nn.functional.mse_loss(values_new, flat_returns)
        entropy_loss = -entropy_coef * entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        if (update + 1) % 300 == 0:
            n_recent = min(100, len(episode_rewards))
            avg = np.mean(episode_rewards[-n_recent:]) if episode_rewards else 0.0
            print(f"Update {update+1:4d} | Episodes: {len(episode_rewards):5d} | "
                  f"Avg reward (last {n_recent}): {avg:.1f}")

    envs.close()
    return episode_rewards


def main():
    print("Training A2C on LunarLander-v3 (8 parallel envs, GAE)...")
    rewards = train_a2c(n_updates=3000)

    window = 100
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    else:
        smoothed = np.array(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= window:
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep Moving Avg")
    plt.axhline(y=200, color='g', linestyle='--', label="Solved (200)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A2C on LunarLander-v3 (8 envs, GAE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "a2c_lunarlander.png"), dpi=100)
    print(f"Plot saved to outputs/a2c_lunarlander.png")
    n_recent = min(100, len(rewards))
    print(f"Final {n_recent}-ep avg: {np.mean(rewards[-n_recent:]):.1f}")


if __name__ == "__main__":
    main()
