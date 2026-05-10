"""
PPO for Continuous Control: BipedalWalker-v3

Extends PPO to continuous action spaces using a Gaussian policy.
Actor outputs mean and log_std; actions are sampled from Normal distribution.

Key difference from discrete PPO:
- log_prob = sum of per-dimension Normal log_probs
- Actions clipped to environment bounds after sampling
- Shared actor_log_std as a learned parameter (not state-dependent)
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


class GaussianActorCritic(nn.Module):
    def __init__(self, n_obs=24, n_actions=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, n_actions)
        self.actor_log_std = nn.Parameter(torch.zeros(n_actions))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, state):
        mean, std, value = self(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action.clamp(-1, 1), log_prob, entropy, value

    def evaluate(self, states, actions):
        mean, std, values = self(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = []
    last_adv = 0.0
    values_ext = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
        advantages.insert(0, last_adv)
    return torch.tensor(advantages, dtype=torch.float32, device=DEVICE)


def train_ppo_continuous(
    n_steps=2048,
    n_epochs=10,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=3e-4,
    entropy_coef=0.0,
    value_coef=0.5,
    max_grad_norm=0.5,
    n_updates=500,
):
    env = gym.make("BipedalWalker-v3")
    env.reset(seed=SEED)

    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    net = GaussianActorCritic(n_obs=n_obs, n_actions=n_actions).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    episode_rewards = []
    state, _ = env.reset()
    ep_reward = 0.0

    for update in range(n_updates):
        states, actions, log_probs_old, rewards, values, dones = [], [], [], [], [], []

        for _ in range(n_steps):
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action, log_prob, _, value = net.get_action(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            states.append(state_t)
            actions.append(action)
            log_probs_old.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(float(done))

            ep_reward += reward
            state = next_state

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            _, _, next_value = net(state_t)
            next_value = next_value.item()

        advantages = compute_gae(rewards, values, dones, next_value, gamma, gae_lambda)
        returns = advantages + torch.tensor(values, device=DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        log_probs_old_t = torch.stack(log_probs_old).detach()

        indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, batch_size):
                idx = indices[start:start + batch_size]
                log_probs_new, entropy, values_new = net.evaluate(states_t[idx], actions_t[idx])

                ratio = torch.exp(log_probs_new - log_probs_old_t[idx])
                adv_batch = advantages[idx]
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = value_coef * nn.functional.mse_loss(values_new, returns[idx])
                entropy_loss = -entropy_coef * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        if (update + 1) % 50 == 0:
            n_recent = min(20, len(episode_rewards))
            avg = np.mean(episode_rewards[-n_recent:]) if episode_rewards else 0
            print(f"Update {update+1:4d} | Episodes: {len(episode_rewards):4d} | Avg reward: {avg:.1f}")

    env.close()
    return episode_rewards


def main():
    print("Training PPO on BipedalWalker-v3...")
    rewards = train_ppo_continuous(n_updates=500)

    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid') if len(rewards) >= window else rewards

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= window:
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep Moving Avg")
    plt.axhline(y=300, color='g', linestyle='--', label="Target (300)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on BipedalWalker-v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppo_continuous.png"), dpi=100)
    print(f"Plot saved to outputs/ppo_continuous.png")
    n_recent = min(20, len(rewards))
    final_avg = np.mean(rewards[-n_recent:]) if rewards else 0
    print(f"Final {n_recent}-ep avg: {final_avg:.1f}")


if __name__ == "__main__":
    main()
