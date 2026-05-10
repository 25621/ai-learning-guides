"""
PPO (Proximal Policy Optimization) from Scratch for CartPole-v1

Uses clipped surrogate objective and multiple parallel environments for stability:
  L_CLIP = E[min(r_t(θ) A_t,  clip(r_t(θ), 1-ε, 1+ε) A_t)]
where r_t(θ) = π_new(a|s) / π_old(a|s)

Multiple envs provide decorrelated transitions — the single biggest stability
improvement over naive policy gradient implementations.
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
    def __init__(self, n_obs=4, n_actions=2, hidden=64):
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
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob.numpy(), value.numpy()

    def evaluate(self, states, actions):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def compute_gae(rewards, values, dones, next_values, gamma=0.99, gae_lambda=0.95):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv = np.zeros(n_envs)
    values_ext = np.vstack([values, next_values[np.newaxis, :]])
    for t in reversed(range(n_steps)):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values


def train_ppo(
    n_envs=4,
    n_steps=128,
    n_epochs=4,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=2.5e-4,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    n_updates=500,
):
    envs = gym.vector.SyncVectorEnv([
        (lambda i=i: gym.make("CartPole-v1")) for i in range(n_envs)
    ])
    envs.reset(seed=SEED)

    net = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    episode_rewards = []
    running_reward = np.zeros(n_envs)
    states, _ = envs.reset()

    for update in range(n_updates):
        buf_states = np.zeros((n_steps, n_envs, 4), dtype=np.float32)
        buf_actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        buf_log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        buf_values = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            state_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions, log_probs, values = net.act(state_t)

            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            dones = (terminated | truncated).astype(np.float32)

            buf_states[step] = states
            buf_actions[step] = actions
            buf_log_probs[step] = log_probs
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

        advantages_np, returns_np = compute_gae(
            buf_rewards, buf_values, buf_dones, next_values, gamma, gae_lambda
        )

        flat_states = torch.tensor(buf_states.reshape(-1, 4), dtype=torch.float32, device=DEVICE)
        flat_actions = torch.tensor(buf_actions.reshape(-1), dtype=torch.long, device=DEVICE)
        flat_log_probs_old = torch.tensor(buf_log_probs.reshape(-1), dtype=torch.float32, device=DEVICE)
        flat_returns = torch.tensor(returns_np.reshape(-1), dtype=torch.float32, device=DEVICE)
        flat_adv = torch.tensor(advantages_np.reshape(-1), dtype=torch.float32, device=DEVICE)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        dataset_size = n_steps * n_envs
        indices = np.arange(dataset_size)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                idx = indices[start:start + batch_size]

                log_probs_new, entropy, values_new = net.evaluate(
                    flat_states[idx], flat_actions[idx]
                )
                ratio = torch.exp(log_probs_new - flat_log_probs_old[idx])
                adv_batch = flat_adv[idx]

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = value_coef * nn.functional.mse_loss(values_new, flat_returns[idx])
                entropy_loss = -entropy_coef * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        if (update + 1) % 50 == 0:
            n_recent = min(50, len(episode_rewards))
            avg = np.mean(episode_rewards[-n_recent:]) if episode_rewards else 0
            print(f"Update {update+1:4d} | Episodes: {len(episode_rewards):5d} | "
                  f"Avg reward (last {n_recent}): {avg:.1f}")

    envs.close()
    return episode_rewards


def main():
    print("Training PPO from scratch on CartPole-v1 (4 parallel envs)...")
    rewards = train_ppo(n_envs=4, n_steps=256, n_epochs=4, lr=1e-4, clip_eps=0.1, n_updates=1000)

    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    else:
        smoothed = np.array(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= window:
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep Moving Avg")
    plt.axhline(y=475, color='g', linestyle='--', label="Solved (475)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO from Scratch on CartPole-v1 (4 parallel envs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppo_scratch.png"), dpi=100)
    print(f"Plot saved to outputs/ppo_scratch.png")
    n_recent = min(50, len(rewards))
    print(f"Final {n_recent}-ep avg: {np.mean(rewards[-n_recent:]):.1f}")


if __name__ == "__main__":
    main()
