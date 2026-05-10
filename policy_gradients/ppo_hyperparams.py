"""
PPO Hyperparameter Sensitivity Analysis on CartPole-v1

Varies one hyperparameter at a time (all others fixed at default):
  1. Clip epsilon ε   : [0.05, 0.2, 0.4]
  2. Learning rate    : [1e-4, 3e-4, 1e-3]
  3. Update epochs    : [3, 10, 20]

Each configuration is run over 3 seeds; results are averaged.
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

    def get_action(self, state):
        logits, value = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, states, actions):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = []
    last_adv = 0.0
    values_ext = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
        advantages.insert(0, last_adv)
    return torch.tensor(advantages, dtype=torch.float32, device=DEVICE)


def run_ppo(seed=42, n_steps=256, n_epochs=4, batch_size=64, gamma=0.99,
            gae_lambda=0.95, clip_eps=0.2, lr=3e-4, entropy_coef=0.01,
            value_coef=0.5, max_grad_norm=0.5, n_updates=100):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    net = ActorCritic().to(DEVICE)
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

            next_state, reward, terminated, truncated, _ = env.step(action.item())
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
            _, next_value = net(state_t)
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

    env.close()
    return episode_rewards


def smooth(rewards, window=20):
    if len(rewards) < window:
        return np.array(rewards)
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


def run_experiment(seeds, n_updates, param_name, param_values, run_fn):
    results = {}
    for val in param_values:
        all_rewards = []
        for seed in seeds:
            r = run_fn(seed=seed, val=val, n_updates=n_updates)
            all_rewards.append(r)
        min_len = min(len(r) for r in all_rewards)
        trimmed = [r[:min_len] for r in all_rewards]
        mean_r = np.mean(trimmed, axis=0)
        results[val] = mean_r
        final = np.mean(mean_r[-min(50, len(mean_r)):])
        print(f"  {param_name}={val}: final avg = {final:.1f}")
    return results


def main():
    seeds = [42, 123]
    n_updates = 80

    print("Experiment 1: Clip epsilon")
    clip_results = run_experiment(
        seeds, n_updates, "clip_eps", [0.05, 0.2, 0.4],
        lambda seed, val, n_updates: run_ppo(seed=seed, clip_eps=val, n_updates=n_updates)
    )

    print("Experiment 2: Learning rate")
    lr_results = run_experiment(
        seeds, n_updates, "lr", [1e-4, 3e-4, 1e-3],
        lambda seed, val, n_updates: run_ppo(seed=seed, lr=val, n_updates=n_updates)
    )

    print("Experiment 3: Update epochs")
    epoch_results = run_experiment(
        seeds, n_updates, "n_epochs", [3, 10, 20],
        lambda seed, val, n_updates: run_ppo(seed=seed, n_epochs=val, n_updates=n_updates)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PPO Hyperparameter Sensitivity (CartPole-v1, 3-seed avg)", fontsize=13)

    for ax, results, title, fmt in [
        (axes[0], clip_results, "Clip Epsilon (ε)", "ε={}"),
        (axes[1], lr_results,   "Learning Rate",     "lr={}"),
        (axes[2], epoch_results,"Update Epochs",     "epochs={}"),
    ]:
        for val, mean_r in results.items():
            sm = smooth(mean_r, 20)
            ax.plot(sm, label=fmt.format(val))
        ax.axhline(y=475, color='gray', linestyle='--', alpha=0.5, label="Solved")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (20-ep avg)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppo_hyperparams.png"), dpi=100)
    print(f"\nPlot saved to outputs/ppo_hyperparams.png")


if __name__ == "__main__":
    main()
