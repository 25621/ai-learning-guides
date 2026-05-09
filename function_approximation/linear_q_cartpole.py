"""
Linear Q-Learning for CartPole-v1

Approximates Q(s, a) = w_a · φ(s) where φ(s) is the normalized state vector.
Semi-gradient TD update:
    δ = r + γ max_a' Q(s', a') - Q(s, a)
    w_a += α · δ · φ(s)   (only updates weights for the taken action)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym

# Approximate observation bounds for normalization
OBS_BOUNDS = np.array([4.8, 4.0, 0.418, 4.0], dtype=np.float32)


def normalize(obs):
    return obs / OBS_BOUNDS


class LinearQAgent:
    def __init__(self, n_features=4, n_actions=2, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Weight matrix: W[a] is the weight vector for action a
        self.W = np.zeros((n_actions, n_features), dtype=np.float64)

    def q_values(self, phi):
        return self.W @ phi  # shape: (n_actions,)

    def act(self, phi, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(phi)))

    def update(self, phi, action, reward, phi_next, terminated):
        q_sa = self.q_values(phi)[action]
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(phi_next))
        delta = target - q_sa
        self.W[action] += self.alpha * delta * phi

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(n_episodes=5000, seed=42):
    env = gym.make("CartPole-v1")
    agent = LinearQAgent()
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        phi = normalize(obs)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(phi, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            phi_next = normalize(next_obs)
            agent.update(phi, action, reward, phi_next, terminated)
            phi = phi_next
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    env.close()
    return agent, episode_rewards


def evaluate(agent, n_episodes=200, seed=99):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        phi = normalize(obs)
        done = False
        total = 0
        while not done:
            action = int(np.argmax(agent.q_values(phi)))
            obs, r, terminated, truncated, _ = env.step(action)
            phi = normalize(obs)
            done = terminated or truncated
            total += r
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def plot_results(episode_rewards):
    os.makedirs("outputs", exist_ok=True)
    window = 100
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    rewards = np.array(episode_rewards)
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')

    axes[0].plot(rewards, alpha=0.2, color='steelblue', linewidth=0.7)
    axes[0].plot(range(window - 1, len(rewards)), rolling, color='navy',
                 linewidth=2, label=f'{window}-ep moving avg')
    axes[0].axhline(195, color='red', linestyle='--', linewidth=1.5, label='Solved (195)')
    axes[0].axhline(500, color='green', linestyle='--', linewidth=1, label='Max (500)')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Linear Q-Learning on CartPole-v1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.maximum.accumulate(rewards), color='#e74c3c', linewidth=2,
                 label='Best so far')
    axes[1].axhline(500, color='green', linestyle='--', linewidth=1, label='Max (500)')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Best Episode Reward So Far")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/linear_q_cartpole.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Plot saved to {path}")


def main():
    print("=== Linear Q-Learning on CartPole-v1 ===\n")
    print("Training for 5,000 episodes...")
    agent, rewards = train(n_episodes=5000)

    window = 100
    final_avg = float(np.mean(rewards[-window:]))
    print(f"Final {window}-episode avg reward (training): {final_avg:.1f}")

    mean_eval, std_eval = evaluate(agent)
    print(f"Greedy evaluation (200 eps): {mean_eval:.1f} ± {std_eval:.1f}")
    solved = mean_eval >= 195
    print(f"Solved (≥195 avg): {'✓ YES' if solved else '✗ not yet'}")

    print(f"\nLearned weights (W[action] = weights for action):")
    print(f"  Action 0 (push left):  {agent.W[0].round(4)}")
    print(f"  Action 1 (push right): {agent.W[1].round(4)}")

    plot_results(rewards)


if __name__ == "__main__":
    main()
