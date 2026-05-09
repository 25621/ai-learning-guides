"""
Q-Learning Agent for Frozen Lake

FrozenLake-v1 (4x4, slippery): navigate from S to G without falling into holes.
Q-learning is an off-policy TD control algorithm:
  Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

Key properties:
- Off-policy: learns the optimal Q regardless of which actions are taken
- Uses an ε-greedy policy for exploration
- Converges to Q* under standard conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import gymnasium as gym


def q_learning(n_episodes=50_000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
               seed=42):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start

    episode_rewards = []
    episode_successes = []
    epsilons = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        total_reward = 0

        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[obs])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update (off-policy: uses max over next actions)
            best_next = np.max(Q[next_obs])
            Q[obs, action] += alpha * (reward + gamma * best_next - Q[obs, action])

            obs = next_obs
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_successes.append(1 if total_reward > 0 else 0)
        epsilons.append(epsilon)

    env.close()
    return Q, episode_rewards, episode_successes, epsilons


def evaluate_policy(Q, n_episodes=1000, seed=99):
    """Evaluate greedy policy derived from Q."""
    env = gym.make("FrozenLake-v1", is_slippery=True)
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done = False
        while not done:
            action = np.argmax(Q[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:
                successes += 1
    env.close()
    return successes / n_episodes


def print_q_table(Q):
    actions = ['←', '↓', '→', '↑']
    print("\nQ-Table (best action per state):")
    print("  ", "  ".join(f"Col{c}" for c in range(4)))
    labels = ['S', 'F', 'F', 'F', 'F', 'H', 'F', 'H',
              'F', 'F', 'F', 'H', 'H', 'F', 'F', 'G']
    for r in range(4):
        row_str = f"Row{r} "
        for c in range(4):
            s = r * 4 + c
            best_a = np.argmax(Q[s])
            row_str += f" {actions[best_a]}({labels[s]}) "
        print(row_str)


def plot_results(episode_rewards, episode_successes, epsilons):
    window = 200
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Smoothed success rate
    successes = np.array(episode_successes, dtype=float)
    rolling_success = np.convolve(successes, np.ones(window) / window, mode='valid')
    axes[0].plot(rolling_success, color='#2ecc71', linewidth=2)
    axes[0].axhline(0.7, color='red', linestyle='--', linewidth=1, label='70% target')
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title(f"Q-Learning on Frozen Lake ({window}-ep rolling success rate)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Raw rewards
    axes[1].plot(episode_rewards, alpha=0.2, color='steelblue', linewidth=0.5)
    rolling_r = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    axes[1].plot(rolling_r, color='navy', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Episode Rewards")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Epsilon decay
    axes[2].plot(epsilons, color='#e74c3c', linewidth=1.5)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_title("Exploration Rate (ε) Decay")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/q_learning_frozen_lake.png"
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


def main():
    print("=== Q-Learning on Frozen Lake ===\n")
    print("Training for 50,000 episodes...")
    Q, rewards, successes, epsilons = q_learning(n_episodes=10_000)

    window = 500
    final_success = np.mean(successes[-window:])
    print(f"Final {window}-episode success rate (training): {final_success*100:.1f}%")

    eval_rate = evaluate_policy(Q, n_episodes=1000)
    print(f"Greedy evaluation success rate (1000 eps):       {eval_rate*100:.1f}%")
    print(f"Milestone target (>70%): {'✓ PASSED' if eval_rate >= 0.70 else '✗ not yet'}")

    print_q_table(Q)
    plot_results(rewards, successes, epsilons)


if __name__ == "__main__":
    main()
