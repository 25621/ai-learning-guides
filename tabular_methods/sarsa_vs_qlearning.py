"""
SARSA vs Q-Learning: Safe vs Optimal Paths on Cliff Walking

Key difference:
- Q-learning (off-policy): learns the optimal policy (hug the cliff = -12 steps)
  but during training its ε-exploration causes it to fall off frequently → high variance
- SARSA (on-policy): its updates account for its own ε-greedy exploration,
  so it learns a SAFER path (detour along the top = ~-20 steps) that avoids
  the cliff edge during training

This difference vanishes as ε→0, but at finite ε it's a fundamental distinction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym


def run_agent(mode, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42):
    """
    mode: 'sarsa' or 'qlearning'
    Returns (Q, episode_rewards)
    """
    env = gym.make("CliffWalking-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        action = (env.action_space.sample() if rng.random() < epsilon
                  else np.argmax(Q[obs]))
        total_reward = 0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if mode == 'sarsa':
                next_action = (env.action_space.sample() if rng.random() < epsilon
                               else np.argmax(Q[next_obs]))
                td_target = reward + gamma * Q[next_obs, next_action]
            else:  # q-learning
                td_target = reward + gamma * np.max(Q[next_obs])
                next_action = (env.action_space.sample() if rng.random() < epsilon
                               else np.argmax(Q[next_obs]))

            Q[obs, action] += alpha * (td_target - Q[obs, action])
            obs = next_obs
            action = next_action

        episode_rewards.append(total_reward)

    env.close()
    return Q, episode_rewards


def trace_greedy_path(Q):
    """Simulate one deterministic episode following greedy policy."""
    n_cols = 12
    env = gym.make("CliffWalking-v1")
    obs, _ = env.reset()
    path = [obs]
    done = False
    steps = 0
    total_reward = 0
    while not done and steps < 100:
        action = np.argmax(Q[obs])
        obs, reward, terminated, truncated, _ = env.step(action)
        path.append(obs)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    env.close()
    return path, total_reward


def visualize_paths(Q_sarsa, Q_ql):
    n_rows, n_cols = 4, 12
    path_sarsa, r_sarsa = trace_greedy_path(Q_sarsa)
    path_ql, r_ql = trace_greedy_path(Q_ql)

    def path_to_grid(path):
        grid = np.zeros((n_rows, n_cols))
        for s in path:
            r, c = s // n_cols, s % n_cols
            grid[r, c] += 1
        return grid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, Q, path, reward, title, color in [
        (axes[0], Q_sarsa, path_sarsa, r_sarsa, f'SARSA Greedy Path (reward={r_sarsa})', '#3498db'),
        (axes[1], Q_ql, path_ql, r_ql, f'Q-Learning Greedy Path (reward={r_ql})', '#e74c3c'),
    ]:
        grid = path_to_grid(path)
        ax.imshow(grid, cmap='Blues' if 'SARSA' in title else 'Reds',
                  vmin=0, vmax=max(grid.max(), 1))

        # Mark cliff
        for c in range(1, 11):
            ax.add_patch(plt.Rectangle((c - 0.5, 2.5), 1, 1,
                                       fill=True, color='orange', alpha=0.4))
            ax.text(c, 3, 'C', ha='center', va='center', fontsize=7, color='brown')
        ax.text(0, 3, 'S', ha='center', va='center', fontsize=9, fontweight='bold', color='green')
        ax.text(11, 3, 'G', ha='center', va='center', fontsize=9, fontweight='bold', color='purple')

        for r in range(n_rows + 1):
            ax.axhline(r - 0.5, color='gray', lw=0.4)
        for c in range(n_cols + 1):
            ax.axvline(c - 0.5, color='gray', lw=0.4)

        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = "outputs/sarsa_vs_qlearning_paths.png"
    plt.savefig(out_path, dpi=120)
    print(f"Path comparison saved to {out_path}")


def plot_learning_curves(sarsa_rewards, ql_rewards):
    window = 10
    s_smooth = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')
    q_smooth = np.convolve(ql_rewards, np.ones(window) / window, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s_smooth, color='#3498db', linewidth=2,
            label=f'SARSA ({window}-ep avg) — safe detour')
    ax.plot(q_smooth, color='#e74c3c', linewidth=2,
            label=f'Q-Learning ({window}-ep avg) — hugs cliff, falls often')
    ax.axhline(-13, color='green', linestyle='--', linewidth=1,
               label='Optimal (cliff-hugging) path = -13')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("SARSA vs Q-Learning: Online Performance on Cliff Walking (ε=0.1)")
    ax.set_ylim(-200, 0)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = "outputs/sarsa_vs_qlearning_curves.png"
    plt.savefig(out_path, dpi=120)
    print(f"Learning curves saved to {out_path}")


def main():
    print("=== SARSA vs Q-Learning on Cliff Walking ===\n")
    print("Training SARSA (500 episodes)...")
    Q_sarsa, sarsa_rewards = run_agent('sarsa', n_episodes=500)

    print("Training Q-Learning (500 episodes)...")
    Q_ql, ql_rewards = run_agent('qlearning', n_episodes=500)

    sarsa_final = np.mean(sarsa_rewards[-50:])
    ql_final = np.mean(ql_rewards[-50:])

    print(f"\nFinal 50-ep avg reward:")
    print(f"  SARSA:      {sarsa_final:.1f}  (learns safe detour)")
    print(f"  Q-Learning: {ql_final:.1f}  (converges to optimal but falls often during training)")
    print()
    print("Key insight:")
    print("  SARSA accounts for ε-greedy exploration in its updates → learns safer path")
    print("  Q-learning always updates toward the greedy max → learns optimal but riskier path")
    print("  As ε→0 both algorithms converge to the same (optimal) policy.")

    _, r_sarsa = trace_greedy_path(Q_sarsa)
    _, r_ql = trace_greedy_path(Q_ql)
    print(f"\nGreedy evaluation (no exploration):")
    print(f"  SARSA greedy path reward:      {r_sarsa}")
    print(f"  Q-Learning greedy path reward: {r_ql}")

    plot_learning_curves(sarsa_rewards, ql_rewards)
    visualize_paths(Q_sarsa, Q_ql)


if __name__ == "__main__":
    main()
