"""
SARSA for Cliff Walking

CliffWalking-v1: 4x12 grid.
- Start: bottom-left (row 3, col 0)
- Goal:  bottom-right (row 3, col 11)
- Cliff: bottom row columns 1-10  (reward = -100, episode resets)
- All other steps: reward = -1

SARSA is an on-policy TD control algorithm:
  Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
where a' is the ACTUAL next action chosen by the policy (not the max).

This makes SARSA "safer": it accounts for its own exploratory steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym


def sarsa(n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42):
    env = gym.make("CliffWalking-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Choose first action
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[obs])

        total_reward = 0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Choose next action using ε-greedy (on-policy!)
            if rng.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_obs])

            # SARSA update — uses actual next action (not max)
            Q[obs, action] += alpha * (
                reward + gamma * Q[next_obs, next_action] - Q[obs, action]
            )

            obs = next_obs
            action = next_action

        episode_rewards.append(total_reward)

    env.close()
    return Q, episode_rewards


def print_policy(Q, label):
    actions = ['↑', '→', '↓', '←']
    n_rows, n_cols = 4, 12
    print(f"\n{label}")
    print("-" * (n_cols * 4 + 1))
    for r in range(n_rows):
        row = "|"
        for c in range(n_cols):
            s = r * n_cols + c
            if r == 3 and 1 <= c <= 10:
                row += " C |"  # cliff
            elif r == 3 and c == 11:
                row += " G |"  # goal
            elif r == 3 and c == 0:
                row += " S |"  # start
            else:
                row += f" {actions[np.argmax(Q[s])]} |"
        print(row)
    print("-" * (n_cols * 4 + 1))
    print("(S=Start, G=Goal, C=Cliff)")


def plot_results(sarsa_rewards):
    window = 10
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sarsa_rewards, alpha=0.3, color='#3498db', linewidth=0.8)
    ax.plot(sarsa_smooth, color='#3498db', linewidth=2, label=f'SARSA ({window}-ep avg)')
    ax.axhline(-13, color='green', linestyle='--', linewidth=1.5,
               label='Optimal safe path (~-13)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("SARSA on Cliff Walking")
    ax.set_ylim(-200, 0)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/sarsa_cliff_walking.png"
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


def main():
    print("=== SARSA on Cliff Walking ===\n")
    Q_sarsa, sarsa_rewards = sarsa(n_episodes=500)

    final_avg = np.mean(sarsa_rewards[-50:])
    print(f"Final 50-episode average reward: {final_avg:.1f}")
    print(f"(Optimal safe path is -13; SARSA learns a cautious detour around the cliff)")

    print_policy(Q_sarsa, "SARSA Learned Policy:")
    plot_results(sarsa_rewards)


if __name__ == "__main__":
    main()
