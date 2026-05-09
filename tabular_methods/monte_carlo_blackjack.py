"""
Monte Carlo Control for Blackjack

Gymnasium's Blackjack-v1:
- State: (player_sum, dealer_showing, usable_ace)
- Actions: 0 = stick, 1 = hit
- Reward: +1 win, -1 lose, 0 draw (natural blackjack: +1.5)

Monte Carlo (every-visit, ε-soft) control:
  1. Generate a full episode following ε-greedy policy
  2. For each (s, a) visited, record return G
  3. Update Q(s,a) ← average of all observed returns
  4. Improve policy ε-greedily from Q

No bootstrapping — waits for the episode to end before updating.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import gymnasium as gym


def epsilon_greedy(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    q_vals = [Q[(state, a)] for a in range(n_actions)]
    return int(np.argmax(q_vals))


def generate_episode(env, Q, epsilon, rng, n_actions):
    """Run one episode; return list of (state, action, reward)."""
    obs, _ = env.reset()
    episode = []
    done = False
    while not done:
        action = epsilon_greedy(Q, obs, n_actions, epsilon, rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode.append((obs, action, reward))
        done = terminated or truncated
        obs = next_obs
    return episode


def mc_control(n_episodes=500_000, gamma=1.0, epsilon=0.1, seed=42):
    env = gym.make("Blackjack-v1")
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)

    Q = defaultdict(float)
    returns_count = defaultdict(int)
    returns_sum = defaultdict(float)

    win_rates = []
    window = 1000

    wins = 0
    for ep in range(n_episodes):
        episode = generate_episode(env, Q, epsilon, rng, n_actions)

        # Compute returns (every-visit MC)
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            key = (state, action)
            returns_sum[key] += G
            returns_count[key] += 1
            Q[key] = returns_sum[key] / returns_count[key]

        if episode[-1][2] > 0:
            wins += 1

        if (ep + 1) % window == 0:
            win_rates.append(wins / window)
            wins = 0

    env.close()
    return Q, win_rates


def evaluate_greedy(Q, n_episodes=100_000, seed=999):
    """Evaluate pure greedy policy (ε=0) derived from Q."""
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(Q, obs, n_actions, 0.0, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append(reward)
    env.close()
    wins = sum(r > 0 for r in results)
    draws = sum(r == 0 for r in results)
    losses = sum(r < 0 for r in results)
    return wins / n_episodes, draws / n_episodes, losses / n_episodes


def plot_value_function(Q):
    """Plot V(s) = max_a Q(s,a) for usable and non-usable ace."""
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': '3d'})

    for ax, usable_ace, title in [
        (axes[0], True, 'Usable Ace'),
        (axes[1], False, 'No Usable Ace'),
    ]:
        V = np.zeros((len(player_sums), len(dealer_cards)))
        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                state = (ps, dc, usable_ace)
                V[i, j] = max(Q[(state, a)] for a in range(2))

        X, Y = np.meshgrid(dealer_cards, player_sums)
        ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.8)
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("V*(s)")
        ax.set_title(f"MC Value Function — {title}")

    plt.tight_layout()
    out_path = "outputs/monte_carlo_blackjack_values.png"
    plt.savefig(out_path, dpi=120)
    print(f"Value function plot saved to {out_path}")


def plot_policy(Q):
    """Plot the greedy policy (hit/stick) for both ace cases."""
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, usable_ace, title in [
        (axes[0], True, 'Usable Ace'),
        (axes[1], False, 'No Usable Ace'),
    ]:
        policy_grid = np.zeros((len(player_sums), len(dealer_cards)))
        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                state = (ps, dc, usable_ace)
                q0 = Q[(state, 0)]
                q1 = Q[(state, 1)]
                policy_grid[i, j] = 1 if q1 > q0 else 0  # 1=hit, 0=stick

        im = ax.imshow(policy_grid, origin='lower', cmap='RdYlGn',
                       extent=[0.5, 10.5, 11.5, 21.5], aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_title(f"Policy (Green=Hit, Red=Stick) — {title}")
        ax.set_xticks(dealer_cards)
        ax.set_yticks(player_sums)
        plt.colorbar(im, ax=ax, ticks=[0, 1], label='0=Stick, 1=Hit')

    plt.tight_layout()
    out_path = "outputs/monte_carlo_blackjack_policy.png"
    plt.savefig(out_path, dpi=120)
    print(f"Policy plot saved to {out_path}")


def plot_win_rates(win_rates):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(win_rates, color='#2ecc71', linewidth=2)
    ax.axhline(np.mean(win_rates[-50:]), color='red', linestyle='--',
               linewidth=1.5, label=f'Final avg: {np.mean(win_rates[-50:])*100:.1f}%')
    ax.set_xlabel("Episode (×1000)")
    ax.set_ylabel("Win Rate (1000-ep window)")
    ax.set_title("MC Control: Win Rate over Training (Blackjack)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = "outputs/monte_carlo_blackjack_winrate.png"
    plt.savefig(out_path, dpi=120)
    print(f"Win rate plot saved to {out_path}")


def main():
    print("=== Monte Carlo Control for Blackjack ===\n")
    print("Training for 500,000 episodes (this may take ~30s)...")
    Q, win_rates = mc_control(n_episodes=500_000)

    print(f"\nTraining win rate (last 50 windows): {np.mean(win_rates[-50:])*100:.1f}%")

    print("\nEvaluating greedy policy (100,000 episodes)...")
    win_r, draw_r, loss_r = evaluate_greedy(Q, n_episodes=100_000)
    print(f"Greedy policy evaluation:")
    print(f"  Wins:   {win_r*100:.1f}%")
    print(f"  Draws:  {draw_r*100:.1f}%")
    print(f"  Losses: {loss_r*100:.1f}%")
    print()
    print("Note: A basic strategy (optimal tabular policy) wins ~42-43% in Blackjack.")
    print("MC control should approach this level with enough episodes.")

    plot_win_rates(win_rates)
    plot_value_function(Q)
    plot_policy(Q)


if __name__ == "__main__":
    main()
