"""
Multi-Armed Bandit Problem from Scratch

The classic exploration-exploitation dilemma:
- K slot machines ("arms"), each with unknown reward distributions
- Goal: maximize cumulative reward over T time steps
- Strategy: epsilon-greedy (exploit best-known arm most of the time,
  explore randomly with probability epsilon)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class MultiArmedBandit:
    """K-armed bandit with Gaussian reward distributions."""

    def __init__(self, k=10, seed=42):
        self.k = k
        rng = np.random.default_rng(seed)
        # True mean reward for each arm, drawn from N(0,1)
        self.true_means = rng.normal(0, 1, k)
        self.optimal_arm = np.argmax(self.true_means)

    def pull(self, arm, rng):
        """Return a noisy reward from the chosen arm."""
        return rng.normal(self.true_means[arm], 1.0)


class EpsilonGreedyAgent:
    """Epsilon-greedy agent with sample-average value estimates."""

    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)   # estimated values
        self.n = np.zeros(k)   # pull counts

    def select_action(self, rng):
        if rng.random() < self.epsilon:
            return rng.integers(self.k)
        return np.argmax(self.q)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


def run_experiment(bandit, epsilon, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    agent = EpsilonGreedyAgent(bandit.k, epsilon)
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for t in range(steps):
        arm = agent.select_action(rng)
        reward = bandit.pull(arm, rng)
        agent.update(arm, reward)
        rewards[t] = reward
        optimal_actions[t] = (arm == bandit.optimal_arm)

    return rewards, optimal_actions


def main():
    np.random.seed(42)
    bandit = MultiArmedBandit(k=10, seed=42)

    print("=== Multi-Armed Bandit Experiment ===")
    print(f"True arm means: {bandit.true_means.round(3)}")
    print(f"Optimal arm: {bandit.optimal_arm} (mean={bandit.true_means[bandit.optimal_arm]:.3f})")
    print()

    epsilons = [0.0, 0.01, 0.1]
    steps = 1000
    runs = 200
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    avg_rewards = {eps: np.zeros(steps) for eps in epsilons}
    avg_optimal = {eps: np.zeros(steps) for eps in epsilons}

    for eps in epsilons:
        for run in range(runs):
            r, o = run_experiment(bandit, eps, steps=steps, seed=run)
            avg_rewards[eps] += r
            avg_optimal[eps] += o
        avg_rewards[eps] /= runs
        avg_optimal[eps] /= runs
        print(f"epsilon={eps:.2f} | final avg reward: {avg_rewards[eps][-100:].mean():.3f} "
              f"| optimal % (last 100 steps): {avg_optimal[eps][-100:].mean()*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for eps, color in zip(epsilons, colors):
        label = f"ε={eps}"
        ax1.plot(avg_rewards[eps], label=label, color=color, linewidth=1.5)
        ax2.plot(avg_optimal[eps] * 100, label=label, color=color, linewidth=1.5)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Multi-Armed Bandit: Average Reward over Time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_title("Multi-Armed Bandit: % Optimal Action Chosen")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    out_path = "outputs/multi_armed_bandit.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
