"""
Multi-agent RL in simple matrix games.

A *matrix game* is the simplest possible multi-agent setting:
  - Every "episode" is a single simultaneous move by each agent.
  - Each agent picks one of a small number of actions.
  - A payoff matrix decides who gets what reward.

We train two **independent Q-learners** (each agent treats the other as part
of a non-stationary environment) on three classic games:

    1) Rock-Paper-Scissors        -- zero-sum, mixed Nash equilibrium (1/3,1/3,1/3)
    2) Prisoner's Dilemma         -- general-sum, dominant strategy (defect, defect)
    3) Stag Hunt                  -- coordination game, two pure Nash equilibria

The "learning" here is purely in each agent's policy distribution.  We track
the *action frequencies* and watch them approach (or fail to approach) the
game-theoretic prediction.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Payoff matrices: payoffs[a0, a1] = (reward_for_player_0, reward_for_player_1)
# -----------------------------------------------------------------------------
def rock_paper_scissors():
    # actions: 0=Rock, 1=Paper, 2=Scissors
    p = np.zeros((3, 3, 2))
    # who_beats[(a0, a1)] = 1 means player 0 wins
    rule = {(0, 2): 1, (2, 1): 1, (1, 0): 1}
    for a0 in range(3):
        for a1 in range(3):
            if a0 == a1:
                p[a0, a1] = (0, 0)
            elif rule.get((a0, a1)) == 1:
                p[a0, a1] = (1, -1)
            else:
                p[a0, a1] = (-1, 1)
    return p, ["Rock", "Paper", "Scissors"], "Rock-Paper-Scissors"


def prisoners_dilemma():
    # actions: 0=Cooperate, 1=Defect
    # classic payoffs: T=5 > R=3 > P=1 > S=0
    p = np.array([
        [[3, 3], [0, 5]],
        [[5, 0], [1, 1]],
    ], dtype=float)
    return p, ["Cooperate", "Defect"], "Prisoner's Dilemma"


def stag_hunt():
    # actions: 0=Stag (cooperate on the big prize), 1=Hare (safe small prize)
    # Two pure Nash: (Stag,Stag) is payoff-dominant; (Hare,Hare) is risk-dominant.
    p = np.array([
        [[4, 4], [0, 3]],
        [[3, 0], [2, 2]],
    ], dtype=float)
    return p, ["Stag", "Hare"], "Stag Hunt"


# -----------------------------------------------------------------------------
# Stateless Q-learner: there is no "state" in a 1-shot matrix game.
# Q is just one number per action.
# -----------------------------------------------------------------------------
class StatelessQLearner:
    def __init__(self, n_actions, alpha=0.05, epsilon=0.1, rng=None):
        self.Q = np.zeros(n_actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()

    def act(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # break ties uniformly at random so we don't deterministically pick action 0
        max_q = self.Q.max()
        candidates = np.flatnonzero(self.Q == max_q)
        return int(self.rng.choice(candidates))

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_matrix_game(payoffs, action_names, title, n_steps=20_000, seed=0):
    rng = np.random.default_rng(seed)
    n_actions = payoffs.shape[0]

    a0 = StatelessQLearner(n_actions, rng=rng)
    a1 = StatelessQLearner(n_actions, rng=rng)

    # Track empirical action frequencies in a rolling window so the policies
    # can be seen evolving over training.
    window = 500
    freqs0 = np.zeros((n_steps, n_actions))
    freqs1 = np.zeros((n_steps, n_actions))
    counts0 = np.zeros(n_actions)
    counts1 = np.zeros(n_actions)
    history0 = []
    history1 = []

    for t in range(n_steps):
        u = a0.act()
        v = a1.act()
        r0, r1 = payoffs[u, v]
        a0.update(u, r0)
        a1.update(v, r1)

        history0.append(u)
        history1.append(v)
        counts0[u] += 1
        counts1[v] += 1
        if t >= window:
            old_u = history0[t - window]
            old_v = history1[t - window]
            counts0[old_u] -= 1
            counts1[old_v] -= 1
            denom = window
        else:
            denom = t + 1
        freqs0[t] = counts0 / denom
        freqs1[t] = counts1 / denom

    print(f"\n--- {title} ---")
    print(f"Final Q-values:")
    print(f"  Player 0: {dict(zip(action_names, np.round(a0.Q, 3)))}")
    print(f"  Player 1: {dict(zip(action_names, np.round(a1.Q, 3)))}")
    print(f"Empirical action frequencies over the last {window} steps:")
    print(f"  Player 0: {dict(zip(action_names, np.round(freqs0[-1], 3)))}")
    print(f"  Player 1: {dict(zip(action_names, np.round(freqs1[-1], 3)))}")

    return freqs0, freqs1


def plot_game(freqs0, freqs1, action_names, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12"]
    n_steps = freqs0.shape[0]
    xs = np.arange(n_steps)
    for ax, freqs, who in zip(axes, [freqs0, freqs1], ["Player 0", "Player 1"]):
        for i, name in enumerate(action_names):
            ax.plot(xs, freqs[:, i], label=name, color=colors[i], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_title(who)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Action frequency (rolling window)")
    fig.suptitle(f"Independent Q-learning on {title}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot saved to {out}")


def main():
    print("=== Multi-agent RL on matrix games ===")

    rps, rps_names, rps_title = rock_paper_scissors()
    f0, f1 = train_matrix_game(rps, rps_names, rps_title, n_steps=20_000, seed=0)
    plot_game(f0, f1, rps_names, rps_title, "rps.png")

    pd, pd_names, pd_title = prisoners_dilemma()
    f0, f1 = train_matrix_game(pd, pd_names, pd_title, n_steps=20_000, seed=0)
    plot_game(f0, f1, pd_names, pd_title, "prisoners_dilemma.png")

    sh, sh_names, sh_title = stag_hunt()
    f0, f1 = train_matrix_game(sh, sh_names, sh_title, n_steps=20_000, seed=0)
    plot_game(f0, f1, sh_names, sh_title, "stag_hunt.png")


if __name__ == "__main__":
    main()
