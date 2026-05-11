"""
Self-play Q-learning on Tic-Tac-Toe.

This is the simplest example of "self-play": a single Q-table plays *both*
sides of the game against itself, and improves as it does.  No human games,
no expert demonstrations, no opponent model -- the learner is its own
opponent, and they both get better in lockstep.

Notes on the implementation:

  * State        - the 9-cell board, encoded as a Python tuple of ints
                   0 = empty, 1 = X (player +1), 2 = O (player -1)
  * Q-table      - dict { (board_tuple, player) : np.ndarray(9) }
                   We learn one Q-table from BOTH players' perspectives,
                   using the trick that rewards are negated between turns:
                   X wins  -> +1 for X, -1 for O.

  * Self-play    - on every turn we use the SAME Q-table, but the current
                   player picks an action that maximises Q for them, and the
                   reward sign flips.

After training we run a short evaluation:
  - agent vs random opponent  (agent should win/draw nearly 100% of the time)
  - agent vs agent            (should always end in a draw -- tic-tac-toe is
                               a solved drawn game with perfect play)
And the agent plays a single sample game against a random opponent that is
printed move-by-move so you can read the result.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Tic-Tac-Toe rules
# -----------------------------------------------------------------------------
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def initial_board():
    return (0,) * 9


def legal_actions(board):
    return [i for i, v in enumerate(board) if v == 0]


def apply_action(board, action, player):
    """player is 1 (X) or 2 (O)."""
    new = list(board)
    new[action] = player
    return tuple(new)


def check_winner(board):
    """Return 1 if X wins, 2 if O wins, 0 if draw, None if game is ongoing."""
    for (a, b, c) in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(v != 0 for v in board):
        return 0
    return None


def render(board):
    chars = {0: ".", 1: "X", 2: "O"}
    rows = ["".join(chars[board[r * 3 + c]] for c in range(3)) for r in range(3)]
    return "\n".join(rows)


# -----------------------------------------------------------------------------
# Self-play Q-learning agent
# -----------------------------------------------------------------------------
class SelfPlayQ:
    """One Q-table indexed by (board, player_to_move).

    The Q-value Q[(s, p)][a] estimates the expected return for player p when
    they take action a from board s, with returns from p's point of view.
    """

    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2, rng=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()
        self.Q = {}

    def _q(self, board, player):
        key = (board, player)
        if key not in self.Q:
            self.Q[key] = np.zeros(9)
            # Set illegal actions to -inf so they are never picked.
            for a in range(9):
                if board[a] != 0:
                    self.Q[key][a] = -np.inf
        return self.Q[key]

    def act(self, board, player, greedy=False):
        legal = legal_actions(board)
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        q = self._q(board, player)
        # Argmax with random tie-breaking among legal moves.
        legal_q = np.array([q[a] for a in legal])
        max_q = legal_q.max()
        best = [a for a, qv in zip(legal, legal_q) if qv == max_q]
        return int(self.rng.choice(best))

    def update(self, board, player, action, reward, next_board, done):
        q = self._q(board, player)
        if done:
            target = reward
        else:
            # Next state's mover is the OTHER player.
            next_player = 3 - player
            next_q = self._q(next_board, next_player)
            # The next player will try to maximise THEIR Q -- which is the
            # negative of our return (zero-sum), so we subtract.
            target = reward - self.gamma * next_q.max()
        q[action] += self.alpha * (target - q[action])


# -----------------------------------------------------------------------------
# Training loop: self-play
# -----------------------------------------------------------------------------
def play_self_play_episode(agent):
    """One self-play episode. Both players act with the same agent."""
    board = initial_board()
    player = 1
    history = []  # (board, player, action)

    while True:
        action = agent.act(board, player)
        next_board = apply_action(board, action, player)
        winner = check_winner(next_board)
        history.append((board, player, action, next_board))
        if winner is not None:
            # Walk history backwards, assigning rewards.
            # reward = +1 if the player who just moved won;
            #          0 if draw;
            # for earlier players: shaped to 0, bootstrap handles the rest.
            for i, (s, p, a, ns) in enumerate(history):
                done = (i == len(history) - 1)
                if done:
                    if winner == 0:
                        r = 0.0
                    elif winner == p:
                        r = 1.0
                    else:
                        r = -1.0
                else:
                    r = 0.0
                agent.update(s, p, a, r, ns, done)
            return winner
        board = next_board
        player = 3 - player


# -----------------------------------------------------------------------------
# Opponents for evaluation
# -----------------------------------------------------------------------------
class RandomAgent:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def act(self, board, player, greedy=False):
        return int(self.rng.choice(legal_actions(board)))


def play_game(agent_X, agent_O, greedy_X=True, greedy_O=True, verbose=False):
    board = initial_board()
    player = 1
    while True:
        agent = agent_X if player == 1 else agent_O
        greedy = greedy_X if player == 1 else greedy_O
        action = agent.act(board, player, greedy=greedy)
        board = apply_action(board, action, player)
        if verbose:
            print(f"Player {'X' if player == 1 else 'O'} -> {action}")
            print(render(board) + "\n")
        winner = check_winner(board)
        if winner is not None:
            return winner
        player = 3 - player


def evaluate(agent, opponent, n_games=400, agent_plays=1):
    wins = draws = losses = 0
    for i in range(n_games):
        # alternate who goes first across games for fairness
        if (agent_plays == 1) ^ (i % 2 == 0):
            w = play_game(agent, opponent)
            our = 1
        else:
            w = play_game(opponent, agent)
            our = 2
        if w == 0:
            draws += 1
        elif w == our:
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== Self-play Q-learning on Tic-Tac-Toe ===")
    rng = np.random.default_rng(0)
    agent = SelfPlayQ(alpha=0.2, gamma=0.95, epsilon=0.2, rng=rng)
    random_op = RandomAgent(rng=np.random.default_rng(1))

    n_episodes = 50_000
    eval_every = 2_500
    eval_points, win_rates, draw_rates, loss_rates = [], [], [], []

    for ep in range(1, n_episodes + 1):
        play_self_play_episode(agent)
        # Slowly decay exploration so the agent commits to its best play.
        if ep % 5_000 == 0:
            agent.epsilon = max(0.02, agent.epsilon * 0.7)

        if ep % eval_every == 0:
            w, d, l = evaluate(agent, random_op, n_games=200)
            eval_points.append(ep)
            win_rates.append(w / 200)
            draw_rates.append(d / 200)
            loss_rates.append(l / 200)
            print(f"  ep={ep:>6}  vs random:  win={w/200:.2f}  draw={d/200:.2f}  loss={l/200:.2f}  "
                  f"|Q|={len(agent.Q)}  eps={agent.epsilon:.3f}")

    # Final evaluation
    print("\nFinal evaluation:")
    w, d, l = evaluate(agent, random_op, n_games=1000)
    print(f"  vs Random   (1000 games): wins={w}  draws={d}  losses={l}")

    # Self vs self: should virtually always draw -- tic-tac-toe is a drawn game.
    rng2 = np.random.default_rng(2)
    eval_agent = SelfPlayQ(rng=rng2)
    eval_agent.Q = agent.Q  # share the table, but greedy decisions
    self_draws = 0
    for _ in range(200):
        w = play_game(eval_agent, eval_agent)
        if w == 0:
            self_draws += 1
    print(f"  agent vs itself (200 games, greedy): draws={self_draws} / 200")

    # Show one game against a random opponent move-by-move.
    print("\nSample game (trained agent X vs random O):")
    play_game(agent, random_op, verbose=True)

    # Plot the win-rate curve
    plt.figure(figsize=(10, 5))
    plt.plot(eval_points, win_rates, label="Win",  color="#27ae60", linewidth=2)
    plt.plot(eval_points, draw_rates, label="Draw", color="#f39c12", linewidth=2)
    plt.plot(eval_points, loss_rates, label="Loss", color="#e74c3c", linewidth=2)
    plt.xlabel("Self-play episodes")
    plt.ylabel("Fraction of games vs random opponent")
    plt.title("Self-play Q-learning: skill vs a random opponent over time")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "self_play_tic_tac_toe.png")
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
