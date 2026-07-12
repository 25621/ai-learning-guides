"""A gridworld MDP with every piece of (S, A, P, R, gamma) as an explicit array.

The point of this module is that nothing is hidden behind an env.step() call:
`P` is a (S, A, S') tensor of transition probabilities, `R` is a (S, A) matrix
of expected immediate rewards, and every algorithm in Phase 1 is a few lines of
numpy on top of those arrays. Imported by projects 02, 04, and 05 via sys.path.

Conventions
-----------
- A state is a grid cell (row, col); wall cells are not states at all.
- Actions are 0=up, 1=right, 2=down, 3=left. A move that would leave the grid
  or enter a wall keeps the agent where it is (a "bump").
- With slip probability `slip`, the agent moves perpendicular to the intended
  direction (slip/2 each way) instead of where it aimed.
- Rewards live on *arrival*: every transition pays `step_reward`, plus the
  terminal cell's bonus if the transition enters a terminal cell.
- Terminal cells are absorbing: every action self-loops with reward 0, so the
  discounted sum after termination is exactly 0.
"""

import numpy as np

ACTIONS = ["up", "right", "down", "left"]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["↑", "→", "↓", "←"]


class Gridworld:
    def __init__(self, n_rows=5, n_cols=5, walls=(), terminals=None,
                 step_reward=-0.04, slip=0.2, gamma=0.9):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.walls = set(walls)
        self.terminals = dict(terminals or {})   # {(row, col): bonus}
        self.step_reward = step_reward
        self.slip = slip
        self.gamma = gamma

        self.states = [(r, c) for r in range(n_rows) for c in range(n_cols)
                       if (r, c) not in self.walls]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.n_states, self.n_actions = len(self.states), len(ACTIONS)
        self.P, self.R = self._build()

    def _move(self, cell, a):
        r, c = cell
        dr, dc = DELTAS[a]
        nxt = (r + dr, c + dc)
        if (nxt[0] < 0 or nxt[0] >= self.n_rows or
                nxt[1] < 0 or nxt[1] >= self.n_cols or nxt in self.walls):
            return cell                                   # bump: stay put
        return nxt

    def _build(self):
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        for s, cell in enumerate(self.states):
            if cell in self.terminals:                    # absorbing
                P[s, :, s] = 1.0
                continue
            for a in range(A):
                # (probability, realized direction) pairs
                outcomes = [(1.0 - self.slip, a),
                            (self.slip / 2, (a - 1) % 4),
                            (self.slip / 2, (a + 1) % 4)]
                for prob, direction in outcomes:
                    if prob == 0.0:
                        continue
                    nxt = self._move(cell, direction)
                    s2 = self.index[nxt]
                    P[s, a, s2] += prob
                    R[s, a] += prob * (self.step_reward
                                       + self.terminals.get(nxt, 0.0))
        assert np.allclose(P.sum(axis=2), 1.0), "each P[s, a, :] must sum to 1"
        return P, R


# ---------------------------------------------------------------------------
# Planning with the explicit arrays
# ---------------------------------------------------------------------------

def value_iteration(P, R, gamma, tol=1e-10, max_iters=200_000):
    """Synchronous optimality backups V <- max_a (R + gamma P V).

    Returns (V, Q, greedy policy, iterations, history of V after each backup).
    """
    S = P.shape[0]
    V = np.zeros(S)
    history = [V.copy()]
    for k in range(1, max_iters + 1):
        Q = R + gamma * P @ V                     # (S, A)
        V_new = Q.max(axis=1)
        history.append(V_new.copy())
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * P @ V
    return V, Q, Q.argmax(axis=1), k, history


def policy_matrices(P, R, pi):
    """Collapse (P, R) under a stochastic policy pi (S, A) into P_pi, r_pi."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    return P_pi, r_pi


def policy_evaluation_solve(P, R, pi, gamma):
    """Closed form: V^pi = (I - gamma P_pi)^-1 r_pi, via a linear solve."""
    P_pi, r_pi = policy_matrices(P, R, pi)
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * P_pi, r_pi)


def policy_evaluation_iterative(P, R, pi, gamma, tol=1e-9, max_iters=200_000):
    """Iterative expectation backups V <- r_pi + gamma P_pi V."""
    P_pi, r_pi = policy_matrices(P, R, pi)
    V = np.zeros(P.shape[0])
    history = [V.copy()]
    for k in range(1, max_iters + 1):
        V_new = r_pi + gamma * P_pi @ V
        history.append(V_new.copy())
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V, k, history


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

WALL_COLOR = "#52514e"
GOAL_COLOR = "#1baf7a"
PIT_COLOR = "#e34948"


def _cell_colors(gw, V):
    """Sequential single-hue fill (light -> blue) for the value heatmap."""
    import matplotlib.colors as mcolors
    lo, hi = float(np.min(V)), float(np.max(V))
    span = (hi - lo) or 1.0
    light, dark = np.array(mcolors.to_rgb("#eaf1fb")), np.array(mcolors.to_rgb("#2a78d6"))
    return {s: tuple(light + (dark - light) * ((V[i] - lo) / span))
            for i, s in enumerate(gw.states)}


def draw_grid(ax, gw, V=None, policy=None, start=None, show_values=True,
              title=None):
    """Draw the world; optionally a value heatmap and greedy-policy arrows."""
    import matplotlib.patches as mpatches

    fills = _cell_colors(gw, V) if V is not None else {}
    for r in range(gw.n_rows):
        for c in range(gw.n_cols):
            cell = (r, c)
            x, y = c, gw.n_rows - 1 - r          # row 0 on top
            if cell in gw.walls:
                face = WALL_COLOR
            elif cell in gw.terminals:
                face = GOAL_COLOR if gw.terminals[cell] > 0 else PIT_COLOR
            else:
                face = fills.get(cell, "#f4f4f1")
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=face,
                                            edgecolor="#fcfcfb", linewidth=2))
            if cell in gw.walls:
                continue
            i = gw.index[cell]
            if cell in gw.terminals:
                ax.text(x + 0.5, y + 0.5, f"{gw.terminals[cell]:+g}",
                        ha="center", va="center", fontsize=12, color="white",
                        fontweight="bold")
                continue
            if policy is not None:
                ax.text(x + 0.5, y + 0.62 if show_values and V is not None else y + 0.5,
                        ARROWS[int(policy[i])], ha="center", va="center",
                        fontsize=13, color="#0b0b0b")
            if show_values and V is not None:
                ax.text(x + 0.5, y + 0.28, f"{V[i]:.2f}", ha="center",
                        va="center", fontsize=7.5, color="#52514e")
            if start == cell:
                ax.text(x + 0.14, y + 0.82, "S", ha="center", va="center",
                        fontsize=10, color="#0b0b0b", fontweight="bold")
    ax.set_xlim(0, gw.n_cols)
    ax.set_ylim(0, gw.n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b", pad=8)


# ---------------------------------------------------------------------------
# The default 5x5 world shared by projects 01 and 02
# ---------------------------------------------------------------------------

def default_world(gamma=0.9):
    """5x5 grid: +1 goal, -1 pit, three walls, -0.04 step cost, 20% slip."""
    return Gridworld(
        n_rows=5, n_cols=5,
        walls=[(1, 1), (1, 2), (3, 3)],
        terminals={(0, 4): +1.0, (2, 4): -1.0},
        step_reward=-0.04, slip=0.2, gamma=gamma,
    )
