"""Shared tools for the FrozenLake dynamic-programming and TD projects.

Everything here works on *explicit* MDP arrays extracted from a Gymnasium
toy-text environment: P with shape (S, A, S') and R with shape (S, A), the
same representation project 01 built by hand for its gridworld. Imported by
projects 07 and 09 via sys.path.

FrozenLake actions: 0=left, 1=down, 2=right, 3=up.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01-build-a-gridworld"))
from plot_style import INK, INK_SECONDARY  # noqa: E402

ARROWS = "←↓→↑"  # left, down, right, up


# ---------------------------------------------------------------------------
# From a Gymnasium env to (P, R) arrays
# ---------------------------------------------------------------------------

def mdp_arrays(env):
    """Extract dense P (S, A, S') and R (S, A) from a toy-text env.

    Gymnasium stores dynamics as env.unwrapped.P[s][a] = list of
    (prob, next_state, reward, terminated) tuples; R[s, a] is the
    expected immediate reward under that distribution. Terminal states
    already self-loop with reward 0, i.e. they are absorbing.
    """
    u = env.unwrapped
    S, A = u.observation_space.n, u.action_space.n
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for prob, s2, r, _term in u.P[s][a]:
                P[s, a, s2] += prob
                R[s, a] += prob * r
    assert np.allclose(P.sum(axis=2), 1.0)
    return P, R


def terminal_states(P, R):
    """Boolean mask of absorbing states (all actions self-loop, reward 0)."""
    S = P.shape[0]
    self_loop = np.array([np.allclose(P[s, :, s], 1.0) for s in range(S)])
    return self_loop & np.all(R == 0, axis=1)


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    """Bellman optimality backups until the value change falls below tol.

    Returns (V, history) where history[k] = ||V_{k+1} - V_k||_inf.
    """
    V = np.zeros(P.shape[0])
    history = []
    for _ in range(max_iter):
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        history.append(delta)
        V = V_new
        if delta < tol:
            break
    return V, history


def greedy_policy(P, R, gamma, V):
    return (R + gamma * P @ V).argmax(axis=1)


def policy_matrices(P, R, policy):
    """Collapse (P, R) under a fixed deterministic policy to (S, S) and (S,)."""
    idx = np.arange(P.shape[0])
    return P[idx, policy], R[idx, policy]


def policy_evaluation_solve(P, R, gamma, policy):
    """Exact V^pi by solving the linear Bellman system.

    For gamma = 1 the full matrix (I - P_pi) is singular because absorbing
    terminals self-loop, so the system is solved on non-terminal states only
    (terminals are pinned at V = 0, which is exact for absorbing states).
    """
    P_pi, R_pi = policy_matrices(P, R, policy)
    live = ~terminal_states(P, R)
    V = np.zeros(P.shape[0])
    A = np.eye(live.sum()) - gamma * P_pi[np.ix_(live, live)]
    V[live] = np.linalg.solve(A, R_pi[live])
    return V


def policy_evaluation_sweeps(P, R, gamma, policy, n_sweeps, V=None):
    """n_sweeps synchronous Bellman-expectation backups (approximate eval)."""
    P_pi, R_pi = policy_matrices(P, R, policy)
    if V is None:
        V = np.zeros(P.shape[0])
    for _ in range(n_sweeps):
        V = R_pi + gamma * P_pi @ V
    return V


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def rollout_stats(env, policy, episodes=10_000, seed=0):
    """Play the greedy policy; return (success_rate, mean_steps_to_goal)."""
    successes, steps_when_success = 0, []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, t = False, 0
        while not done:
            s, r, term, trunc, _ = env.step(int(policy[s]))
            done = term or trunc
            t += 1
        if r > 0:
            successes += 1
            steps_when_success.append(t)
    return successes / episodes, float(np.mean(steps_when_success))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

HOLE_COLOR = "#3b3a37"
GOAL_COLOR = "#1baf7a"
CLIFF_COLOR = "#3b3a37"


def _value_fill(v, vmax):
    """Light blue ramp for values in [0, vmax]."""
    t = 0.0 if vmax <= 0 else np.clip(v / vmax, 0, 1)
    lo = np.array([244, 244, 241])  # near-white surface
    hi = np.array([122, 174, 229])  # calm blue
    rgb = (lo + t * (hi - lo)).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def draw_lake(ax, desc, V=None, policy=None, title=None, show_values=True):
    """Heatmap of V over the lake with greedy-policy arrows on top.

    desc is env.unwrapped.desc (bytes grid of S/F/H/G).
    """
    import matplotlib.patches as mpatches

    n_rows, n_cols = desc.shape
    vmax = V.max() if V is not None else 1.0
    for r in range(n_rows):
        for c in range(n_cols):
            ch = desc[r, c].decode()
            x, y = c, n_rows - 1 - r  # row 0 on top
            s = r * n_cols + c
            if ch == "H":
                face = HOLE_COLOR
            elif ch == "G":
                face = GOAL_COLOR
            else:
                face = _value_fill(V[s], vmax) if V is not None else "#f4f4f1"
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=face,
                                            edgecolor="#fcfcfb", linewidth=1.5))
            if ch == "H":
                continue
            if ch == "G":
                ax.text(x + 0.5, y + 0.5, "+1", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
                continue
            if policy is not None:
                dy = 0.62 if (show_values and V is not None) else 0.5
                ax.text(x + 0.5, y + dy, ARROWS[int(policy[s])], ha="center",
                        va="center", fontsize=12, color=INK)
            if show_values and V is not None:
                ax.text(x + 0.5, y + 0.28, f"{V[s]:.2f}", ha="center",
                        va="center", fontsize=6.5, color=INK_SECONDARY)
            if ch == "S":
                ax.text(x + 0.15, y + 0.82, "S", ha="center", va="center",
                        fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=INK, pad=8)
