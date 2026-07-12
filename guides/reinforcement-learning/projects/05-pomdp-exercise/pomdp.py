"""Break the Markov property on purpose: an agent that sees its row, not its column.

The world is a 3x5 grid with a +1 goal in the center. The underlying MDP is
trivial. But the agent only observes its ROW (three possible observations), so
every cell in a row is aliased: from observation "row 1" the goal may be to
the left or to the right, and no memoryless policy can know which.

We compute exactly, with no learning:
  1. the optimal full-state policy (value iteration on the MDP),
  2. the best DETERMINISTIC memoryless policy (all 4^3 = 64 enumerated),
  3. the best STOCHASTIC memoryless policy (gradient ascent on exact J),
and verify best-memoryless < full-state optimal, while stochastic beats
deterministic — the classic aliased-POMDP ordering.

Run:  python pomdp.py        (~60 s on CPU)
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-build-a-gridworld"))
import plot_style as ps
from gridworld import (ACTIONS, ARROWS, Gridworld, draw_grid,
                       policy_evaluation_solve, value_iteration)

GOAL = (1, 2)
GAMMA = 0.95


def make_world():
    return Gridworld(n_rows=3, n_cols=5, walls=[], terminals={GOAL: +1.0},
                     step_reward=-0.04, slip=0.0, gamma=GAMMA)


def obs_of(cell):
    return cell[0]                      # the agent sees only its row


def expand(gw, pi_obs):
    """Lift an observation policy (n_obs, A) to a state policy (S, A)."""
    pi = np.zeros((gw.n_states, gw.n_actions))
    for s, cell in enumerate(gw.states):
        pi[s] = pi_obs[obs_of(cell)]
    return pi


def J(gw, pi, starts):
    """Exact expected discounted return from the uniform start distribution."""
    V = policy_evaluation_solve(gw.P, gw.R, pi, gw.gamma)
    return V[starts].mean(), V


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def optimize_stochastic(gw, starts, init_logits, iters=250, lr=1.0, eps=1e-5):
    """Gradient ascent on J(theta) with exact numerical gradients."""
    theta = init_logits.copy()

    def value(th):
        return J(gw, expand(gw, softmax(th)), starts)[0]

    best = value(theta)
    for _ in range(iters):
        grad = np.zeros_like(theta)
        for i in np.ndindex(theta.shape):
            up, dn = theta.copy(), theta.copy()
            up[i] += eps
            dn[i] -= eps
            grad[i] = (value(up) - value(dn)) / (2 * eps)
        theta = theta + lr * grad
        best = max(best, value(theta))
    return theta, value(theta)


def main():
    rng = np.random.default_rng(0)
    gw = make_world()
    starts = np.array([s for s, cell in enumerate(gw.states) if cell != GOAL])
    n_obs = gw.n_rows

    # ----- 1. full-state optimum ------------------------------------------
    V_star, _, pi_star, _, _ = value_iteration(gw.P, gw.R, gw.gamma)
    J_star = V_star[starts].mean()
    print(f"full-state optimal:              J = {J_star:+.4f}")

    # ----- 2. best deterministic memoryless policy (exhaustive) -----------
    det_results = []
    for assignment in itertools.product(range(gw.n_actions), repeat=n_obs):
        pi_obs = np.zeros((n_obs, gw.n_actions))
        pi_obs[np.arange(n_obs), assignment] = 1.0
        j, _ = J(gw, expand(gw, pi_obs), starts)
        det_results.append((j, assignment))
    det_results.sort(reverse=True)
    J_det, best_assignment = det_results[0]
    print(f"best deterministic memoryless:   J = {J_det:+.4f}   "
          f"policy = {[ACTIONS[a] for a in best_assignment]} "
          f"(all {len(det_results)} enumerated)")

    # ----- 3. best stochastic memoryless policy ---------------------------
    # one restart warm-started at the best deterministic policy (so the
    # optimizer can only improve on it), plus random restarts
    inits = [np.log(np.full((n_obs, 4), 1e-3))]
    inits[0][np.arange(n_obs), list(best_assignment)] = 0.0
    inits += [rng.normal(0, 1, (n_obs, 4)) for _ in range(5)]
    J_sto, theta_best = -np.inf, None
    for init in inits:
        theta, j = optimize_stochastic(gw, starts, init)
        if j > J_sto:
            J_sto, theta_best = j, theta
    pi_sto = softmax(theta_best)
    print(f"best stochastic memoryless:      J = {J_sto:+.4f}")
    for o in range(n_obs):
        probs = ", ".join(f"{ACTIONS[a]} {pi_sto[o, a]:.2f}"
                          for a in range(4) if pi_sto[o, a] > 0.01)
        print(f"   row {o}: {probs}")

    # baseline: uniform random over actions
    J_unif, _ = J(gw, np.full((gw.n_states, 4), 0.25), starts)
    print(f"uniform random baseline:         J = {J_unif:+.4f}")

    # ----- the ordering the stub promises ---------------------------------
    assert J_det < J_star - 0.05, "memoryless should be clearly suboptimal"
    assert J_sto > J_det + 0.01, "stochastic should beat deterministic"
    print("\nverified: J(det memoryless) < J(stochastic memoryless) < J(full state)")

    # which start cells does the best deterministic policy strand forever?
    pi_obs_det = np.zeros((n_obs, 4))
    pi_obs_det[np.arange(n_obs), list(best_assignment)] = 1.0
    _, V_det = J(gw, expand(gw, pi_obs_det), starts)
    stranded = [gw.states[s] for s in starts if V_det[s] < -0.5]
    print(f"cells the deterministic policy strands (V ≈ -0.04/(1-γ)): {stranded}")

    # ----- figures ---------------------------------------------------------
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    draw_grid(axes[0], gw, V=V_star, policy=pi_star,
              title="What the MDP solver sees: V* and π* per cell")

    # right panel: the aliased view - rows shaded, one arrow per row
    ax = axes[1]
    row_fill = ["#dce9f9", "#c2d9f4", "#a4c6ee"]
    for r in range(gw.n_rows):
        for c in range(gw.n_cols):
            x, y = c, gw.n_rows - 1 - r
            cell = (r, c)
            face = "#1baf7a" if cell == GOAL else row_fill[r]
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=face,
                                            edgecolor="#fcfcfb", linewidth=2))
            if cell == GOAL:
                ax.text(x + 0.5, y + 0.5, "+1", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
            elif V_det[gw.index[cell]] < -0.5:
                ax.text(x + 0.5, y + 0.32, "✗", ha="center", va="center",
                        fontsize=10, color="#e34948", fontweight="bold")
        a = int(best_assignment[r])
        ax.text(-0.45, gw.n_rows - 1 - r + 0.5,
                f"sees “row {r}”\nacts {ARROWS[a]}", ha="center", va="center",
                fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xlim(-1.05, gw.n_cols)
    ax.set_ylim(0, gw.n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("What the agent sees: one action per row "
                 "(✗ = stranded forever)", fontsize=11, loc="left",
                 color=ps.INK, pad=8)
    fig.tight_layout()
    fig.savefig("outputs/aliasing.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/aliasing.png")

    # returns bar chart
    fig, ax = ps.new_axes(7.2, 3.4)
    names = ["full-state optimal", "stochastic memoryless",
             "deterministic memoryless", "uniform random"]
    vals = [J_star, J_sto, J_det, J_unif]
    colors = [ps.SERIES[0], ps.SERIES[1], ps.SERIES[3], ps.BASELINE]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + (0.012 if v >= 0 else -0.012), yi, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9,
                color=ps.INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=ps.INK)
    ax.grid(axis="y", visible=False)
    ps.finish(fig, ax, "Seeing less is worth less: expected return from a uniform start",
              "expected discounted return J(π)", "",
              "outputs/returns.png")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
