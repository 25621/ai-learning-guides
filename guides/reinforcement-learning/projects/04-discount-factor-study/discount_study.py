"""Sweep the discount factor on one fixed gridworld and watch "optimal" change.

The world has a small +1 goal two steps from the start and a big +10 goal
eight steps away. Nothing about the world changes across runs — only gamma —
yet the optimal policy flips from grabbing the +1 to walking the long way to
the +10 once the effective horizon 1/(1-gamma) covers the distance.

Run:  python discount_study.py        (~10 s on CPU)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-build-a-gridworld"))
import plot_style as ps
from gridworld import Gridworld, draw_grid, policy_matrices, value_iteration

START = (4, 0)
NEAR, FAR = (4, 2), (0, 4)
SWEEP_GAMMAS = [0.5, 0.9, 0.99, 0.999]


def make_world(gamma):
    """Same task every time; only gamma varies."""
    return Gridworld(
        n_rows=5, n_cols=5,
        walls=[(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)],
        terminals={NEAR: +1.0, FAR: +10.0, (2, 2): -1.0},
        step_reward=-0.04, slip=0.2, gamma=gamma,
    )


def absorption_prob(gw, pi, target):
    """P(the greedy policy's random walk ends in `target`), start-by-start.

    For an absorbing chain, p(s) = sum_s' P_pi[s, s'] p(s') with p = 1 at the
    target terminal and p = 0 at the others - a linear system.
    """
    pi_onehot = np.zeros((gw.n_states, gw.n_actions))
    pi_onehot[np.arange(gw.n_states), pi] = 1.0
    P_pi, _ = policy_matrices(gw.P, gw.R, pi_onehot)
    term_idx = [gw.index[c] for c in gw.terminals]
    free = [s for s in range(gw.n_states) if s not in term_idx]
    b = P_pi[np.ix_(free, [gw.index[target]])].sum(axis=1)
    A = np.eye(len(free)) - P_pi[np.ix_(free, free)]
    p_free = np.linalg.solve(A, b)
    p = np.zeros(gw.n_states)
    p[free] = p_free
    p[gw.index[target]] = 1.0
    return p


def solve(gamma):
    gw = make_world(gamma)
    V, Q, pi, iters, _ = value_iteration(gw.P, gw.R, gamma, tol=1e-10)
    p_far = absorption_prob(gw, pi, FAR)[gw.index[START]]
    return gw, V, pi, iters, p_far


def main():
    # ----- the four headline gammas ---------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    print(f"{'gamma':>7} {'horizon':>9} {'V*(start)':>10} {'P(reach +10)':>13} "
          f"{'VI iters':>9}")
    for ax, gamma in zip(axes, SWEEP_GAMMAS):
        gw, V, pi, iters, p_far = solve(gamma)
        horizon = 1.0 / (1.0 - gamma)
        print(f"{gamma:>7} {horizon:>9.0f} {V[gw.index[START]]:>10.3f} "
              f"{p_far:>13.3f} {iters:>9}")
        draw_grid(ax, gw, V=V, policy=pi, start=START,
                  title=f"γ = {gamma}  (horizon ≈ {horizon:.0f} steps)")
    fig.tight_layout()
    fig.savefig("outputs/policies.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/policies.png")

    # ----- fine sweep: where exactly does optimal flip? -------------------
    gammas = 1.0 - np.logspace(-3, np.log10(0.5), 28)
    gammas = np.sort(gammas)
    p_fars, v_starts, iter_counts = [], [], []
    for gamma in gammas:
        gw, V, pi, iters, p_far = solve(gamma)
        p_fars.append(p_far)
        v_starts.append(V[gw.index[START]])
        iter_counts.append(iters)
    p_fars = np.array(p_fars)

    # measured flip point: first gamma whose greedy policy heads for the +10
    flip = gammas[np.argmax(p_fars > 0.5)]
    print(f"measured flip: the greedy policy switches to the +10 at "
          f"γ ≈ {flip:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    horizons = 1.0 / (1.0 - gammas)
    axes[0].semilogx(horizons, p_fars, color=ps.SERIES[0], linewidth=2,
                     marker="o", markersize=3.5)
    axes[0].axvline(1.0 / (1.0 - flip), color=ps.INK_MUTED, linewidth=1,
                    linestyle="--")
    axes[0].text(1.0 / (1.0 - flip) * 1.12, 0.45,
                 f"flip at γ ≈ {flip:.2f}", fontsize=8.5,
                 color=ps.INK_SECONDARY)
    axes[0].set_title("The destination flips: P(reach +10 | start, greedy π)",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[0].set_xlabel("effective horizon 1/(1−γ), log scale",
                       color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("absorption probability at +10",
                       color=ps.INK_SECONDARY, fontsize=10)

    axes[1].semilogx(horizons, v_starts, color=ps.SERIES[1], linewidth=2,
                     marker="o", markersize=3.5)
    axes[1].axhline(10.0, color=ps.INK_MUTED, linewidth=1, linestyle="--")
    axes[1].text(2.2, 10.25, "the +10 itself", fontsize=8.5,
                 color=ps.INK_MUTED)
    for gamma in SWEEP_GAMMAS:
        h = 1.0 / (1.0 - gamma)
        i = int(np.argmin(np.abs(horizons - h)))
        axes[1].annotate(f"γ={gamma}", (h, v_starts[i]),
                         textcoords="offset points", xytext=(4, -12),
                         fontsize=8, color=ps.INK_SECONDARY)
    axes[1].set_ylim(-0.5, 11)
    axes[1].set_title("Patience raises the ceiling: V*(start) vs horizon",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("effective horizon 1/(1−γ), log scale",
                       color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("V*(start)", color=ps.INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig("outputs/flip_and_cost.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/flip_and_cost.png")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
