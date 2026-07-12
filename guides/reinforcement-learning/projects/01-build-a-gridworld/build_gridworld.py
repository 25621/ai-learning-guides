"""Build a 5x5 gridworld MDP, print its (S, A, P, R, gamma) pieces, solve it.

Run:  python build_gridworld.py        (~2 s on CPU)
"""

import numpy as np

import plot_style as ps
from gridworld import (ACTIONS, ARROWS, Gridworld, default_world, draw_grid,
                       value_iteration)


def show_transition(gw, cell, a):
    """Print the full transition distribution and reward for one (s, a)."""
    s = gw.index[cell]
    print(f"\nP(s' | s={cell}, a={ACTIONS[a]}):")
    for s2 in np.nonzero(gw.P[s, a])[0]:
        tag = ""
        if gw.states[s2] == cell:
            tag = "   <- bump (stayed put)"
        if gw.states[s2] in gw.terminals:
            tag = "   <- terminal"
        print(f"   -> {gw.states[s2]}: {gw.P[s, a, s2]:.2f}{tag}")
    print(f"R(s, a) = {gw.R[s, a]:+.3f}")


def main():
    gw = default_world(gamma=0.9)
    start = (4, 0)

    # ----- the five pieces, as concrete objects --------------------------
    print("S: ", gw.n_states, "states (walls are not states); first five:",
          gw.states[:5])
    print("A: ", gw.n_actions, "actions:", ACTIONS)
    print("P:  array of shape", gw.P.shape, "(S, A, S'), rows sum to",
          gw.P.sum(axis=2).min(), "..", gw.P.sum(axis=2).max())
    print("R:  array of shape", gw.R.shape, "(S, A), expected immediate reward")
    print("gamma:", gw.gamma)

    # ----- sanity checks the arrays must pass ----------------------------
    assert np.allclose(gw.P.sum(axis=2), 1.0)
    # bumping into the top wall from the top-left corner keeps you there
    s00 = gw.index[(0, 0)]
    assert gw.P[s00, 0, s00] >= 0.8 + 0.1  # up is blocked, left slip is too
    # terminal states are absorbing with zero reward
    for cell in gw.terminals:
        t = gw.index[cell]
        assert np.allclose(gw.P[t, :, t], 1.0) and np.allclose(gw.R[t], 0.0)
    print("\nsanity checks passed: P is a distribution, bumps stay put, "
          "terminals absorb")

    # ----- a few transitions worth staring at ----------------------------
    show_transition(gw, (0, 0), 0)   # bump into the top-left corner
    show_transition(gw, (0, 3), 1)   # step into the +1 goal
    show_transition(gw, (2, 3), 1)   # step into the -1 pit

    # ----- solve it with value iteration ---------------------------------
    V, Q, pi, iters, _ = value_iteration(gw.P, gw.R, gw.gamma)
    print(f"\nvalue iteration converged in {iters} backups")
    print(f"V*(start {start}) = {V[gw.index[start]]:.3f}")
    print("greedy action at start:", ARROWS[pi[gw.index[start]]])

    # ----- figures --------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    draw_grid(axes[0], gw, start=start, show_values=False,
              title="The world: +1 goal, −1 pit, walls, −0.04 per step")
    draw_grid(axes[1], gw, V=V, policy=pi, start=start,
              title="V* heatmap and the greedy policy (γ = 0.9)")
    fig.tight_layout()
    fig.savefig("outputs/gridworld.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/gridworld.png")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
