"""Policy evaluation two ways: one linear solve vs many Bellman backups.

For a fixed (uniform random) policy on the 5x5 gridworld from project 01,
compute V^pi exactly as V = (I - gamma P_pi)^-1 r_pi, then watch iterative
policy evaluation converge to that same vector at geometric rate gamma.

Run:  python matrix_inverse_eval.py        (~5 s on CPU)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-build-a-gridworld"))
import plot_style as ps
from gridworld import (default_world, draw_grid, policy_evaluation_iterative,
                       policy_evaluation_solve)


def main():
    gammas = [0.5, 0.9, 0.99]
    tol = 1e-9

    fig, ax = ps.new_axes(7.2, 4.4)
    summary = []
    # warm up LAPACK so the first timed solve isn't paying import costs
    _ = np.linalg.solve(np.eye(4), np.ones(4))
    for gamma, color in zip(gammas, ps.SERIES):
        gw = default_world(gamma=gamma)
        pi = np.full((gw.n_states, gw.n_actions), 1.0 / gw.n_actions)

        t0 = time.perf_counter()
        V_closed = policy_evaluation_solve(gw.P, gw.R, pi, gamma)
        t_solve = time.perf_counter() - t0

        t0 = time.perf_counter()
        V_iter, iters, history = policy_evaluation_iterative(
            gw.P, gw.R, pi, gamma, tol=tol)
        t_iter = time.perf_counter() - t0

        np.testing.assert_allclose(V_iter, V_closed, atol=1e-7)
        err = [np.max(np.abs(Vk - V_closed)) for Vk in history]
        summary.append((gamma, iters, t_solve, t_iter, err))

        ks = np.arange(len(err))
        ax.semilogy(ks, err, color=color, linewidth=2, label=f"γ = {gamma}")
        # theoretical contraction rate: error shrinks by at least gamma / backup
        ref = err[0] * gamma ** ks
        ax.semilogy(ks, ref, color=color, linewidth=1, linestyle="--",
                    alpha=0.6)

    ax.set_ylim(1e-10, 1e2)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.98, 0.95, "dashed: err₀ · γᵏ (the contraction bound)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=ps.INK_MUTED)
    ps.finish(fig, ax,
              "Iterative policy evaluation converges to the closed form at rate γ",
              "Bellman backups k", "‖V_k − V_closed‖∞ (log scale)",
              "outputs/convergence.png")

    print(f"{'gamma':>7} {'iters to 1e-9':>14} {'solve (ms)':>11} "
          f"{'iterate (ms)':>13}")
    for gamma, iters, t_solve, t_iter, err in summary:
        print(f"{gamma:>7} {iters:>14} {t_solve * 1e3:>11.3f} "
              f"{t_iter * 1e3:>13.3f}")
        # predicted iteration count: err_0 * gamma^k < tol
        k_pred = int(np.ceil(np.log(tol / err[1]) / np.log(gamma))) + 1
        print(f"        predicted from γ-contraction: ~{k_pred} iterations")

    # ----- the value function itself, both ways --------------------------
    gw = default_world(gamma=0.9)
    pi = np.full((gw.n_states, gw.n_actions), 1.0 / gw.n_actions)
    V_closed = policy_evaluation_solve(gw.P, gw.R, pi, 0.9)
    V_iter, _, _ = policy_evaluation_iterative(gw.P, gw.R, pi, 0.9, tol=tol)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    draw_grid(axes[0], gw, V=V_closed,
              title="V^π by matrix inverse (γ = 0.9, random policy)")
    draw_grid(axes[1], gw, V=V_iter,
              title=f"V^π by iteration (max difference {np.max(np.abs(V_iter - V_closed)):.1e})")
    fig.tight_layout()
    fig.savefig("outputs/v_random_policy.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/v_random_policy.png")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
