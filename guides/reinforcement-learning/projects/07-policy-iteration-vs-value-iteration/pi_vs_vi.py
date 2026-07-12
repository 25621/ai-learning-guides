"""Policy iteration vs value iteration on FrozenLake 8x8 (slippery).

Both are instances of generalized policy iteration; they differ only in how
much *evaluation* happens between *improvements*:

  value iteration          -> 1 backup sweep of evaluation, then improve
  modified policy iteration -> m sweeps of evaluation, then improve
  policy iteration          -> exact evaluation (linear solve), then improve

To compare costs fairly we count expected-value backups: one backup =
computing r + gamma * sum_s' P(s'|s,a) V(s') for one (s, a) pair. A value-
iteration sweep or a policy improvement costs S*A backups; one sweep of
fixed-policy evaluation costs S (one action per state); an exact linear
solve is priced as its Gaussian-elimination equivalent, ~S^2/3 backup-sized
operations per solve (a backup and a row operation are both O(S) dot
products).

gamma = 0.99 (not 1.0) so the linear system for *any* policy — including the
terrible initial one that may never reach a terminal — stays nonsingular.
"""

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "06-value-iteration-on-frozenlake"))

from frozenlake_lib import (greedy_policy, mdp_arrays,  # noqa: E402
                            policy_evaluation_solve, policy_evaluation_sweeps,
                            value_iteration)
from plot_style import SERIES, finish, new_axes  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

GAMMA = 0.99
TOL = 1e-10


def run_value_iteration(P, R, V_star):
    """Returns (V, policy, errors-per-improvement, backups-per-improvement)."""
    S, A, _ = P.shape
    V = np.zeros(S)
    errors, backups = [np.abs(V - V_star).max()], [0]
    total = 0
    while True:
        V_new = (R + GAMMA * P @ V).max(axis=1)
        total += S * A
        delta = np.abs(V_new - V).max()
        V = V_new
        errors.append(np.abs(V - V_star).max())
        backups.append(total)
        if delta < TOL:
            break
    return V, greedy_policy(P, R, GAMMA, V), errors, backups


def run_policy_iteration(P, R, V_star, m=None):
    """m = None -> exact evaluation (classic PI); else m evaluation sweeps."""
    S, A, _ = P.shape
    policy = np.zeros(S, dtype=int)
    V = np.zeros(S)
    errors, backups = [np.abs(V - V_star).max()], [0]
    total = 0
    n_improvements = 0
    while True:
        if m is None:
            V = policy_evaluation_solve(P, R, GAMMA, policy)
            total += S * S // 3            # linear-solve price
        else:
            V = policy_evaluation_sweeps(P, R, GAMMA, policy, m, V)
            total += m * S                 # m cheap fixed-policy sweeps
        Q = R + GAMMA * P @ V
        total += S * A                     # the improvement step
        # keep the incumbent action on ties so the policy can stabilize
        new_policy = np.where(Q.max(axis=1) - Q[np.arange(S), policy] > 1e-12,
                              Q.argmax(axis=1), policy)
        n_improvements += 1
        errors.append(np.abs(V - V_star).max())
        backups.append(total)
        if np.array_equal(new_policy, policy) and (
                m is None or np.abs((Q.max(axis=1) - V)).max() < TOL):
            break
        policy = new_policy
    return V, policy, errors, backups, n_improvements


def main():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    P, R = mdp_arrays(env)
    S, A, _ = P.shape

    # ground truth to measure error against
    V_star, _ = value_iteration(P, R, GAMMA, tol=1e-14)

    t0 = time.perf_counter()
    V_vi, pi_vi, err_vi, bk_vi = run_value_iteration(P, R, V_star)
    t_vi = time.perf_counter() - t0

    t0 = time.perf_counter()
    V_pi, pi_pi, err_pi, bk_pi, n_impr = run_policy_iteration(P, R, V_star)
    t_pi = time.perf_counter() - t0

    print(f"=== FrozenLake 8x8, gamma={GAMMA} ===")
    print(f"value iteration : {len(err_vi)-1:4d} sweeps        "
          f"{bk_vi[-1]:>9,} backups  {t_vi*1e3:7.1f} ms")
    print(f"policy iteration: {n_impr:4d} improvements  "
          f"{bk_pi[-1]:>9,} backups  {t_pi*1e3:7.1f} ms")

    # same answer?
    print(f"\nmax |V_PI - V_VI|          = {np.abs(V_pi - V_vi).max():.2e}")
    same = (pi_pi == pi_vi).sum()
    print(f"greedy policies agree on   {same}/{S} states")
    if same < S:
        Q = R + GAMMA * P @ V_star
        gaps = [Q[s].max() - Q[s, a] for s in np.nonzero(pi_pi != pi_vi)[0]
                for a in (pi_pi[s], pi_vi[s])]
        print(f"...and where they differ the Q-gap is < {max(gaps):.1e} (ties)")

    # the GPI spectrum: m sweeps of evaluation per improvement
    curves = {"VI (m = 1)": (err_vi, bk_vi)}
    for m in (5, 20):
        _, _, err, bk, n = run_policy_iteration(P, R, V_star, m=m)
        curves[f"MPI (m = {m})"] = (err, bk)
        print(f"modified PI m={m:2d}: {n} improvements, {bk[-1]:,} backups")
    curves["PI (exact eval)"] = (err_pi, bk_pi)

    fig, ax = new_axes(7.6, 4.4)
    for color, (label, (err, bk)) in zip(SERIES, curves.items()):
        err = np.maximum(err, 1e-14)
        ax.plot(np.array(bk) / 1000, err, color=color, linewidth=1.8,
                marker="o" if len(err) < 30 else None, markersize=3.5)
        frac = 0.72 if label.startswith("MPI (m = 5") else 0.5
        k = int(len(err) * frac)
        ax.annotate(label, (bk[k] / 1000, err[k]), color=color, fontsize=9,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_yscale("log")
    finish(fig, ax,
           "One dial, one trade-off: evaluation depth per improvement",
           "expected-value backups (thousands)", "‖V − V*‖∞",
           OUT / "gpi_spectrum.png")


if __name__ == "__main__":
    main()
