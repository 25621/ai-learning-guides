"""Ten Bellman optimality backups on a 3-state MDP, traced number by number.

The MDP is designed so every backup is halving arithmetic you can do in your
head (gamma = 0.5, deterministic transitions, V* = (2, 4, 8) exactly):

    states   s0 -> s1 -> s2          actions: "stay" or "advance"
    stay:    self-loop, reward +0.5
    advance: s0->s1 (r 0), s1->s2 (r 0), s2->s2 (r +4)

Run:  python hand_trace.py        (~2 s on CPU)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "01-build-a-gridworld"))
import plot_style as ps

GAMMA = 0.5
STATES = ["s0", "s1", "s2"]
ACTIONS = ["stay", "advance"]
V_STAR = np.array([2.0, 4.0, 8.0])


def build_mdp():
    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))
    for s in range(3):
        P[s, 0, s] = 1.0                 # stay: self-loop
        R[s, 0] = 0.5
        nxt = min(s + 1, 2)              # advance: move right (s2 self-loops)
        P[s, 1, nxt] = 1.0
        R[s, 1] = 4.0 if s == 2 else 0.0
    return P, R


def main():
    P, R = build_mdp()

    # closed-form check that V* really is (2, 4, 8): V* must be the fixed
    # point of the optimality backup
    Q_star = R + GAMMA * P @ V_STAR
    np.testing.assert_allclose(Q_star.max(axis=1), V_STAR)
    print("verified: V* = (2, 4, 8) is a fixed point of the optimality backup")

    # ----- ten synchronous backups, printed as a hand-trace table --------
    V = np.zeros(3)
    rows = [(0, V.copy(), None)]
    print(f"\n{'k':>2} | {'V(s0)':>7} {'V(s1)':>7} {'V(s2)':>7} | "
          f"{'greedy':>16} | {'‖V_k − V*‖∞':>12}")
    err0 = np.max(np.abs(V - V_STAR))
    print(f"{0:>2} | {V[0]:>7.4f} {V[1]:>7.4f} {V[2]:>7.4f} | "
          f"{'—':>16} | {err0:>12.4f}")
    for k in range(1, 11):
        Q = R + GAMMA * P @ V
        V = Q.max(axis=1)
        greedy = ",".join(ACTIONS[a][0].upper() for a in Q.argmax(axis=1))
        err = np.max(np.abs(V - V_STAR))
        rows.append((k, V.copy(), err))
        print(f"{k:>2} | {V[0]:>7.4f} {V[1]:>7.4f} {V[2]:>7.4f} | "
              f"{greedy:>16} | {err:>12.4f}")

    # contraction check: the error must shrink by at least gamma per backup
    errs = [np.max(np.abs(v - V_STAR)) for _, v, _ in rows]
    ratios = [e2 / e1 for e1, e2 in zip(errs, errs[1:]) if e1 > 0]
    assert all(r <= GAMMA + 1e-12 for r in ratios), ratios
    print(f"\nerror ratio per backup: {min(ratios):.3f} .. {max(ratios):.3f} "
          f"(never above γ = {GAMMA})")

    # ----- figures --------------------------------------------------------
    import matplotlib.pyplot as plt
    ks = np.arange(11)
    traj = np.array([v for _, v, _ in rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    for i, (name, color) in enumerate(zip(STATES, ps.SERIES)):
        axes[0].plot(ks, traj[:, i], color=color, linewidth=2, marker="o",
                     markersize=4, label=name)
        axes[0].axhline(V_STAR[i], color=color, linewidth=1, linestyle="--",
                        alpha=0.55)
        axes[0].text(10.15, V_STAR[i], f"V*({name}) = {V_STAR[i]:g}",
                     color=color, fontsize=8.5, va="center")
    axes[0].set_xlim(0, 12.6)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_title("Each backup pulls V toward the fixed point",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[0].set_xlabel("backup k", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("V_k(s)", color=ps.INK_SECONDARY, fontsize=10)

    axes[1].semilogy(ks, errs, color=ps.SERIES[0], linewidth=2, marker="o",
                     markersize=4)
    axes[1].semilogy(ks, errs[0] * GAMMA ** ks, color=ps.INK_MUTED,
                     linewidth=1, linestyle="--")
    axes[1].text(6.0, errs[0] * GAMMA ** 5.3, "8 · γᵏ (contraction bound)",
                 color=ps.INK_MUTED, fontsize=8.5, rotation=-24)
    axes[1].set_title("The gap to V* halves every backup (γ = 0.5)",
                      color=ps.INK, fontsize=11, loc="left", pad=10)
    axes[1].set_xlabel("backup k", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("‖V_k − V*‖∞ (log scale)", color=ps.INK_SECONDARY,
                       fontsize=10)

    fig.tight_layout()
    fig.savefig("outputs/backups.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print("wrote outputs/backups.png")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
