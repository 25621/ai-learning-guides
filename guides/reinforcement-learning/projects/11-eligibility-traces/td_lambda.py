"""TD(lambda) with eligibility traces on the 19-state random walk.

The environment (Sutton & Barto's long random walk): states 1..19, start in
the middle, each step moves left or right with probability 1/2, terminating
with reward -1 at the left end and +1 at the right end. Its true value
function is exactly linear, V(i) = i/10 - 1, so prediction error is
measurable without any reference implementation.

TD(lambda), online, with traces:
    delta  = r + gamma * V(s') - V(s)
    e(s)  <- 1                (replacing)   or   e(s) + 1   (accumulating)
    V     <- V + alpha * delta * e          (every state at once)
    e     <- gamma * lambda * e

lambda = 0 is TD(0); lambda = 1 is (online) Monte Carlo. The sweep over
(lambda, alpha) reproduces the book's classic U-curves.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

from plot_style import SERIES, finish, new_axes  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

N = 19                       # non-terminal states 1..19; terminals 0 and 20
START = 10
TRUE_V = np.arange(1, N + 1) / 10.0 - 1.0
GAMMA = 1.0
LAMBDAS = [0.0, 0.5, 0.9, 1.0]
ALPHAS = np.linspace(0.05, 1.0, 20)
RUNS = 100
EPISODES = 10                # error averaged over the first 10 episodes


def walk(rng):
    """One episode: list of states visited (1..19) and the final reward."""
    s, states = START, [START]
    while True:
        s = s + (1 if rng.random() < 0.5 else -1)
        if s == 0:
            return states, -1.0
        if s == N + 1:
            return states, +1.0
        states.append(s)


def td_lambda_episode(V, states, final_r, lam, alpha, replacing):
    e = np.zeros(N)
    for t, s in enumerate(states):
        i = s - 1
        nxt = states[t + 1] - 1 if t + 1 < len(states) else None
        delta = (final_r if nxt is None else GAMMA * V[nxt]) - V[i]
        if replacing:
            e[i] = 1.0
        else:
            e[i] += 1.0
        V += alpha * delta * e
        e *= GAMMA * lam


def rms(V):
    return float(np.sqrt(np.mean((V - TRUE_V) ** 2)))


def sweep(lam, alpha, replacing=True, runs=RUNS):
    """Mean RMS error over the first EPISODES episodes, averaged over runs."""
    total = 0.0
    for run in range(runs):
        rng = np.random.default_rng(10_000 * run + 17)
        V = np.zeros(N)
        for _ in range(EPISODES):
            states, final_r = walk(rng)
            td_lambda_episode(V, states, final_r, lam, alpha, replacing)
            total += rms(V)
    return total / (runs * EPISODES)


def main():
    # --- the classic sweep: RMS error vs alpha, one curve per lambda --------
    print("=== replacing traces: RMS error over first 10 episodes ===")
    print("lambda | best alpha | RMS at best alpha")
    fig, ax = new_axes(7.6, 4.4)
    label_at = {0.0: (12, 5, 8), 0.5: (12, 5, -14), 0.9: (5, -2, -15),
                1.0: (9, 6, 6)}  # lambda -> (alpha index, dx, dy)
    for color, lam in zip(SERIES, LAMBDAS):
        errs = np.array([sweep(lam, a) for a in ALPHAS])
        best = errs.argmin()
        print(f"  {lam:3.1f}  |   {ALPHAS[best]:.2f}     | {errs[best]:.3f}")
        ax.plot(ALPHAS, errs, color=color, linewidth=1.8, marker="o",
                markersize=3)
        k, dx, dy = label_at[lam]
        ax.annotate(f"λ = {lam}", (ALPHAS[k], errs[k]), color=color,
                    fontsize=9.5, xytext=(dx, dy),
                    textcoords="offset points")
    ax.set_ylim(0.1, 0.65)
    finish(fig, ax,
           "19-state random walk: error vs step size, per lambda "
           f"(replacing traces, {RUNS} runs)",
           "step size alpha", "RMS error, first 10 episodes",
           OUT / "lambda_sweep.png")

    # --- replacing vs accumulating at lambda = 0.9 --------------------------
    print("\n=== replacing vs accumulating traces at lambda = 0.9 ===")
    fig, ax = new_axes(7.2, 4.2)
    for color, (replacing, label) in zip((SERIES[0], SERIES[2]),
                                         [(True, "replacing traces"),
                                          (False, "accumulating traces")]):
        errs = np.array([sweep(0.9, a, replacing=replacing) for a in ALPHAS])
        show = np.minimum(errs, 5)  # divergence otherwise dwarfs the plot
        ax.plot(ALPHAS, show, color=color, linewidth=1.8, marker="o",
                markersize=3)
        ax.annotate(label, (ALPHAS[10], show[10]), color=color, fontsize=9.5,
                    xytext=(6, 6 if replacing else -12),
                    textcoords="offset points")
        worst = errs.max()
        print(f"{label:20s}: RMS at alpha=1.0 is {errs[-1]:8.2f} "
              f"(worst over the sweep {worst:8.2f})")
    ax.set_yscale("log")
    finish(fig, ax,
           "Why traces are capped: a revisited state must not double its credit",
           "step size alpha", "RMS error (log scale, clipped at 5)",
           OUT / "replacing_vs_accumulating.png")

    # --- learning curves at each lambda's best alpha ------------------------
    print("\n=== learning curves (RMS after each episode, best alpha) ===")
    best_alpha = {}
    for lam in LAMBDAS:
        errs = [sweep(lam, a) for a in ALPHAS]
        best_alpha[lam] = float(ALPHAS[int(np.argmin(errs))])
    fig, ax = new_axes(7.2, 4.2)
    n_show = 50
    for color, lam in zip(SERIES, LAMBDAS):
        curve = np.zeros(n_show)
        for run in range(RUNS):
            rng = np.random.default_rng(999 * run + 3)
            V = np.zeros(N)
            for ep in range(n_show):
                states, final_r = walk(rng)
                td_lambda_episode(V, states, final_r, lam, best_alpha[lam],
                                  replacing=True)
                curve[ep] += rms(V) / RUNS
        ax.plot(np.arange(1, n_show + 1), curve, color=color, linewidth=1.8)
        ax.annotate(f"λ = {lam} (α = {best_alpha[lam]:.2f})",
                    (n_show * 0.7, curve[int(n_show * 0.7)]), color=color,
                    fontsize=9, xytext=(4, 5), textcoords="offset points")
        print(f"  lambda={lam:3.1f}, alpha={best_alpha[lam]:.2f}: "
              f"RMS after 50 episodes = {curve[-1]:.3f}")
    finish(fig, ax, "Each lambda at its own best step size",
           "episode", "RMS error", OUT / "learning_curves.png")


if __name__ == "__main__":
    main()
