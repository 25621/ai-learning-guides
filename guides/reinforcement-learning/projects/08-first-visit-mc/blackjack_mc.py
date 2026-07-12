"""First-visit Monte Carlo evaluation of a fixed Blackjack policy,
checked against the exact analytic answer.

Policy (Sutton & Barto section 5.1): stick on 20 or 21, otherwise hit.

Blackjack-v1 draws cards with replacement (an infinite deck), which makes the
true V^pi computable by plain dynamic programming over at most a few hundred
(sum, usable-ace) configurations — so the sampled estimate can be graded
against the truth, state by state.

The env is created with sab=False, natural=False (payoff = the pure
win/lose/draw comparison). This is deliberate: the v1 DEFAULT is sab=True,
whose natural-blackjack autowin pays +1 against a dealer 21 only when the
player's 21 came from the first two cards — information the observed state
(sum, upcard, usable) does not contain. Under that rule the observed state
(21, up, usable) is not Markov: its value depends on how you got there, and
no state-value function can be "correct" for it. sab_demo() below measures
the discrepancy; with sab=False every observed state is Markov and the MC
estimate must converge to the DP answer.
"""

import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from plot_style import (INK, INK_SECONDARY, SERIES, SURFACE, finish,  # noqa: E402
                        new_axes)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# card values dealt by the env: 1-9 with prob 1/13 each, 10 with prob 4/13
CARD_P = {c: 1 / 13 for c in range(1, 10)}
CARD_P[10] = 4 / 13


def stick_on_20(ps):
    return ps >= 20


# ---------------------------------------------------------------------------
# Exact V^pi by dynamic programming (infinite deck => small closed recursion)
# ---------------------------------------------------------------------------
# A hand is (raw_sum, has_usable_ace). The observed player sum is raw + 10
# when the ace is usable. Drawing card c: the raw sum grows by c, and an ace
# stays/becomes usable only while raw + 10 <= 21.

def _draw(raw, usable, c):
    raw2 = raw + c
    usable2 = (usable or c == 1) and raw2 + 10 <= 21
    return raw2, usable2


def _effective(raw, usable):
    return raw + 10 if usable else raw


@lru_cache(maxsize=None)
def dealer_final_dist(raw, usable):
    """Distribution over dealer's final effective sum (22 = bust)."""
    eff = _effective(raw, usable)
    if eff >= 17:
        return {min(eff, 22): 1.0}
    dist = {}
    for c, p in CARD_P.items():
        for final, q in dealer_final_dist(*_draw(raw, usable, c)).items():
            dist[final] = dist.get(final, 0.0) + p * q
    return dist


@lru_cache(maxsize=None)
def dealer_dist_from_upcard(upcard):
    """Dealer holds the upcard plus one hidden card, then plays to 17+."""
    dist = {}
    for c, p in CARD_P.items():
        raw, usable = _draw(*_draw(0, False, upcard), c)
        for final, q in dealer_final_dist(raw, usable).items():
            dist[final] = dist.get(final, 0.0) + p * q
    return dist


def stick_value(ps, upcard):
    v = 0.0
    for final, q in dealer_dist_from_upcard(upcard).items():
        v += q * (1.0 if final == 22 or ps > final else
                  (-1.0 if ps < final else 0.0))
    return v


@lru_cache(maxsize=None)
def exact_v(ps, upcard, usable):
    """V^pi for the observed state, for the stick-on-20 policy."""
    if stick_on_20(ps):
        return stick_value(ps, upcard)
    raw = ps - 10 if usable else ps
    v = 0.0
    for c, p in CARD_P.items():
        raw2, usable2 = _draw(raw, usable, c)
        ps2 = _effective(raw2, usable2)
        v += p * (-1.0 if ps2 > 21 else exact_v(ps2, upcard, usable2))
    return v


# ---------------------------------------------------------------------------
# First-visit Monte Carlo on the actual env
# ---------------------------------------------------------------------------

PLOT_STATES = [(ps, up, ua) for ps in range(12, 22) for up in range(1, 11)
               for ua in (False, True)]


def run_mc(episodes, checkpoints, seed=0):
    """Returns {n: V-estimate dict} at each checkpoint + visit stats."""
    env = gym.make("Blackjack-v1", sab=False, natural=False)
    returns_sum, returns_sq, returns_n = {}, {}, {}
    snapshots = {}
    repeats = 0
    rng_marks = set(checkpoints)
    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=seed + ep)
        visited = []
        seen = set()
        done = False
        while not done:
            if s not in seen:            # first visit only
                visited.append(s)
                seen.add(s)
            else:
                repeats += 1
            a = 0 if stick_on_20(s[0]) else 1          # 0=stick, 1=hit
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
        for st in visited:               # gamma=1, single terminal reward:
            returns_sum[st] = returns_sum.get(st, 0.0) + r
            returns_sq[st] = returns_sq.get(st, 0.0) + r * r
            returns_n[st] = returns_n.get(st, 0) + 1
        if ep in rng_marks:
            snapshots[ep] = {st: returns_sum[st] / returns_n[st]
                             for st in returns_sum}
    stderr = {st: np.sqrt(max(returns_sq[st] / n - (returns_sum[st] / n) ** 2, 0)
                          / n) for st, n in returns_n.items()}
    return snapshots, returns_n, stderr, repeats


def sab_demo(episodes=200_000, seed=1):
    """Show that the sab=True default makes (21, up, usable) non-Markov.

    Split the returns that follow a visit to any (21, up, usable=True) state
    by how the state was reached: dealt as a natural vs hit into. Under
    sab=True the two conditional means differ by a wide margin (the autowin
    only fires for naturals); under sab=False they agree.
    """
    print("\nwhy the env's DEFAULT rules were switched off "
          "(returns from (21, up, usable)):")
    for sab in (True, False):
        env = gym.make("Blackjack-v1", sab=sab, natural=False)
        env.reset(seed=seed)
        buckets = {"dealt (natural)": [], "hit into": []}
        for _ in range(episodes):
            s, _ = env.reset()
            how = "dealt (natural)" if s[0] == 21 and s[2] else None
            done = False
            while not done:
                if how is None and s[0] == 21 and s[2]:
                    how = "hit into"
                a = 0 if stick_on_20(s[0]) else 1
                s, r, term, trunc, _ = env.step(a)
                done = term or trunc
            if how:
                buckets[how].append(r)
        means = {k: np.mean(v) for k, v in buckets.items()}
        print(f"  sab={sab!s:5}: " + "   ".join(
            f"{k}: {means[k]:+.3f} (n={len(buckets[k]):,})" for k in buckets))
    print("  -> same observed state, different value depending on the path:"
          " under sab=True that state is not Markov.")


def rms_error(V_est, states):
    errs = [(V_est.get((ps, up, bool(ua)), 0.0) - exact_v(ps, up, ua)) ** 2
            for ps, up, ua in states]
    return float(np.sqrt(np.mean(errs)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

DIVERGING = LinearSegmentedColormap.from_list(
    "vpi", ["#e34948", "#f6e4e3", "#fcfcfb", "#dbe7f5", "#2a78d6"])


def value_grid(V, usable):
    """(10, 10) array: rows = player sum 12..21, cols = dealer upcard A..10."""
    g = np.full((10, 10), np.nan)
    for i, ps in enumerate(range(12, 22)):
        for j, up in enumerate(range(1, 11)):
            v = V((ps, up, usable)) if callable(V) else V.get((ps, up, usable))
            g[i, j] = np.nan if v is None else v
    return g


def heatmap_figure(snapshots, out_path):
    cols = [("MC, 10k episodes", lambda st: snapshots[10_000].get(st)),
            ("MC, 500k episodes", lambda st: snapshots[500_000].get(st)),
            ("exact (DP over the deck)", lambda st: exact_v(*st))]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for row, usable in enumerate((True, False)):
        for col, (label, V) in enumerate(cols):
            ax = axes[row][col]
            g = value_grid(V, usable)
            im = ax.imshow(g, origin="lower", cmap=DIVERGING, vmin=-1, vmax=1,
                           extent=(0.5, 10.5, 11.5, 21.5), aspect="auto")
            ax.set_xticks([1, 4, 7, 10])
            ax.set_xticklabels(["A", "4", "7", "10"], fontsize=8,
                               color=INK_SECONDARY)
            ax.set_yticks([12, 15, 18, 21])
            ax.tick_params(colors=INK_SECONDARY, labelsize=8)
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel("usable ace" if usable else "no usable ace",
                              fontsize=9, color=INK_SECONDARY)
            if row == 1:
                ax.set_xlabel("dealer shows", fontsize=9, color=INK_SECONDARY)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    cbar.set_label("V^pi  (expected final reward)", fontsize=9,
                   color=INK_SECONDARY)
    fig.suptitle("Stick-on-20 policy: sampled values converge to the exact ones",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    episodes = 500_000
    checkpoints = sorted({int(n) for n in np.logspace(3, np.log10(episodes), 14)}
                         | {10_000, episodes})
    snapshots, counts, stderr, repeats = run_mc(episodes, checkpoints)

    print(f"episodes: {episodes:,}   distinct states visited: {len(counts)}")
    print(f"in-episode state repeats observed: {repeats} "
          "(a Blackjack state can never recur within an episode, so "
          "first-visit and every-visit MC coincide here)")

    # convergence against the exact values
    print("\nRMS error vs the exact V^pi (over the 200 plotted states):")
    errs = [(n, rms_error(snapshots[n], PLOT_STATES)) for n in checkpoints]
    for n, e in errs:
        if n in (1_000, 10_000, 100_000, episodes):
            print(f"  {n:>9,} episodes: {e:.4f}")

    # spot-check a few states
    print("\nspot checks (state: MC@500k +/- 1.96 SE vs exact):")
    for st in [(20, 10, False), (16, 10, False), (21, 1, True), (13, 2, True)]:
        mc = snapshots[episodes].get((st[0], st[1], st[2]))
        print(f"  sum={st[0]:2d} dealer={st[1]:2d} usable={st[2]!s:5}: "
              f"{mc:+.4f} +/- {1.96 * stderr[st]:.4f} vs {exact_v(*st):+.4f}  "
              f"(n={counts.get(st, 0):,} first visits)")

    sab_demo()

    heatmap_figure(snapshots, OUT / "blackjack_value.png")

    fig, ax = new_axes(7.2, 4.2)
    ns = np.array([n for n, _ in errs], float)
    es = np.array([e for _, e in errs])
    ax.plot(ns, es, color=SERIES[0], linewidth=1.8, marker="o", markersize=3.5)
    ref = es[3] * np.sqrt(ns[3] / ns)
    ax.plot(ns, ref, color=SERIES[2], linewidth=1.4, linestyle="--")
    ax.annotate("measured RMS error", (ns[6], es[6]), color=SERIES[0],
                fontsize=9, xytext=(8, 8), textcoords="offset points")
    ax.annotate("1/sqrt(N) reference", (ns[1], ref[1]), color=SERIES[2],
                fontsize=9, xytext=(10, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    finish(fig, ax, "Monte Carlo error shrinks like one over sqrt(episodes)",
           "episodes", "RMS error vs exact V^pi", OUT / "mc_error.png")


if __name__ == "__main__":
    main()
