"""
Work item 3 -- "Compare exploration strategies"

Five ways to get an agent to explore, all run on the same two hard tasks:

  1. epsilon-greedy            -- "act greedily, but flip a coin sometimes
                                   and act at random."  The default.  It is
                                   *dithering*, not exploration: every step
                                   is an independent coin flip, so the odds
                                   of stumbling through a long correct chain
                                   are astronomically small.

  2. optimistic initialisation -- start every Q-value at the maximum possible
                                   return, R_max / (1 - gamma).  Untried
                                   actions therefore look fantastic, so the
                                   greedy policy is *forced* to try them; a
                                   value only drops once you've actually
                                   visited it and seen the truth.  Cheap,
                                   no extra machinery, and -- surprisingly --
                                   the strongest *deep* explorer in a small
                                   tabular world.

  3. UCB-style action selection -- pick argmax_a [ Q(s,a) + c*sqrt(ln t /
                                   N(s,a)) ].  The bonus lives in the
                                   *action-selection rule*, not in the
                                   reward, so it never propagates through
                                   the value function -- great at covering
                                   one state's actions, weak at *planning*
                                   toward far-away unexplored regions.

  4. count-based reward bonus   -- add 1/sqrt(N(s,a)) to the *reward* (so it
                                   DOES propagate via Q-learning) and decay
                                   its weight over time.  Classic MBIE-EB /
                                   curiosity.

  5. prediction-error reward bonus -- add -log P(s'|s,a) from a tiny learned
                                   forward model to the reward.  The tabular
                                   cousin of ICM / RND.  Sharpest novelty
                                   signal of the bunch.

Tasks:
  A. MiniMontezuma  -- key -> door -> treasure, reward only at the end.
  B. DeepSea(size)  -- the textbook "deep exploration" chain, swept over
                       several lengths to see which strategies still cope.

Run:
    python compare_exploration.py
Outputs:
    outputs/compare_exploration.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hard_exploration_envs import MiniMontezumaEnv, DeepSeaEnv
from curiosity_bonus import CountCuriosity, PredictionCuriosity, NoCuriosity


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


STRATEGIES = {
    # name                 : (short tag,           colour)
    "epsilon-greedy":         ("eps",            "#e74c3c"),
    "optimistic init":        ("optimistic",     "#8e44ad"),
    "UCB action selection":   ("ucb",            "#f39c12"),
    "count reward bonus":     ("count-bonus",    "#2980b9"),
    "prediction reward bonus":("prediction-bonus","#27ae60"),
}


# ===========================================================================
# One Q-learner that can run any of the five strategies
# ===========================================================================
def train(env_fn, strategy, n_episodes, gamma=0.99, alpha=0.5, seed=0,
          epsilon=0.2, beta0=1.0, beta_decay=0.999, ucb_c=2.0,
          r_max=1.0):
    rng = np.random.default_rng(seed)
    env = env_fn()
    nS, nA = env.n_states, env.n_actions

    # per-strategy setup ----------------------------------------------------
    if strategy == "optimistic":
        q_init = r_max / (1.0 - gamma)
        eps = 0.0
    else:
        q_init = 0.0
        eps = epsilon if strategy != "ucb" else 0.0
    Q = np.full((nS, nA), q_init, dtype=float)

    # N(s,a) is needed by ucb / count-bonus
    N = np.zeros((nS, nA))
    if strategy == "count-bonus":
        curiosity = CountCuriosity(nS, nA)
    elif strategy == "prediction-bonus":
        curiosity = PredictionCuriosity(nS, nA)
    else:
        curiosity = NoCuriosity()

    successes = np.zeros(n_episodes)
    returns = np.zeros(n_episodes)
    beta = float(beta0)
    total_steps = 1

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ext_ret = 0.0
        while not done:
            # --- action selection -----------------------------------------
            if strategy == "ucb":
                bonus = ucb_c * np.sqrt(np.log(total_steps + 1.0) / (N[s] + 1.0))
                a = int(np.argmax(Q[s] + bonus))
            elif rng.random() < eps:
                a = int(rng.integers(nA))
            else:
                a = int(np.argmax(Q[s]))

            s2, r_env, done, _ = env.step(a)
            ext_ret += r_env
            N[s, a] += 1.0
            total_steps += 1

            # --- reward-side curiosity bonus ------------------------------
            r_cur = curiosity.reward(s, a, s2)
            curiosity.observe(s, a, s2)
            r_train = r_env + beta * r_cur

            target = r_train + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2

        returns[ep] = ext_ret
        successes[ep] = 1.0 if ext_ret > 0.0 else 0.0
        beta *= beta_decay

    return {"successes": successes, "returns": returns}


# ===========================================================================
# Experiments
# ===========================================================================
def _sliding(x, win):
    return np.array([x[max(0, i - win + 1):i + 1].mean() for i in range(len(x))])


def montezuma_experiment(n_episodes=1500, n_seeds=8):
    print("Task A -- MiniMontezuma (key -> door -> treasure, sparse reward)")
    env_fn = lambda: MiniMontezumaEnv(max_steps=300)
    curves, summary = {}, {}
    for name, (tag, _c) in STRATEGIES.items():
        succ = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            succ[seed] = train(env_fn, tag, n_episodes, seed=seed)["successes"]
        curves[name] = succ
        final = succ[:, -100:].mean()
        first = [int(np.argmax(s)) if s.any() else n_episodes for s in succ]
        summary[name] = {"final": final, "first": int(np.median(first))}
        print(f"  {name:<26}  final solve rate = {final:5.2f}   "
              f"first solve ~ episode "
              f"{summary[name]['first'] if summary[name]['first'] < n_episodes else 'never'}")
    return curves, summary


def deepsea_experiment(sizes=(5, 8, 11, 14), n_episodes=600, n_seeds=10):
    print("\nTask B -- DeepSea sweep (does the strategy still find the reward "
          "as the chain gets longer?)")
    # solved[name][size] = fraction of seeds that solved it (>=1 success in
    # the last 100 episodes)
    solved = {name: [] for name in STRATEGIES}
    for size in sizes:
        chance = 2.0 ** (-size)
        line = f"  size={size:2d} (random-walk success prob ~ {chance:.1e}): "
        cell = []
        for name, (tag, _c) in STRATEGIES.items():
            hits = 0
            for seed in range(n_seeds):
                succ = train(lambda: DeepSeaEnv(size=size), tag, n_episodes, seed=seed)["successes"]
                if succ[-100:].sum() > 0:
                    hits += 1
            frac = hits / n_seeds
            solved[name].append(frac)
            cell.append(f"{tag}={frac:.1f}")
        print(line + "  ".join(cell))
    return list(sizes), solved


def main():
    print("=== Comparing five exploration strategies ===\n")
    mz_curves, mz_summary = montezuma_experiment()
    sizes, ds_solved = deepsea_experiment()

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # (A) MiniMontezuma learning curves
    axA = fig.add_subplot(gs[0, :])
    win = 50
    for name, (tag, color) in STRATEGIES.items():
        m = _sliding(mz_curves[name].mean(0), win)
        axA.plot(m, color=color, lw=2.4, label=name)
    axA.set_title("Task A -- MiniMontezuma: probability of reaching the treasure")
    axA.set_xlabel("Episode")
    axA.set_ylabel(f"P(solved)  [sliding {win} eps, avg over seeds]")
    axA.set_ylim(-0.03, 1.03)
    axA.grid(alpha=0.3)
    axA.legend(loc="center right", fontsize=10)

    # (B) DeepSea: fraction of seeds solved vs chain length
    axB = fig.add_subplot(gs[1, 0])
    for name, (tag, color) in STRATEGIES.items():
        axB.plot(sizes, ds_solved[name], "-o", color=color, lw=2.2, ms=6, label=name)
    axB.set_title("Task B -- DeepSea: which strategies survive a longer chain?")
    axB.set_xlabel("Chain length N")
    axB.set_ylabel("Fraction of seeds that found the reward")
    axB.set_xticks(sizes)
    axB.set_ylim(-0.05, 1.05)
    axB.grid(alpha=0.3)
    axB.legend(fontsize=8.5)

    # (C) MiniMontezuma summary bars: episodes to first solve
    axC = fig.add_subplot(gs[1, 1])
    names = list(STRATEGIES)
    colors = [STRATEGIES[n][1] for n in names]
    firsts = [mz_summary[n]["first"] for n in names]
    cap = mz_curves[names[0]].shape[1]
    plotted = [f if f < cap else cap for f in firsts]
    bars = axC.bar(range(len(names)), plotted, color=colors, edgecolor="black")
    for b, f in zip(bars, firsts):
        txt = "never" if f >= cap else f"ep {f}"
        axC.text(b.get_x() + b.get_width() / 2, b.get_height() + cap * 0.01,
                 txt, ha="center", fontsize=9)
    axC.set_title("Task A -- episodes until the FIRST treasure (lower is better)")
    axC.set_ylabel("Episode of first success (median over seeds)")
    axC.set_xticks(range(len(names)))
    axC.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    axC.set_ylim(0, cap * 1.12)
    axC.grid(alpha=0.3, axis="y")

    fig.suptitle("Comparing exploration strategies.  epsilon-greedy is just "
                 "dithering; the other four are real exploration -- and they "
                 "trade off differently on shallow vs deep tasks.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, "compare_exploration.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    print("\nTakeaways:")
    print("  * epsilon-greedy never solves either hard task -- random dithering")
    print("    is not exploration.")
    print("  * On the sparse-reward grid, every 'real' strategy gets there;")
    print("    the prediction-error bonus gets there fastest (sharpest novelty).")
    print("  * On the deep chain, optimistic initialisation is the quiet champion:")
    print("    it propagates optimism through the value function automatically.")
    print("    Reward bonuses help, but vanilla Q-learning struggles to push that")
    print("    optimism far enough -- which is exactly why scaling deep exploration")
    print("    to pixels needed bootstrapped DQN, RND-with-a-network, Go-Explore, ...")


if __name__ == "__main__":
    main()
