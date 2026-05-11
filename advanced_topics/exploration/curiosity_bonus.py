"""
Work item 1 -- "Implement a curiosity bonus"

A *curiosity bonus* (a.k.a. an **intrinsic reward**) is an extra reward the
agent hands ITSELF for visiting things it finds surprising or novel.  The
environment's real ("extrinsic") reward is left untouched -- we just train
the agent on the sum

        r_total(s, a, s')  =  r_env(s, a, s')  +  beta * r_curiosity(s, a, s')

and let the weight `beta` decay over time.  When the real reward is sparse
(you only see it at the end of a long correct sequence -- think
Montezuma's Revenge), this curiosity term is the thing that *pulls* the
agent down the chain in the first place.

This file is the small "library" the other two scripts import.  It
implements two classic flavours of curiosity, both fully tabular so you
can watch them work:

  (A) COUNT-BASED novelty      r_cur = 1 / sqrt(N(s, a) + 1)
      "I've barely tried this -> reward me for trying it again."  The
      granddaddy of exploration bonuses (MBIE-EB, UCB).  Naturally decays
      as the visit count piles up.

  (B) PREDICTION-ERROR novelty  --  the idea behind ICM (Pathak et al. 2017)
      Keep a tiny learned *forward model* P(s' | s, a) -- here just a table
      of transition counts -- and let curiosity be the model's SURPRISE at
      what actually happened:
            r_cur = -log P(observed s' | s, a)
      A brand-new (s, a) is maximally surprising; once the (deterministic)
      transition has been seen a few times the model predicts it perfectly
      and the bonus collapses to ~0.  That is exactly ICM's slogan --
      "curiosity = the error of a model that predicts the consequences of
      your actions" -- minus the neural network.

The demo (`python curiosity_bonus.py`) trains a tabular Q-learner on
`MiniMontezumaEnv`, a key -> door -> treasure gridworld where the reward
shows up ONLY at the treasure.  Plain epsilon-greedy never gets there;
the curiosity bonuses do.

Outputs:
    outputs/curiosity_bonus.png   -- learning curves + grid visitation maps
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hard_exploration_envs import MiniMontezumaEnv


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Curiosity modules  (the heart of this work item)
# ===========================================================================
class NoCuriosity:
    """The baseline: no intrinsic reward at all (plain epsilon-greedy)."""
    name = "epsilon-greedy (no curiosity)"

    def reward(self, s, a, s2):
        return 0.0

    def observe(self, s, a, s2):
        pass


class CountCuriosity:
    """Count-based novelty bonus:  r_cur = 1 / sqrt(N(s, a) + 1).

    N(s, a) is how many times we have *taken* action a in state s.  The +1
    makes the very first try worth 1.0; afterwards the bonus shrinks as
    1/sqrt(N), so heavily-trodden state-actions stop being interesting.
    """
    name = "count-based bonus  1/sqrt(N)"

    def __init__(self, n_states, n_actions):
        self.N = np.zeros((n_states, n_actions))

    def reward(self, s, a, s2):
        return 1.0 / np.sqrt(self.N[s, a] + 1.0)

    def observe(self, s, a, s2):
        self.N[s, a] += 1.0


class PredictionCuriosity:
    """Prediction-error (ICM-style) novelty bonus.

    A tabular forward model: trans[s, a, s'] = how many times taking a in s
    landed in s'.  The model's probability of the observed transition is
        P(s' | s, a) = trans[s, a, s'] / sum_x trans[s, a, x]
    and curiosity is the surprisal  -log P(s' | s, a).  An (s, a) that has
    never been tried (or that just did something the model has never seen)
    pays the maximum surprise, log(n_states).  Once the deterministic
    transition has been observed a few times P -> 1 and the bonus -> 0.
    """
    name = "prediction-error bonus  -log P(s'|s,a)"

    def __init__(self, n_states, n_actions):
        self.trans = np.zeros((n_states, n_actions, n_states))
        self.max_surprise = float(np.log(n_states))

    def reward(self, s, a, s2):
        total = self.trans[s, a].sum()
        if total == 0.0:
            return self.max_surprise            # never tried this (s, a)
        p = self.trans[s, a, s2] / total
        if p == 0.0:
            return self.max_surprise            # tried (s, a) but never saw this s'
        return min(-np.log(p), self.max_surprise)

    def observe(self, s, a, s2):
        self.trans[s, a, s2] += 1.0


_CURIOSITY = {
    "none":       NoCuriosity,
    "count":      CountCuriosity,
    "prediction": PredictionCuriosity,
}


def make_curiosity(kind, n_states, n_actions):
    cls = _CURIOSITY[kind]
    return cls() if cls is NoCuriosity else cls(n_states, n_actions)


# ===========================================================================
# A tabular Q-learner that trains on (extrinsic + beta * curiosity)
# ===========================================================================
def q_learning_with_curiosity(env_fn, curiosity_kind="prediction", n_episodes=1500,
                               alpha=0.5, gamma=0.99, epsilon=0.2,
                               beta0=1.0, beta_decay=0.999,
                               q_init=0.0, seed=0, track_predicate=None):
    """Tabular Q-learning with an optional decaying curiosity bonus.

    Args:
        env_fn          : zero-arg factory returning a fresh env each call.
        curiosity_kind  : one of {"none", "count", "prediction"}.
        beta0/beta_decay: curiosity weight starts at beta0, *= beta_decay
                          after every episode (so exploration fades).
        q_init          : initial Q value (set high for "optimistic init").
        track_predicate : optional fn(env) -> bool, sampled at episode end
                          (e.g. "did the agent ever hold the key?").

    Returns dict with:
        returns       : extrinsic return per episode (curiosity excluded)
        successes     : 1.0 if returns[ep] > 0 else 0.0
        tracked       : track_predicate value per episode (or None)
        state_visits  : how often each state was visited (single run)
        Q             : the final Q-table
    """
    rng = np.random.default_rng(seed)
    env = env_fn()
    n_states, n_actions = env.n_states, env.n_actions
    Q = np.full((n_states, n_actions), float(q_init))
    cur = make_curiosity(curiosity_kind, n_states, n_actions)

    returns = np.zeros(n_episodes)
    successes = np.zeros(n_episodes)
    tracked = np.zeros(n_episodes) if track_predicate is not None else None
    state_visits = np.zeros(n_states)
    beta = float(beta0)

    for ep in range(n_episodes):
        s = env.reset()
        state_visits[s] += 1
        done = False
        ext_return = 0.0
        ever_true = False
        while not done:
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                a = int(np.argmax(Q[s]))
            s2, r_env, done, _ = env.step(a)
            ext_return += r_env

            r_cur = cur.reward(s, a, s2)
            cur.observe(s, a, s2)
            bonus = beta * r_cur

            target = r_env + bonus + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])

            s = s2
            state_visits[s] += 1
            if track_predicate is not None and track_predicate(env):
                ever_true = True

        returns[ep] = ext_return
        successes[ep] = 1.0 if ext_return > 0.0 else 0.0
        if tracked is not None:
            tracked[ep] = 1.0 if ever_true else 0.0
        beta *= beta_decay

    return {"returns": returns, "successes": successes, "tracked": tracked,
            "state_visits": state_visits, "Q": Q}


# ===========================================================================
# Demo: curiosity vs no-curiosity on MiniMontezuma
# ===========================================================================
def _sliding(x, win):
    return np.array([x[max(0, i - win + 1):i + 1].mean() for i in range(len(x))])


def main():
    n_episodes = 1500
    n_seeds = 8
    win = 50
    env_fn = lambda: MiniMontezumaEnv(max_steps=300)
    has_key = lambda env: env.has_key

    print("=== Curiosity bonuses on MiniMontezuma (key -> door -> treasure) ===")
    print(f"Reward (+1) is given ONLY at the treasure, after ~15 perfect moves.")
    print(f"Averaging over {n_seeds} seeds, {n_episodes} episodes each.\n")

    flavours = {
        "none":       "#e74c3c",
        "count":      "#2980b9",
        "prediction": "#27ae60",
    }
    results = {}
    sample_run = {}
    for kind in flavours:
        succ = np.zeros((n_seeds, n_episodes))
        keyrate = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            out = q_learning_with_curiosity(env_fn, curiosity_kind=kind,
                                            n_episodes=n_episodes, seed=seed,
                                            track_predicate=has_key)
            succ[seed] = out["successes"]
            keyrate[seed] = out["tracked"]
            if seed == 0:
                sample_run[kind] = out
        results[kind] = {"succ": succ, "keyrate": keyrate}

        final_treasure = succ[:, -100:].mean()
        final_key = keyrate[:, -100:].mean()
        first = [int(np.argmax(s)) if s.any() else n_episodes for s in succ]
        med_first = int(np.median(first))
        label = _CURIOSITY[kind].__name__
        print(f"  {label:<22}  reaches key {final_key:5.2f}   "
              f"reaches treasure {final_treasure:5.2f}   "
              f"first treasure ~ episode "
              f"{med_first if med_first < n_episodes else 'never'}")

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(14.5, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05],
                          width_ratios=[2.4, 1, 1, 1])

    # (1) treasure-success curves
    ax1 = fig.add_subplot(gs[0, 0])
    for kind, color in flavours.items():
        m = _sliding(results[kind]["succ"].mean(0), win)
        ax1.plot(m, color=color, lw=2.3, label=_CURIOSITY[kind].name)
    ax1.set_ylabel(f"P(reach treasure)  [sliding {win} eps, {n_seeds} seeds]")
    ax1.set_xlabel("Episode")
    ax1.set_title("Reaching the treasure")
    ax1.set_ylim(-0.03, 1.03)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="center right", fontsize=9)

    # (2) key-pickup curves (an intermediate milestone)
    ax2 = fig.add_subplot(gs[1, 0])
    for kind, color in flavours.items():
        m = _sliding(results[kind]["keyrate"].mean(0), win)
        ax2.plot(m, color=color, lw=2.3, label=_CURIOSITY[kind].name)
    ax2.set_ylabel(f"P(pick up the key)  [sliding {win} eps]")
    ax2.set_xlabel("Episode")
    ax2.set_title("Reaching the key (a milestone halfway down the chain)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center right", fontsize=9)

    # (3-5) state-visitation heatmaps over the grid (single seed).  We sum
    # the has_key=0 and has_key=1 slices so the picture is a plain 2-D map.
    probe = MiniMontezumaEnv()
    rows, cols = probe.rows, probe.cols
    wall = np.array([[1 if probe.grid[r][c] == "#" else 0
                      for c in range(cols)] for r in range(rows)])
    for i, (kind, color) in enumerate(flavours.items()):
        axh = fig.add_subplot(gs[:, 1 + i])
        v = sample_run[kind]["state_visits"]
        grid_visits = np.zeros((rows, cols))
        for st in range(probe.n_states):
            rr, cc, _hk = MiniMontezumaEnv.decode(st, cols)
            grid_visits[rr, cc] += v[st]
        disp = np.log1p(grid_visits)
        disp = np.ma.array(disp, mask=wall)
        cmap = plt.cm.magma.copy()
        cmap.set_bad("dimgray")
        axh.imshow(disp, cmap=cmap, aspect="equal")
        # annotate key / door / treasure / start
        axh.scatter([probe.start[1]], [probe.start[0]], marker="s", s=70,
                    color="white", edgecolor="black", label="start")
        axh.scatter([probe.key_pos[1]], [probe.key_pos[0]], marker="P", s=110,
                    color="gold", edgecolor="black", label="key")
        axh.scatter([probe.door_pos[1]], [probe.door_pos[0]], marker="|", s=200,
                    color="orange", linewidths=3, label="door")
        axh.scatter([probe.treasure_pos[1]], [probe.treasure_pos[0]], marker="*",
                    s=200, color="cyan", edgecolor="black", label="treasure")
        axh.set_title(_CURIOSITY[kind].name.split("(")[0].split("bonus")[0].strip(),
                      fontsize=9)
        axh.set_xticks([]); axh.set_yticks([])
        if i == 0:
            axh.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                       ncol=2, fontsize=7.5)

    fig.suptitle("Curiosity bonus = self-given reward for novelty.  "
                 "Right: where seed-0 spent its steps  (log scale; gray = walls).  "
                 "No-curiosity stays huddled near the start; curiosity floods both rooms.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "curiosity_bonus.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    print("\nTakeaway: epsilon-greedy treats every step as an independent coin "
          "flip and basically never threads the key->door->treasure chain.  "
          "A curiosity bonus turns 'I haven't seen this' into reward, so the "
          "agent systematically pushes into unexplored territory -- and as the "
          "novelty wears off, beta and the bonus both fade, leaving a clean "
          "exploiting policy.")


if __name__ == "__main__":
    main()
