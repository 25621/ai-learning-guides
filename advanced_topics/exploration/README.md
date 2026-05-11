# Phase 5 · Exploration (sparse-reward & hard-exploration problems)

Three practical works that build the exploration toolkit from scratch on
two tiny, fully-tabular "hard exploration" environments — so you can *see*
exactly what each idea does without any neural-network noise.

| # | Implementation | Concept Explainer |
|---|----------------|-------------------|
| 1 | [`curiosity_bonus.py`](curiosity_bonus.py) | [Curiosity Bonus (Intrinsic Motivation)](curiosity_bonus_explained.md) |
| 2 | [`montezuma_revenge.py`](montezuma_revenge.py) | [Training on Montezuma's Revenge](montezuma_revenge_explained.md) |
| 3 | [`compare_exploration.py`](compare_exploration.py) | [Comparing Exploration Strategies](compare_exploration_explained.md) |

Shared scaffolding: [`hard_exploration_envs.py`](hard_exploration_envs.py)
defines the two environments every script uses.

## The two environments

- **`DeepSeaEnv(size=N)`** — the textbook *deep-exploration* chain (from
  bsuite / "Deep Exploration via Bootstrapped DQN"). An `N x N` grid; you
  descend one row per step and choose left/right. The only positive reward
  hides in the bottom-right corner, behind `N` consecutive correct moves,
  each carrying a tiny immediate cost. A uniform-random policy reaches it
  with probability `2^-N` — about one in a thousand at `N=10`, one in a
  million at `N=20`. The cleanest demonstration that "act randomly
  sometimes" (epsilon-greedy) is *not* exploration.

- **`MiniMontezumaEnv`** — a two-room gridworld with the *skeleton* of
  Montezuma's Revenge's first screen: walk to a **key**, pick it up (now
  the **door** in the dividing wall opens), walk through, reach the
  **treasure**. Reward (+1) is given *only* at the treasure, after ~15
  perfect moves. The `has_key` flag is part of the state, so there is a
  whole second room of genuinely-new states to be curious about once you
  grab the key.

Both are deterministic, have a few hundred states at most, and expose a
minimal Gym-like API (`reset()` / `step(a)` with integer states/actions).

## Quick Start

```bash
# from the repo root
source venv/bin/activate
cd advanced_topics/exploration

# sanity-check the environments
python hard_exploration_envs.py

# 1) Implement a curiosity bonus (count-based + prediction-error / ICM-style)
python curiosity_bonus.py
#   -> outputs/curiosity_bonus.png

# 2) "Train on Montezuma's Revenge": touch the real Atari game, then train
#    a tabular agent on the MiniMontezuma scale model
python montezuma_revenge.py
#   -> outputs/montezuma_revenge.png

# 3) Compare five exploration strategies on both environments
python compare_exploration.py
#   -> outputs/compare_exploration.png
```

All plots land in `outputs/`. Everything runs on CPU in a couple of
minutes per script.

> The real `ALE/MontezumaRevenge-v5` environment is *optional*.
> `montezuma_revenge.py` will use it if `ale-py` is installed (it prints
> the observation shape and shows a random agent scoring 0.0), and prints
> install instructions and skips that part otherwise. The tabular scale
> model runs regardless.

## What You Should Observe

| Script | Expected outcome |
|--------|------------------|
| `hard_exploration_envs.py` | A scripted always-RIGHT rollout on DeepSea returns ~+0.99; a scripted key→door→treasure path on MiniMontezuma returns +1.0 in 15 steps. |
| `curiosity_bonus.py` | On MiniMontezuma: **epsilon-greedy never reaches the treasure (score 0 forever)**; the **count-based bonus** reliably finds the key and reaches the treasure on a good fraction of episodes; the **prediction-error bonus** finds the treasure within ~20–25 episodes and then solves it every time. |
| `montezuma_revenge.py` | The real game: a random agent scores **0.0** over 2000 steps (sparse reward made concrete). The scale model: the **no-curiosity** agent stays at return 0; the **curiosity** agent learns the optimal ~15-step key→door→treasure route. |
| `compare_exploration.py` | epsilon-greedy never solves either task. On MiniMontezuma the four "real" strategies all get there, **prediction-error fastest**. On the DeepSea length sweep, **optimistic initialisation is the quiet champion** of *deep* exploration; UCB-in-action-selection and reward bonuses help but fade as the chain grows — which is why scaling deep exploration to pixels needed bootstrapped DQN / RND-with-a-network / Go-Explore. |

## How They Fit Together

```
       ┌─────────────────────────────────────┐
       │ curiosity_bonus.py:                 │
       │   intrinsic reward = novelty        │  <- the core mechanism
       │   (count-based & prediction-error)  │
       └──────────────┬──────────────────────┘
                      │  q_learning_with_curiosity(...)  +  the bonus classes
                      ▼
       ┌─────────────────────────────────────┐
       │ montezuma_revenge.py:               │
       │   the sparse-reward poster child;   │  <- "why exploration matters"
       │   real ALE touch + tabular training │
       └──────────────┬──────────────────────┘
                      │
                      ▼
       ┌─────────────────────────────────────┐
       │ compare_exploration.py:             │
       │   epsilon-greedy / optimistic /     │  <- the strategy zoo, head to head
       │   UCB / count / prediction          │
       └─────────────────────────────────────┘
```

The shared idea: **a reward that is zero almost everywhere can't teach
anything by itself.** Exploration is the art of either (a) being
*optimistic* about what you haven't tried, or (b) manufacturing a dense,
self-generated "novelty" signal that carries you to the first real reward.
Once you can do that in a Q-table, the same recipe scales up — with
neural networks doing the heavy lifting — to RND, Go-Explore, NGU,
Agent57, and the rest of modern hard-exploration RL.

## Dependencies

Everything works with what is already in the repo's `requirements.txt`:

- `numpy` — Q-tables, count tables, the tiny forward model
- `matplotlib` — plots and grid heat-maps
- `gymnasium` — only needed for the *optional* real Montezuma touch in
  `montezuma_revenge.py` (plus `ale-py`); the tabular environments are
  pure NumPy

## Where to Go Next

- 📄 **ICM** (Pathak et al., 2017) — curiosity = forward-model prediction
  error in a learned feature space: [arxiv.org/abs/1705.05363](https://arxiv.org/abs/1705.05363)
- 📄 **RND** (Burda et al., 2018) — the one that finally beat Montezuma's
  Revenge: [arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894)
- 📄 **Go-Explore** (Ecoffet et al., 2019) — "remember promising states
  and return to them": [arxiv.org/abs/1901.10995](https://arxiv.org/abs/1901.10995)
- 📄 **Unifying count-based exploration** (Bellemare et al., 2016) — the
  bridge between counting and prediction: [arxiv.org/abs/1606.01868](https://arxiv.org/abs/1606.01868)
- 📄 **Deep Exploration via Bootstrapped DQN** (Osband et al., 2016) — the
  source of the DeepSea task: [arxiv.org/abs/1602.04621](https://arxiv.org/abs/1602.04621)
