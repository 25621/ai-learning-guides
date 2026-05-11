# Phase 5 · Multi-Agent RL

Three practical works that build up multi-agent RL from the simplest setting
(one shot, two actions) to a proper environment library:

| # | Implementation | Concept Explainer |
|---|----------------|-------------------|
| 1 | [`matrix_games.py`](matrix_games.py) | [Matrix Games: The Simplest Multi-Agent World](matrix_games_explained.md) |
| 2 | [`self_play_tic_tac_toe.py`](self_play_tic_tac_toe.py) | [Self-Play: Teaching an Agent by Letting It Play Itself](self_play_tic_tac_toe_explained.md) |
| 3 | [`pettingzoo_explore.py`](pettingzoo_explore.py) | [Exploring PettingZoo Environments](pettingzoo_explore_explained.md) |

## Quick Start

```bash
# from the repo root
source venv/bin/activate
cd advanced_topics/multi_agent_rl

# 1) Independent Q-learners on RPS / Prisoner's Dilemma / Stag Hunt
python matrix_games.py

# 2) Self-play Q-learning on Tic-Tac-Toe
python self_play_tic_tac_toe.py

# 3) PettingZoo-style API demo + independent learners on a coordination game
python pettingzoo_explore.py
#   (optional) for the real PettingZoo rollout at the end:
#   pip install "pettingzoo[classic]"
```

All plots land in `outputs/`.

## What You Should Observe

| Script | Expected outcome |
|--------|------------------|
| `matrix_games.py` | RPS: both players hover near ⅓-⅓-⅓ (mixed Nash), the curves chase each other. Prisoner's Dilemma: Defect rises to ~1.0 for both — selfish equilibrium. Stag Hunt: most seeds collapse to (Hare, Hare). |
| `self_play_tic_tac_toe.py` | After 50k self-play episodes: **~95-99% win-or-draw vs random**, **all 200 draws** in agent-vs-agent greedy play (tic-tac-toe is a drawn game). |
| `pettingzoo_explore.py` | AEC random rollout averages ~0 return. Independent Q-learners climb from ~0 to **+20 to +25** per episode (max +25). If `pettingzoo` is installed the real `rps_v2` env also rolls out at the end. |

## How They Fit Together

```
       ┌──────────────────────────────┐
       │ matrix_games.py:             │
       │   1-shot, no state           │  <- pure interaction dynamics
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ self_play_tic_tac_toe.py:    │
       │   state + turn-taking        │  <- AlphaZero's core idea, in a Q-table
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ pettingzoo_explore.py:       │
       │   real MARL API + training   │  <- gateway to MPE, MAgent, Atari MA, ...
       └──────────────────────────────┘
```

The shared idea: **the environment is other agents, and other agents change
as they learn.** Once you can navigate non-stationarity, you can scale the
same recipe up to MADDPG, QMIX, MAPPO, AlphaZero, OpenAI Five, AlphaStar.

## Dependencies

Everything works with what is already in `requirements.txt`:

- `numpy` for tables and math
- `matplotlib` for plots
- `gymnasium` is not strictly required by these scripts but is imported
  transitively if you `pip install pettingzoo[classic]`

To enable the real PettingZoo rollout at the end of `pettingzoo_explore.py`:

```bash
pip install "pettingzoo[classic]"
```

The script gracefully skips that section if the package is missing.
