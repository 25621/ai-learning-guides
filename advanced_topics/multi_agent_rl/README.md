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

# 1) Independent Q-learners on three classic one-shot games:
#    - RPS (Rock-Paper-Scissors): purely competitive, zero-sum — one player's
#      gain is the other's loss.  No stable pure strategy exists.
#    - Prisoner's Dilemma: both players can cooperate or defect; rational
#      selfishness pushes both toward the worse outcome.
#    - Stag Hunt: two hunters must coordinate to catch a stag (big reward) or
#      play it safe alone with a hare (small reward).
python matrix_games.py

# 2) Self-play Q-learning on Tic-Tac-Toe
#    The same agent plays both X and O, improving against itself episode after
#    episode — the same idea that powers AlphaGo Zero.
python self_play_tic_tac_toe.py

# 3) PettingZoo-style API demo + independent learners on a coordination game
#    A coordination game rewards both agents for making the same choice.
#    The script demonstrates both the AEC (turn-based) and Parallel (simultaneous)
#    API styles, then trains Q-learners until coordination emerges.
#    A "rollout" here means running one complete episode from reset to done,
#    collecting observations, actions, and rewards at each step.
python pettingzoo_explore.py
#   (optional) for the real PettingZoo rollout at the end:
#   pip install "pettingzoo[classic]"
```

All plots land in `outputs/`.

## What You Should Observe

| Script | Expected outcome |
|--------|------------------|
| `matrix_games.py` | **RPS:** both players hover near ⅓-⅓-⅓ — the *mixed Nash equilibrium* (each action chosen with equal probability so the opponent can't exploit any pattern). The curves chase each other rather than settling, because any drift is punished. **Prisoner's Dilemma:** "Defect" rises to ~1.0 for both — the *selfish equilibrium* (a stable state where no single player benefits from changing strategy). **Stag Hunt:** most seeds collapse to (Hare, Hare) — the safe-but-worse equilibrium. |
| `self_play_tic_tac_toe.py` | After 50k self-play episodes: **~95-99% win-or-draw vs random**, **all 200 draws** in agent-vs-agent greedy play. Tic-tac-toe is a *drawn game*: with perfect play from both sides, neither player can force a win — every game ends in a tie. All-draws from the trained agent is therefore proof of convergence to near-optimal play. |
| `pettingzoo_explore.py` | The *AEC* (Agent-Environment-Cycle) random rollout averages ~0 return — agents cycle through turns one at a time but act randomly. Independent Q-learners then climb from ~0 to **+20 to +25** per episode (max +25). If `pettingzoo` is installed the real `rps_v2` env also rolls out at the end. |

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

**Glossary for the diagram:**

| Term | What it means |
|------|---------------|
| **MARL** | Multi-Agent Reinforcement Learning — the field of RL where more than one agent learns simultaneously in a shared environment. |
| **MPE** | Multi-Particle Environments — a suite of small 2-D cooperative and competitive tasks (e.g. "cooperative navigation", "predator-prey") shipped with PettingZoo. A standard MARL testbed. |
| **MAgent** | A grid-world engine designed for *hundreds or thousands* of agents at once. Used to study emergent group behaviour (e.g. battle, gather). Available via `pettingzoo[magent]`. |
| **Atari MA** | Multi-agent Atari games in PettingZoo (e.g. Pong duel, Space Invaders co-op). Classic Atari games repurposed so two human-or-AI players compete or cooperate in the same ROM. |

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
