# Phase 5 — Model-Based RL

This directory implements the three practical work items for the **Model-Based RL**
section of Phase 5.

| File | What it does |
|------|--------------|
| `dyna_q.py` | Classic Dyna-Q on a deterministic GridWorld maze. Compares planning steps n = 0, 5, 50. |
| `world_model.py` | Trains an MLP world model on CartPole-v1 transitions. Saves `outputs/world_model.pt`. |
| `model_based_planning.py` | Loads the saved world model and uses random-shooting MPC to balance the pole. |

Each script has a matching `*_explained.md` written in clear, accessible
language with real-life examples.

---

## Running

The repo's existing `venv` already has `numpy`, `matplotlib`, `gymnasium`, and
`torch` installed (verified in `venv/bin`). From the project root:

```bash
source venv/bin/activate
cd advanced_topics/model_based_rl

python dyna_q.py                  # ~5 s, writes outputs/dyna_q.png
python world_model.py             # ~30 s, writes outputs/world_model.{png,pt}
python model_based_planning.py    # ~1 min, writes outputs/model_based_planning.png
```

Run them in that order — `model_based_planning.py` loads the checkpoint produced
by `world_model.py`.

---

## What each experiment shows

### 1. Dyna-Q (`dyna_q.py`)
Three Q-learning agents differ only in how many *imagined* updates they do per
real step. The plot's left panel (log scale) makes it obvious: with `n = 50`
planning steps, the agent finds a near-optimal route within ~2 episodes; pure
Q-learning (`n = 0`) needs an order of magnitude more steps to get there.

Expected first-episode lengths (deterministic, seed = 42):
- `n = 0`  → several thousand steps for ep 1, dropping toward ~16 by ep 50
- `n = 5`  → ep 1 still long, but ep 2–5 already near optimal
- `n = 50` → ep 2 already at near-optimal length

The right panel renders the maze and overlays the greedy path of the best agent.

### 2. World model (`world_model.py`)
We collect 20,000 random transitions, then train an MLP to predict
`Δstate`, `reward`, and the termination probability. Training loss drops to
~10⁻³ within 30 epochs. Open-loop rollout error grows with horizon (one of the
core challenges of model-based RL):

- horizon 1   → ~0.01 L2
- horizon 5   → ~0.1 L2
- horizon 20  → ~0.5 L2 (errors compound)

The trained network is saved to `outputs/world_model.pt`.

### 3. Model-based planning (`model_based_planning.py`)
Random shooting MPC over the learned model, evaluated over 10 random seeds:

- Random policy           → ~20 reward (pole drops fast)
- MPC horizon = 1, K = 50 → ~30–60 reward (myopic; can't recover from tilt)
- MPC horizon = 15, K = 200 → much higher, often close to 500 (the max)

The horizon-15 planner never trained a policy — every action comes from
imagining 200 random 15-step rollouts through the learned model and picking the
best one.

---

## How this maps onto modern model-based RL

```
   Dyna-Q  ──(swap table for neural net)──▶ World model + MPC  ──(swap planner for tree search)──▶ MuZero
                                                    │
                                                    └──(train a policy inside imagination)──▶ Dreamer
```

You now have the building blocks for the entire model-based RL family.
