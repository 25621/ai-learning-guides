# Phase 5 — Model-Based RL

This directory implements the three practical work items for the **Model-Based RL**
section of Phase 5.

| File | What it does |
|------|--------------|
| `dyna_q.py` | Classic **Dyna-Q** — a Q-learning agent that also memorises what happened and *replays those memories* to learn faster. Tested on a deterministic GridWorld maze; compares planning steps n = 0, 5, 50. |
| `world_model.py` | Trains an **MLP world model** (a small neural network that learns to predict "if I take action A in state S, what will happen next?") on **CartPole-v1 transitions** (the recorded snapshots of a balancing-pole game: where the pole was, what action was taken, where it ended up). Saves the trained network to `outputs/world_model.pt`. |
| `model_based_planning.py` | Loads the saved world model and uses **random-shooting MPC** (tries thousands of random action sequences in imagination, picks the one that looks best, then executes only the first action) to balance the pole. |

> **What is `outputs/world_model.pt`?**
> A `.pt` file is a **PyTorch checkpoint** — a binary snapshot of a trained neural
> network's weights and biases, saved with `torch.save()`. Think of it as a
> "save file" for the model, so `model_based_planning.py` can load an already-trained
> network without rerunning the training step.

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
real step. **Imagined** (or *simulated*) updates use the agent's internal memory
of past `(state, action → next state, reward)` pairs — like mentally replaying a
past experience instead of physically repeating it — to squeeze extra learning
out of each real interaction. The plot's left panel (log scale) makes it obvious:
with `n = 50` planning steps, the agent finds a near-optimal route within ~2
episodes; pure Q-learning (`n = 0`) needs an order of magnitude more steps to
get there.

Expected first-episode lengths (deterministic, seed = 42):
- `n = 0`  → several thousand steps for ep 1, dropping toward ~16 by ep 50
- `n = 5`  → ep 1 still long, but ep 2–5 already near optimal
- `n = 50` → ep 2 already at near-optimal length

The right panel renders the maze and overlays the greedy path of the best agent.

### 2. World model (`world_model.py`)
We collect 20,000 random **transitions** (each transition is one recorded step:
`state + action → next state + reward + done`; like a single frame of a video
game together with what button was pressed and what score changed) from
CartPole-v1, then train an MLP to predict `Δstate` (the *change* in the pole's
position and velocity), `reward`, and the **termination probability** (how likely
the episode is to end — i.e., the pole falling over — at that step). Training
loss drops to ~10⁻³ within 30 epochs.

**Open-loop rollout error** is the accumulated prediction mistake when you run
the model for several steps in a row *without* checking back against the real
environment ("open loop" = no feedback correction). It grows with **horizon**
(the number of steps you look ahead) because every small error feeds into the
next prediction:

- horizon 1   → ~0.01 L2
- horizon 5   → ~0.1 L2
- horizon 20  → ~0.5 L2 (errors compound)

Think of it like whispering a sentence through a chain of 20 people: each person
mishears slightly, and by the end the message is very different from the
original. This compounding error is one of the core challenges of model-based RL.

The trained network is saved to `outputs/world_model.pt`.

### 3. Model-based planning (`model_based_planning.py`)
Random shooting MPC over the learned model, evaluated over 10 random seeds:

- Random policy               → ~20 reward (pole drops fast)
- MPC horizon = 1, K = 50    → ~30–60 reward (**myopic** — a horizon-1 planner
  only looks one step ahead, like a driver who steers only for the next metre
  instead of the next curve; it cannot anticipate and recover from a building
  tilt before it is already too late)
- MPC horizon = 15, K = 200  → much higher, often close to 500 (the max)

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

- **MuZero** (DeepMind, 2019) — instead of planning with a lookup table like
  Dyna-Q, it uses a *learned* neural world model and replaces random shooting
  with a smart tree-search (MCTS). It mastered Chess, Go, and Atari without
  ever being told the rules of those games.
- **Dreamer** (Google Brain, 2019) — uses the same neural world model idea, but
  instead of searching through action sequences at test time, it trains a full
  policy (and a value function) entirely *inside* the imagination of the model.
  This makes it very sample-efficient on continuous-control tasks like robotics.

You now have the building blocks for the entire model-based RL family.
