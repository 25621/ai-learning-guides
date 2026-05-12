# Phase 5 · Model-Based RL

Three connected practical works that build on each other:

| # | Implementation | Concept Explainer |
|---|----------------|-------------------|
| 1 | [`dyna_q.py`](dyna_q.py) | [Dyna-Q: Learning Faster by Imagining](dyna_q_explained.md) |
| 2 | [`world_model.py`](world_model.py) | [Training a World Model](world_model_explained.md) |
| 3 | [`model_based_planning.py`](model_based_planning.py) | [Planning with a Learned Model (MPC)](model_based_planning_explained.md) |

## Quick Start

```bash
# from the repo root
source venv/bin/activate
cd advanced_topics/model_based_rl

# 1) Dyna-Q on the 6x9 Dyna Maze (tabular, no deep learning)
python dyna_q.py

# 2) Train a neural-network world model on CartPole-v1
python world_model.py
#   -> saves outputs/world_model.pt for step (3)

# 3) Use the learned model to plan via MPC + random shooting
#    MPC (Model Predictive Control): at every step, imagine many futures with
#    the world model, pick the best plan, execute only its first action, repeat.
#    Random shooting: generate N random action sequences, score each through the
#    model, and commit to the first action of the highest-scoring one.
python model_based_planning.py
```

All plots and the saved world-model checkpoint go to `outputs/`.

## What You Should Observe

| Script | Expected outcome |
|--------|------------------|
| `dyna_q.py`               | `n=50` planning reaches a near-optimal path within a handful of episodes; `n=0` needs many more episodes to get there. By episode 50 all three have converged to ~10 steps — the gap is in *how fast* (see the learning-curve plot), not the final step count. |
| `world_model.py`          | Validation MSE (Mean Squared Error, measuring how wrong the model's predictions are compared to reality) drops to ~1e-4–1e-3 in 30 epochs. 1-step error is tiny (like predicting the weather for tomorrow), but 20-step rollout error visibly compounds (like predicting the weather a month from now). |
| `model_based_planning.py` | MPC averages **150–500** reward; random baseline averages ~22. Many MPC episodes hit the 500-step ceiling. |

## How They Fit Together

```
       ┌──────────────────────────────┐
       │ dyna_q.py:                   │
       │   tabular model + planning   │  <- toy demo of the idea
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ world_model.py:              │
       │   neural-network model       │  <- scales to continuous states
       └──────────────┬───────────────┘
                      │ saves world_model.pt
                      │   (a PyTorch checkpoint — the trained network weights
                      │    serialised to disk so the next script can load them)
                      ▼
       ┌──────────────────────────────┐
       │ model_based_planning.py:     │
       │   MPC w/ random shooting     │  <- planning at decision time
       └──────────────────────────────┘
```

The shared idea: **build a model of the world, then use it.** That is the
foundation of powerful modern model-based RL systems, such as:
- **MuZero** (DeepMind's AI that learned to play chess, shogi, and Atari games without being told the rules).
- **Dreamer** (An RL agent that learns behaviors purely by imagining futures in its "sleep", like a human dreaming about a maze to find the way out).
- **PETS** (Probabilistic Ensembles with Trajectory Sampling, a system that learns how uncertain its models are to avoid catastrophic mistakes, like a self-driving car being cautious in heavy rain).
