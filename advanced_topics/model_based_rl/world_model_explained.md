# Training a World Model: Teaching the Agent to Dream 🌍

## What Is a "World Model"?

A **world model** is the agent's *internal copy of the universe*. Give it a
state and an action, and it predicts what will happen next:

```
(state, action)  ──►  Neural Network  ──►  (next_state, reward)
```

It is not the real world — it is a **simulator the agent built for itself** by
watching reality and learning to mimic it.

Once trained, the model lets the agent ask "what-if" questions without taking
any real action:

> *"If I push left now and then right twice, where will I end up? Will the pole
> fall?"*

The agent can ponder a hundred plans inside its model in the time it would take
to make one real move. That is the whole point.

---

## A Real-Life Analogy

Think about how *you* solve a puzzle. You don't physically move every piece
into every slot. You **imagine** what happens if piece A goes here. If that
mental simulation looks wrong, you reject it before lifting a finger.

Your brain has a learned world model — built from years of seeing how objects
behave — that lets you simulate outcomes before committing.

Other examples:

- **A chess player** imagines moves several turns ahead.
- **A driver** thinking, "If I brake now, the car behind has enough room."
- **A child** stacking blocks: "If I put the big one on top, the tower will
  wobble." (They learned this model by previously knocking towers down.)

In every case, **a mental model + imagination = better decisions with less risk**.

---

## How Does the Agent Build Its Model?

It just **watches**. Specifically:

1. **Collect data.** Let any policy (even random) interact with the real
   environment for a while. Save every transition:
   ```
   (state, action, reward, next_state)
   ```
2. **Train a neural network** to predict `next_state` and `reward` from
   `(state, action)`. This is supervised learning: each saved transition is a
   labelled example where the input is "what the agent saw and did" and the
   label is "what actually happened next."
3. **Validate.** Hold out 10% of the data and check the model's predictions
   against the real ones. Low error means the model has captured the
   environment's **dynamics**: how states change after actions.

The trick we use: instead of predicting `next_state` directly, predict the
**delta** `next_state − state`. Most physics is incremental ("the cart moved a
tiny bit"), and small targets are kinder to neural networks.

---

## Our Setup

| Choice | Value | Why |
|--------|-------|-----|
| Environment | `CartPole-v1` | 4-D state, 2 actions — easy to model |
| Data | 20,000 transitions from a random policy | Wide coverage of the state space |
| Network | MLP, 2 × 128 ReLU hidden | MLP = Multi-Layer Perceptron (standard "vanilla" neural network). Two hidden layers of 128 neurons using ReLU activations. Enough capacity, fast to train. |
| Loss | MSE on `(delta_state, reward)` | MSE = Mean Squared Error (average of squared prediction errors). Standard regression loss. |
| Optimizer | Adam, lr = 1e-3, 30 epochs | Adam = adaptive optimizer (adjusts learning rates per parameter automatically). Off-the-shelf means no special tuning needed. |

The whole training finishes in a few seconds on CPU.

---

## What Does "Good" Look Like?

Two diagnostics matter:

### 1. Single-step accuracy (validation MSE)

This is "how well does the model predict ONE step into the future?" After 30
epochs you should see validation MSE in the **1e-4 to 1e-3** range. That is
tiny — pole angles and cart positions are accurate to a few decimal places.

### 2. **Compounding error** on k-step rollouts

This is the *real* test. Take a state, feed it through the model, then take
its prediction and feed it back through the model — for `k` steps in a row.
The error grows because every step adds a bit of noise on top of the previous
prediction.

```
Step  1:  L2 error ≈ 0.01   (almost perfect)
Step  5:  L2 error ≈ 0.05
Step 10:  L2 error ≈ 0.15
Step 20:  L2 error ≈ 0.40   (visibly drifting)
```

*(L2 error = Euclidean distance between the predicted next state and the real one —
think of it as "how far off is the model's guess in the 4-D state space?")*

**Why this matters.** If we plan 15 steps ahead with the model, the *exact*
state at step 15 will be wrong — but if the relative ranking of "good plans
vs. bad plans" is preserved, planning still works. (This is what
`model_based_planning.py` exploits.)

The plot in `outputs/world_model.png` shows both diagnostics side by side: the
training-loss curve goes nicely down on a log scale, and the rollout-error
curve goes nicely up.

---

## Why Predict the *Delta*?

Compare two ways of phrasing the same problem to the network:

| Target | Typical magnitude | Easy or hard? |
|--------|------------------:|--------------|
| `next_state`        | 0–2.4 (cart pos) | Network must reproduce position **and** the tiny change |
| `next_state - state`| ~0.02            | Network just learns the tiny change |

Predicting the delta also means: if the network outputs zeros (as an untrained, beginner
network often does), the prediction is simply "nothing moved" — a sensible, safe default for a single
timestep. In contrast, predicting the absolute `next_state` directly would initially output completely random garbage values, causing early training to be highly unstable.

---

## What This Buys Us

A trained world model is the foundation for:

- **Planning** — search over imagined action sequences (see
  `model_based_planning.py`).
- **Dyna-style augmentation** — train a Q-network on imagined data to
  multiply sample efficiency.
- **Curiosity / exploration** — visit states the model can't predict well.
- **Dreamer / World-Models papers** — train a *policy* entirely inside the
  model with zero real-world interaction beyond initial data collection.

---

## Limits and Cautions

- **Out-of-distribution drift.** The model only knows the part of the world it
  has seen. Plan too aggressively and you end up in regions the model has never
  visited — predictions there are pure fantasy.
- **Compounding error.** Planning over long **horizons** (many steps into the future) is unreliable due to accumulating errors, as the chart shows.
  Modern systems address this by using **probabilistic ensembles** (training multiple models and checking if they agree, like in PETS or Dreamer) so the planner
  knows exactly *how uncertain* the model is at every step and can avoid risky, unknown paths.
- **Stochastic environments.** A standard deterministic regressor predicts only the *mean* average
  outcome and completely misses the spread of possible outcomes. Complex, real-world environments require probabilistic
  models (like those with Gaussian outputs, or **latent stochastic models** — networks that
  encode the world state as a probability distribution in a compressed space,
  letting them capture genuine randomness rather than averaging it away) to accurately represent uncertainty and randomness.

---

## Key Words

| Term | Plain English |
|------|---------------|
| **World model** | A neural net that mimics the environment |
| **Dynamics** | The function `(s, a) → s'` |
| **Reward model** | The function `(s, a) → r` (often bundled in) |
| **One-step prediction** | What the model outputs from a real state |
| **Rollout** | Repeated one-step predictions, feeding outputs back in |
| **Compounding error** | Small errors that grow over a rollout |

---

## One-Sentence Summary

> **A world model is a tiny neural copy of the universe that the agent can
> consult — and dream inside — before risking a real action.**

Next: `model_based_planning.py` puts this model to work for actual decision-making.
