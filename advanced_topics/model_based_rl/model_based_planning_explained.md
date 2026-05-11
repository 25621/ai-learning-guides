# Planning With a Learned World Model

## The Big Idea

Imagine you are about to cross a busy street. Before you take a step, your brain
**imagines** a few possible futures:

- "If I run now → I might get to the other side."
- "If I walk slowly → a car might come and hit me."
- "If I wait 3 seconds, then walk → safe!"

You pick the future you like best, and you take **the first step** of that plan.
Then you do the whole thing over again from the new spot.

That is called **planning**. And when the imagination uses a *learned* world model
(from `world_model.py`), it is called **model-based planning**, the same idea
behind MuZero and Dreamer.

---

## Model Predictive Control (MPC)

We use the simplest possible planner, called **random shooting**:

```
At every real step:
  1. Roll 200 dice to invent 200 plans, each 15 actions long.
  2. For each plan, IMAGINE the future with the world model.
  3. Add up the reward of each imagined future.
  4. Pick the plan with the highest score.
  5. Do ONLY THE FIRST action of that plan, for real.
  6. Repeat from the new state.
```

**Real-life example:** Playing chess.
- You think about 10 moves you could make.
- For each, you imagine what the opponent might do.
- You pick the move that leads to the best position you can see.
- You play just that one move, then look again.

Even though you imagined many moves, you only ever play one at a time. That is
MPC.

---

## Why Imagine 15 Steps Instead of Just 1?

A myopic planner (horizon = 1) only sees **one move ahead**. CartPole punishes
this badly — the pole might be tipping slowly, and the only way to recover is a
sequence of 5 or 10 well-timed pushes. A 1-step planner can't see that.

A 15-step planner imagines a longer movie. It can spot a plan like:

> "Push left, left, right, right, left — by the end the pole is balanced again."

Even if the imagination gets a bit wrong by step 15, the **first** action is what
matters, and 15 steps of context gives that first action much better information.

---

## What the Code Compares

We test three controllers on CartPole for 10 game seeds each:

| Controller | What it does |
|------------|--------------|
| 🩶 Random | Pushes randomly. No brain. Drops the pole quickly. |
| 🟡 MPC horizon = 1 | Plans only 1 step ahead. Misses the longer story. |
| 🟢 MPC horizon = 15 | Plans 15 steps ahead through the learned model. Should balance for much longer. |

The amazing thing: **the horizon-15 controller never trained on CartPole.** It only
trained the *model* of CartPole. The actions come from planning, not from a
policy network.

---

## The Pattern Behind Modern Model-Based RL

Almost every modern model-based RL paper looks like this loop:

```
Real experience  →  better world model  →  better imagined plans  →  better real experience
```

- **Dyna-Q** uses tabular planning. (We did this in `dyna_q.py`.)
- **MPC + neural world model** = what we did here.
- **MuZero** uses a learned model PLUS a Monte Carlo tree search instead of random
  shooting. Same shape.
- **Dreamer** trains a policy *inside* the imagined world, instead of planning at
  each step. Same shape.

You now understand the foundation of all of them.

---

## Real-Life Examples of MPC

| Real life | The "model" | The "planning" |
|-----------|-------------|----------------|
| Self-driving car | Physics of the car | Sample 100 future trajectories, pick safest |
| Rocket landing (SpaceX) | Rocket dynamics | Recompute the descent plan every 10ms |
| Chess engine | Rules of chess | Search future board positions, pick best move |
| Walking robot | Joint physics | Pick foot placements that stay balanced |
| Climate control | Heat model of the building | Choose when to turn on the AC |

---

## What the Plot Shows

A bar chart with three bars:

- Random: low bar (cart-pole falls quickly).
- MPC horizon = 1: a little better, but the pole still falls.
- MPC horizon = 15: much taller bar — close to the 500-step max — because the
  planner can see far enough ahead to keep the pole steady.

The dashed green line at 500 is the maximum possible reward (CartPole ends after
500 steps).

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| Planning | Trying things out in your imagination before doing them |
| MPC | Plan a long path, take 1 step, plan again |
| Horizon (H) | How many steps ahead we imagine |
| Random shooting | Roll dice to invent many plans, pick the best |
| Sample efficiency | Doing well *without* needing tons of real-world tries |

---

## The Whole Phase 5 Picture

You have now built:
1. **Dyna-Q** — learn from real + imagined steps using a tabular model.
2. **World model** — a neural network that predicts the future.
3. **MPC planner** — use the world model to choose actions.

These three pieces are the heart of **model-based reinforcement learning**.
Stack them together, swap the planner for a search tree, and you have MuZero.
Train a policy inside the world model and you have Dreamer.

Congratulations — you now speak the language of modern model-based RL. 🎉
