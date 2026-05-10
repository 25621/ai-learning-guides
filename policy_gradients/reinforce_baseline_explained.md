# REINFORCE with Baseline: Cutting Through the Noise

## The Problem with Plain REINFORCE

Imagine you're a student trying to decide whether your answer on a test was good.

**Bad feedback:** "You got 7 points!"

Is 7 good? If the maximum is 10, yes! If everyone else got 9, no! Without context,
you can't tell if you should change your answer style.

This is exactly the problem with REINFORCE: it uses **raw returns** (G_t) to evaluate
actions. A total return score of 200 points might be amazing or terrible depending on the situation.

---

## Enter the Baseline

A **baseline** b(s) is a reference point: "What reward do I **expect** in this situation?"

Instead of asking "Was this action good?", we ask:

> **"Was this action better than what I'd normally expect?"**

```
Old signal: update ∝ G_t
New signal: update ∝ (G_t - b(s_t))
```

**Real-life example:** You scored 85 on a math test.
- If the class average is 60 → your answer was **25 points above average** → great!
- If the class average is 90 → your answer was **5 points below average** → needs work!

The **advantage** (G_t - b(s)) is positive when you did better than expected and
negative when you did worse. This is a much cleaner learning signal!

---

## What's the Baseline?

The natural baseline is the **value function V(s)**:

> V(s) = "Expected total reward if I'm in state s and play my current policy"

We learn this with a separate **Value Network** (also called the baseline network or critic):

```
State  →  [128 neurons]  →  [128 neurons]  →  V(s)   (single number)
```

For each state the agent visits, V(s) predicts the expected return. If the actual
return G_t is higher than V(s), the action was better than expected!

---

## Two Networks Learning Together

```
Episode happens
     ↓
Compute actual returns G_t
     ↓
         ┌─────────────────────────────┐
         │ Advantage = G_t - V(s_t)    │
         │  +: action was better        │
         │  -: action was worse         │
         └─────────────────────────────┘
              ↓                  ↓
    Update Policy Network   Update Value Network
    (make good actions     (make predictions more
     more/less likely)      accurate next time)
```

**Real-life example:** Two friends go to a restaurant together.

- Friend 1 (Value Network): "I predict this dish will be a 7/10"
- Friend 2 (Policy Network): You try the dish and rate it 9/10
- Advantage = 9 - 7 = +2 → "That was better than expected! Order it again!"

Next visit, Friend 1 updates their prediction closer to 9/10.
Friend 2 is more likely to order that dish next time.

---

## Why Does This Reduce Variance?

**Mathematical proof (intuition):**

Without baseline: `gradient ∝ ∇log π(a|s) × G_t`

The G_t values vary a lot from episode to episode:
```
Episode 1: G = [45, 44, 43, ..., 1]   (medium game)
Episode 2: G = [500, 499, ..., 1]      (great game!)
Episode 3: G = [12, 11, ..., 1]        (terrible game)
```

The gradient estimates jump wildly because G_t is large and noisy.

With baseline: `gradient ∝ ∇log π(a|s) × (G_t - V(s_t))`

The advantage (G_t - V(s_t)) is much smaller and centered near zero:
```
Episode 1: advantage ≈ [-2, +1, -3, ..., 0]   (small, centered)
Episode 2: advantage ≈ [+10, +8, ..., +3]      (this game WAS great)
Episode 3: advantage ≈ [-5, -6, ..., -2]       (this game WAS bad)
```

**Real-life example:** Measuring your running speed.
- Without baseline: "I ran 8 km/h" (meaningless without context)
- With baseline: "I ran 2 km/h FASTER than my average" (clearly good!)

The advantage is always a comparison — it's naturally smaller and more stable.

---

## Crucially: No Bias!

The baseline doesn't change WHAT the algorithm learns — only HOW FAST and STABLY it learns.

**Why?** Because the expected advantage is always 0 in expectation:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Any b(s) that doesn't depend on the action works as a valid baseline!

**Real-life example:** Grading on a curve doesn't change who performed best — it just
makes the scores easier to interpret. The ranking stays the same; only the scale changes.

---

## The Results

```
No baseline  — Final 100-ep avg: 500.0, grad var: 599.3
With baseline — Final 100-ep avg: 491.4, grad var: 578.8
```

Both methods reach near-perfect performance on CartPole, but notice:
1. The **gradient variance** is measurable (plot right side shows the variance over training)
2. With baseline, the agent reaches high performance **more reliably** — fewer crashes back to low reward during training

The variance reduction is more dramatic in harder environments (LunarLander, MuJoCo).

---

## Key Equations

```
Baseline value:   V(s) ← V(s) + α(G_t - V(s))   [minimize MSE]
Policy gradient:  θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Advantage:        A_t = G_t - V(s_t)
```

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Baseline b(s)** | Expected reward in state s — our reference point |
| **Advantage A_t** | "Was this action better than expected?" |
| **Value Network** | A neural net that learns to predict expected returns |
| **Variance reduction** | Less noise in gradient estimates → more stable learning |
| **Unbiased** | The baseline doesn't change the target policy on average; it only makes the learning signal less noisy and more stable |

---

## What's Next?

The baseline is actually the beginning of something much more powerful: **Actor-Critic** methods.

Instead of computing V(s) only at the end of an episode, the Actor-Critic updates
V(s) at every single step using **Temporal Difference** learning. This makes updates
much faster and allows the agent to learn from incomplete episodes!

See `a2c_lunarlander.py` for the full Actor-Critic implementation.
