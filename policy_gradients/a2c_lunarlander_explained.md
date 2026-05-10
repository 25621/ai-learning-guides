# A2C: The Actor and the Critic Work Together

## The Big Idea

REINFORCE waits until the game is completely over before updating. That's like a coach who
watches an entire football game in silence, then gives all feedback at the end.

**A2C (Advantage Actor-Critic)** gives feedback DURING the game — every few steps, the coach
pauses to say: "That pass was great! That tackle was bad!"

This is much faster and more efficient.

---

## Meet the Two Characters

> **What is LunarLander?** Throughout this document we use the **LunarLander** environment — a physics simulation where you control a small spacecraft and must land it softly on a target pad on the moon using three engines (left, main, and right). It is a standard benchmark in reinforcement learning, available in the Gymnasium library.

### The Actor 🎭
The **Actor** is the policy — it decides which action to take.

> "I'm in this state. Should I fire the left engine or the right engine?"

**Real-life example:** The *driver* of a car who turns the steering wheel and presses pedals.

### The Critic 🎬
The **Critic** estimates how good the current situation is — the value V(s).

> "Being in THIS state is worth about +150 points of total future reward."

**Real-life example:** The *navigator* sitting next to the driver, saying "We're on a good road —
expect to arrive in 30 minutes." or "We're heading into traffic — this is going to be slow."

### They Share a Brain
In our implementation, both use the **same neural network backbone**:

```
          State (8 numbers for LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Shared Layers           │
          │  [256 neurons] → ReLU    │
          │  [256 neurons] → ReLU    │
          └────────┬────────┬────────┘
                   ↓        ↓
          Actor Head    Critic Head
          [4 outputs]   [1 output]
          (action probs) (V(s))
```

- **ReLU** (Rectified Linear Unit): an activation function applied after each layer — it outputs `max(0, x)`, keeping positive values and zeroing out negatives. This lets the network learn non-linear patterns.
- **action probs**: the probability of taking each of the 4 actions. The Actor samples from this distribution to choose an action each step.

**Real-life example:** One brain, two jobs — like a taxi driver who both drives (actor)
AND knows if the route is good (critic). Sharing the brain means learning faster!

---

## The Advantage: Was This Better Than Expected?

Just like REINFORCE with baseline, A2C computes the **Advantage**:

> A(s, a) = "Actual result" − "What we expected"

But here, "actual result" comes from the Critic's **n-step bootstrap** (**bootstrapping** = using the Critic's own prediction V(s) to approximate the value of future steps, instead of waiting for the actual episode to end — like estimating your final exam score mid-semester using your current grade):

```
Actual TD return: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Advantage A_t = TD return - V(s_t)
```

**Real-life example:** You expect to score 3 goals this game (V(s)). If you score 5 goals,
your advantage is +2. If you score 1 goal, your advantage is -2.

Positive advantage → "that action helped more than expected → do it more!"
Negative advantage → "that action helped less than expected → do it less!"

---

## Why Use Multiple Parallel Environments?

Our A2C uses **8 copies** of LunarLander running at the same time!

**Why?** Because experiences from a single environment are **correlated** — one step
follows the previous step closely. This correlation fools the neural network into thinking
patterns are more reliable than they are.

With 8 environments, each step gives 8 independent experiences from very different situations.
This breaks the correlation and stabilizes training dramatically.

**Real-life example:** To learn about weather, which is better:
- Watching one city for 8 consecutive hours (correlated — if it was sunny at 2pm, it's probably sunny at 3pm)
- Watching 8 cities simultaneously (decorrelated — different weather patterns, more information!)

```
Environment 1: [landed on moon, fire left, crash, reset...]
Environment 2: [falling too fast, fire both, hover, land...]
Environment 3: [tilting right, fire right, stabilize, land...]
...
Environment 8: [drifting left, fire left, steady, ...]
```

All 8 update the network simultaneously — 8× more diverse experience per update!

---

## N-Step Updates: Don't Wait for the Game to End

REINFORCE waits for a full episode (could be 1000 steps!).

A2C updates every **n_steps = 128 steps**:

```
Play 128 steps across 8 environments
    → Get 128 × 8 = 1024 experience tuples
    → Compute advantages and returns
    → Update the Actor and Critic
    → Play 128 more steps...
```

**Real-life example:** A student studying for an exam.
- REINFORCE style: Read the entire textbook, THEN take practice tests.
- A2C style: Read 10 pages, do a quiz, read 10 more pages, do a quiz...

More frequent feedback = faster learning!

---

## Three Losses Combined

A2C trains with three loss terms simultaneously:

A **loss** is the number the optimizer tries to minimize. Smaller loss means the network's
current behavior is closer to the training objective.

### 1. Actor Loss (Policy Gradient)
Make advantageous actions more likely:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
If A > 0: increase probability of that action
If A < 0: decrease probability of that action

### 2. Critic Loss (Value Function MSE)
Make value predictions more accurate (**MSE** = Mean Squared Error: square the prediction error and average it — squaring penalizes large mistakes more heavily than small ones):
```
L_critic = E[(V(s) - return)²]
```
Like training any **regression** model (regression = predicting a continuous number, here the expected return V(s)) — minimize prediction error.

### 3. Entropy Bonus (Exploration)
Keep the policy from becoming too confident too fast:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
High entropy = diverse action choices = exploration
Low entropy = confident, narrow choices = exploitation

**Real-life example:** The entropy bonus is like a teacher saying "Don't just guess A on every
multiple-choice question! Try different answers so you learn what works."

```
Total loss = L_actor + 0.5 × L_critic - 0.01 × entropy
```

---

## LunarLander: A Harder Challenge

**LunarLander-v3** is a Gymnasium (formerly OpenAI Gym) environment — "v3" is the version number indicating the third revision of this environment. The agent controls a small spacecraft that must land safely on a designated pad on the moon. It is much harder than CartPole:
- 8-dimensional state space (position, velocity, angle, leg contact, fuel)
- 4 discrete actions (do nothing, fire left, fire main, fire right)
- Reward: +100 for landing, -100 for crashing, small fuel penalties

The training curve shows gradual improvement from highly negative rewards toward positive ones.
A2C on LunarLander requires significant experience before the lander learns basic stability.

---

## Key Equations

```
n-step return:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Advantage:      A_t = G_t - V(s_t)
Actor update:   θ_π ← θ_π - α ∇ L_actor
Critic update:  θ_V ← θ_V - α ∇ L_critic
```

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Actor** | The policy — decides what to do |
| **Critic** | The value function — judges how good the situation is |
| **Advantage** | "Was this better than expected?" (actual - expected) |
| **N-step return** | Look n steps into the future before bootstrapping with V(s) |
| **Parallel envs** | Multiple environments for decorrelated, diverse experience |
| **Entropy bonus** | Encourages the actor to keep trying new things |

---

## What's Next?

A2C is great but has one major weakness: it updates the policy too aggressively sometimes.
A single bad update can destroy all the good learning in a previous update.

**PPO (Proximal Policy Optimization)** fixes this with a clever "safety clip" that prevents
any single update from changing the policy too much.

See `ppo_scratch.py` for the PPO implementation!
