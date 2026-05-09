# Linear Q-Learning for CartPole 🎪

## What Is CartPole?

Imagine a broomstick balanced upright on your finger. If you move your finger left or right
just a little bit, you can keep the broomstick from falling. That is **CartPole**!

A tiny robot sits on a cart (a box on wheels) and has a pole sticking up on top.
The robot can only push the cart **left** or **right**. It has to learn to keep that
pole balanced as long as possible — just like you balancing a broomstick!

The robot can see 4 things about the world:
1. Where the cart is
2. How fast the cart is moving
3. How much the pole is leaning
4. How fast the pole is leaning

---

## The Big Problem: Too Many States!

Remember Q-learning from Phase 2? It used a big table to remember how good each action
is in each situation (state). That worked great for Frozen Lake — there were only 16
squares on the ice.

But CartPole is different! The cart can be at **any position**, moving at **any speed**,
with the pole at **any angle**. There are basically **infinite possible states**! We can't
make a table with infinite rows. We would need a universe-sized notebook!

**Real-life example:** Imagine you are learning to ride a bike. You can't memorize every
possible wobble — there are too many! Instead, you learn a **rule**: "when I lean left, 
push right; when I lean right, push left." A simple rule works for ALL wobbles.

---

## The Solution: A Magic Formula

**Linear function approximation** replaces the giant table with a **tiny formula**:

> **Score(situation, action) = w₁ × cart_position + w₂ × cart_speed + w₃ × pole_angle + w₄ × pole_speed**

- The `w` numbers are called **weights** — they're like knobs you can twist
- We learn **different weights for each action** (push-left and push-right)
- The formula gives a score for how good each action is right now

**Real-life example:** Think of a simple recipe: "1 cup flour + 2 eggs + ½ cup butter."
The weights (1, 2, ½) tell you how much each ingredient matters. We're learning the
recipe for good decisions!

---

## How Does It Learn?

The robot tries things, gets feedback, and tweaks the weights:

1. **Robot pushes the cart** (picks the action with the highest score)
2. **Physics happen** (the pole tilts a bit, the cart moves)
3. **Robot gets a reward** (+1 for every step the pole stays up, 0 if it falls)
4. **Robot asks:** "Was the actual result better or worse than I predicted?"
5. **Robot tweaks the weights** to be closer to reality next time

This is the **Semi-Gradient TD Update** — a fancy name for "nudge the recipe a little bit
based on the surprise."

> **New weight = Old weight + Learning rate × (What really happened − What I predicted) × Feature**

---

## What Our Code Found

When you run `linear_q_cartpole.py`, the robot:

- Starts off terrible (the pole falls in 10–30 steps)
- Gradually learns good weights over 3,000 tries
- Eventually keeps the pole balanced for 100–400+ steps!

The plot shows the **learning curve** — how the score gets better over time.
It will be bumpy (learning is never smooth!), but the trend should go up.

---

## Why This is Cool (and Limited!)

**Cool:** A tiny formula with just 8 numbers (4 weights × 2 actions) can balance a pole!
No giant table needed.

**Limited:** The formula is too simple for complex tasks. It assumes bigger numbers always
mean bigger effects (which isn't always true). For harder games like Atari, we need
**neural networks** — which is what DQN does!

---

## Key Vocabulary

| Word | Meaning |
|------|---------|
| **Feature** | One measurable thing about the world (e.g., pole angle) |
| **Weight** | How much a feature affects the decision |
| **Linear** | The formula is just multiplication and addition (no complicated curves) |
| **Semi-gradient** | Update the weights by following the direction of less error |
| **Function approximation** | Using a formula instead of a table |

---

## What's Next?

Linear approximation is like using a straight ruler to draw a curve — it works okay for
simple shapes but not complex ones. For Atari games with millions of possible situations,
we need **Deep Q-Networks (DQN)** — neural networks that can learn much more complex
patterns. That's in the next file!
