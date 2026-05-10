# REINFORCE: Teaching a Robot to Make Better Choices

## What Are We Trying to Do?

Imagine you have a robot playing a video game. Every second, the robot must choose:
**"Should I push the button or not?"**

Instead of memorizing every situation in a table (like Q-learning), we want the robot to learn
a **recipe** — a set of rules that directly says: "In this situation, do this action."

This recipe is called a **policy** (π).

---

## The Old Way vs. The New Way

**Old way (Q-learning / DQN):** Learn how GOOD each action is (Q-values), then pick the best.
> "Pushing LEFT has score 7, pushing RIGHT has score 5 → push LEFT!"

**New way (Policy Gradient):** Directly learn which action to CHOOSE.
> "When the pole tilts right, push RIGHT with 80% chance, push LEFT with 20% chance."

**Real-life example:** Learning to ride a bike.
- The old way: calculate the *exact score* for "lean left 5 degrees" vs "lean left 7 degrees."
- The new way: just practice until your **body** learns — push the foot that feels right!

---

## How Does REINFORCE Work?

REINFORCE watches the robot play a full game from start to finish (one **episode**), then
asks: "Which actions led to a good score? Let's do more of those!"

### Step-by-Step

**1. Play an episode**

The robot makes choices and collects experience:
```
Step 1: State = [pole tilting right] → Action = push RIGHT → Reward = +1
Step 2: State = [pole nearly balanced] → Action = push RIGHT → Reward = +1
Step 3: State = [pole tilting left] → Action = push LEFT  → Reward = +1
...
Step 47: State = [pole fell!] → Episode over
```

**2. Compute returns**

For each step, compute G_t — the **total reward from that point forward**:
```
G_at_step_47 = 1
G_at_step_46 = 1 + 0.99 × 1 = 1.99
G_at_step_45 = 1 + 0.99 × 1.99 = 2.97
...
G_at_step_1  = 47 (roughly — higher return because it was from the start)
```

The γ = 0.99 **discount factor** means rewards far in the future count a little less.

**Real-life example:** Getting a gold star on day 1 of school feels more exciting than knowing
you *might* get one on day 100. Future rewards are "discounted" slightly.

**3. Update the policy**

For each action taken:
> If G_t was HIGH (that action led to a great outcome): **do it more!**
> If G_t was LOW (that action led to a bad outcome): **do it less!**

The math: `loss = -log_prob(action) × G_t`

Taking the gradient and updating the policy is like telling the robot:
*"That action you took at step 20? You should do it 5% more often next time!"*

---

## What is a Policy Network?

Instead of a table, we use a **neural network** to represent the policy.

```
Observation      Policy Network       Action Probabilities
[cart pos]  →   [128 neurons]  →  →  [push LEFT: 30%]
[cart speed] →  [128 neurons]        [push RIGHT: 70%]
[pole angle] →
[pole speed] →
```

The network outputs **probabilities** for each action. We then sample:
> Roll a dice → 1-30: push LEFT, 31-100: push RIGHT

**Real-life example:** A weather app says "70% chance of rain." You don't KNOW it will rain — you
decide based on probability. The robot does the same thing!

---

## Normalization: Why We Subtract the Mean

Before using G_t to update, we normalize:
```
G_normalized = (G - mean(G)) / std(G)
```

**Why?** Imagine all rewards are positive (which they are in CartPole — always +1 per step).
Without normalization, EVERY action looks "good" and the update signal is confusing.

After normalization, some returns are positive (above average → push more), and some are
negative (below average → push less). The signal becomes much cleaner!

**Real-life example:** Your teacher grades on a curve. If the average score is 70 and you got
85, that's great! But if the average is 90 and you got 85, that's below average. The raw score
alone doesn't tell the whole story.

---

## The Problem: High Variance

REINFORCE has a big weakness: **variance**. The returns G_t are very noisy.

**Real-life example:** Imagine judging a chef by tasting only ONE meal from each restaurant.
Sometimes the chef had a bad day, sometimes the ingredients were off. One meal isn't enough
to reliably know if the restaurant is good!

REINFORCE waits for a FULL episode before updating. One episode might be very lucky, another
very unlucky. The gradients jump all over the place.

This is why the learning curve (in the plot) looks jagged:
- Some runs get 500 (amazing!)
- Some runs drop to 50 (terrible!)

Despite the noise, REINFORCE eventually learns — but it takes patience.

---

## The Results

```
Episode  100 | Avg reward (last 100):  43.1
Episode  200 | Avg reward (last 100): 193.9
Episode  500 | Avg reward (last 100): 408.4
Episode 1000 | Avg reward (last 100): 500.0  ← Solved!
```

The robot learns to balance the pole for the maximum 500 steps — SOLVED!

Despite its variance problems, REINFORCE on CartPole is effective because:
1. Episodes are short (so we get many per training run)
2. The optimal policy is simple (mostly push in the direction the pole tilts)

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Policy** | The robot's recipe for choosing actions |
| **Episode** | One full game from start to finish |
| **Return G_t** | Total future reward from this moment |
| **Discount γ** | Future rewards count a little less than immediate ones |
| **Normalize** | Subtract the average so the signal is cleaner |
| **Variance** | How much the gradient estimates jump around |

---

## What's Next?

The big weakness of REINFORCE is **variance**. In the next script (`reinforce_baseline.py`),
we add a **baseline** that dramatically reduces this noise — without changing what the
algorithm learns on average.

The key idea: instead of asking "was this action good?" we ask "was this action **better than
expected**?" That small change makes learning much more stable.
