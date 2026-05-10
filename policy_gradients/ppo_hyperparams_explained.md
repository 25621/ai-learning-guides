# PPO Hyperparameter Sensitivity: What Matters Most?

## Why Hyperparameters Matter

Imagine baking a chocolate cake. The recipe calls for:
- 2 eggs
- 200g flour
- 1 teaspoon baking powder
- 35 minutes at 180°C

If you use 10 eggs, the cake explodes. If you use 0.1 teaspoon of baking powder, it doesn't rise.
If you bake at 300°C for 10 minutes, it burns outside and is raw inside.

**Hyperparameters in PPO are like ingredients and oven settings.** The right combination works
beautifully; wrong settings can prevent learning entirely.

This script systematically tests 3 key hyperparameters by changing only ONE at a time,
running each setting with 3 different random seeds, and comparing the results.

---

## The Three Experiments

### Experiment 1: Clip Epsilon (ε)

```
ε = 0.05   (very conservative — only tiny policy changes allowed)
ε = 0.2    (standard — balanced safety and speed)
ε = 0.4    (aggressive — allows large policy changes)
```

**What does ε control?**

ε is the size of the "safety window" around the old policy:
```
ratio must stay in [1 - ε,  1 + ε]
ε=0.05: ratio in [0.95, 1.05]  ← tiny changes
ε=0.2:  ratio in [0.80, 1.20]  ← standard  
ε=0.4:  ratio in [0.60, 1.40]  ← large changes
```

**Real-life example:** Think of ε as "how far you're allowed to steer the car in one move."
- ε=0.05: Like driving on ice — tiny adjustments only
- ε=0.2:  Normal driving — reasonable turns
- ε=0.4:  Racing driver — aggressive steering, risk of spinning out

**Expected results:**
- ε=0.05: Slow but stable learning (too cautious)
- ε=0.2:  Good balance (the "Goldilocks" value)
- ε=0.4:  Can learn fast but may overshoot and oscillate

---

### Experiment 2: Learning Rate

```
lr = 1e-4  (slow but stable)
lr = 3e-4  (standard)
lr = 1e-3  (fast but risky)
```

**What does learning rate control?**

The learning rate is like the "step size" when climbing a hill:
- Too small: Takes forever to reach the top (converges slowly)
- Too large: You overshoot the peak and fall down the other side (diverges)
- Just right: Steady progress toward the summit

**Real-life example:** Tuning a guitar string.
- lr=1e-4: Tiny turns of the tuning peg — takes forever but precise
- lr=3e-4: Normal tuning — find the right pitch in a few turns
- lr=1e-3: Big yanks on the peg — might snap the string!

**Expected results:**
- lr=1e-4: Eventually good but very slow
- lr=3e-4: Best performance overall
- lr=1e-3: Fast initial progress, then instability

---

### Experiment 3: Update Epochs (K)

```
K = 3   (conservative — few passes through each batch)
K = 10  (standard)
K = 20  (aggressive — many passes through each batch)
```

**What do update epochs control?**

After collecting a rollout (batch of experience), PPO reuses it for K optimization passes.
More epochs = squeeze more learning from each batch, but risk overfitting to stale data.

**Real-life example:** A student practicing with a set of 20 math problems.
- K=3:  Do each problem 3 times → still learning, don't overfit the practice set
- K=10: Do each problem 10 times → solid mastery of these specific problems
- K=20: Do each problem 20 times → memorize solutions without really understanding math!

The clipping prevents catastrophic overfitting, but too many epochs can still slow overall learning.

**Expected results:**
- K=3:  Less efficient (some learning potential wasted per batch)
- K=10: Good balance
- K=20: Risk of policy becoming too confident on stale data

---

## How to Read the Results

The plot shows three graphs, each varying one hyperparameter:

```
Left graph:    Clip Epsilon — which ε learns fastest?
Middle graph:  Learning Rate — which lr is most stable?
Right graph:   Update Epochs — which K finds the best policy?
```

Each line is the **average reward over 3 seeds** (to reduce randomness).

**What to look for:**
1. **Learning speed:** Which line reaches high reward fastest?
2. **Final performance:** Which line achieves the highest final reward?
3. **Stability:** Which line has the least oscillation?

A good hyperparameter balances all three!

---

## Methodology: Scientific Experimentation

This experiment uses **ablation study** design:
1. Pick default values: ε=0.2, lr=3e-4, K=10
2. Change ONE parameter at a time
3. Keep everything else fixed
4. Compare results

This tells us the effect of EACH parameter in isolation.

**Real-life example:** Testing whether a new fertilizer helps plants:
- Change fertilizer, keep everything else the same (same soil, water, sunlight)
- If the plants grow better → fertilizer helped!

---

## Common Findings in Practice

| Hyperparameter | Too Small | Sweet Spot | Too Large |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Slow convergence | ε ≈ 0.2 | Instability |
| **lr** | Too slow | 2.5e-4 to 3e-4 | Divergence |
| **K (epochs)** | Waste data | K = 4-10 | Overfitting |
| **n_steps** | Too noisy | 128-2048 | Memory |
| **batch_size** | Too noisy | 32-256 | Memory |

These "sweet spots" can shift depending on the environment!

---

## The Key Insight: PPO is Relatively Robust

Compared to earlier algorithms (like DQN without target networks), PPO is relatively
robust to hyperparameter choices. The clipping mechanism provides a natural safety net.

**Real-life example:** A car with ABS brakes vs. without:
- Without ABS (DQN): One wrong turn (bad hyperparameter) and you spin out
- With ABS (PPO): The car corrects itself — reasonable hyperparameters all work okay

This robustness is a major reason PPO is the most popular RL algorithm in practice!

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Ablation study** | Change one thing at a time to see its effect |
| **Clip epsilon ε** | Safety boundary — 0.2 is usually best |
| **Learning rate** | Step size — 2.5e-4 to 3e-4 works for most problems |
| **Update epochs K** | How many times to reuse each batch — 4-10 is standard |
| **Seeds** | Repeat each experiment multiple times to reduce luck effects |

---

## Summary: Policy Gradient Methods at a Glance

```
REINFORCE              A2C                    PPO
     │                  │                      │
Full episodes     N-step updates         N-step + clipping
Simple but noisy  Faster but unstable    Stable + efficient
Best for easy     Medium difficulty      Hard environments
environments      environments           (industry standard)
```

**If you only learn ONE algorithm from this phase, learn PPO.** It is the foundation of:
- OpenAI's ChatGPT training (RLHF uses PPO)
- DeepMind's AlphaGo follow-ups
- Most modern robotics research
- Video game playing AIs
