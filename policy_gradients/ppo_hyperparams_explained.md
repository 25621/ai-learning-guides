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
- ε=0.4:  Racing driver — aggressive steering, risk of **spinning out** (losing control because the change is too drastic, like a car skidding off the road)

**Expected results:**
- ε=0.05: Slow but stable learning (too cautious)
- ε=0.2:  Good balance (the **"Goldilocks" value** — not too small, not too large, just right — named after the fairy tale where Goldilocks picks the porridge that is neither too hot nor too cold)
- ε=0.4:  Can learn fast but may **overshoot and oscillate** (overshoot = go past the optimal policy; oscillate = bounce back and forth around it without settling, like a pendulum that swings too far in both directions)

---

### Experiment 2: Learning Rate

```
lr = 1e-4  (slow but stable)
lr = 3e-4  (standard)
lr = 1e-3  (fast but risky)
```

**What does learning rate control?**

The learning rate is like the "step size" when climbing a hill (each step = one update to the neural network's weights, moving it slightly in the direction that improves reward):
- Too small: Takes forever to reach the top (converges slowly)
- Too large: You overshoot the peak and fall down the other side (**diverges** — the training reward collapses or fluctuates wildly instead of improving steadily)
- Just right: Steady progress toward the summit

**Real-life example:** Tuning a guitar string.
- lr=1e-4: Tiny turns of the tuning **peg** (the knob you rotate to tighten or loosen a string) — takes forever but precise
- lr=3e-4: Normal tuning — find the right pitch in a few turns
- lr=1e-3: Big **yanks** (sudden hard pulls) on the peg — might **snap** the string (break it completely, just as too-large updates can break training irreversibly)!

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

After collecting a **rollout** (= playing the game for a stretch of time to gather new experience — like a student doing a homework session before reviewing it), PPO packages that experience into a **batch** (= the full set of state, action, reward tuples from that rollout). It then runs K **passes** (= full sweeps through the batch, each pass updating the network once) over the same data.
More epochs = squeeze more learning from each batch, but risk **overfitting to stale data** (= memorizing patterns that were true under the old policy but are no longer valid once the policy has been updated, like a student who memorizes last year's exam and fails a new one).

**Real-life example:** A student practicing with a set of 20 math problems.
- K=3:  Do each problem 3 times → still learning, don't overfit the practice set
- K=10: Do each problem 10 times → solid mastery of these specific problems
- K=20: Do each problem 20 times → **memorize solutions without really understanding math** (= the model fits the specific batch perfectly but loses the ability to generalize)!

> ⚠️ **"But the results for K=20 look fine — why should I care?"**
> PPO's clipping trick limits how much the policy can change per pass, so K=20 won't cause a sudden collapse.
> However, the agent is still quietly over-adapting to data that no longer reflects what the current policy would actually experience.
> This **slows down long-run learning**: each rollout teaches the agent less than it should, because later passes recycle increasingly stale information.
> The damage is gradual, not dramatic — which is exactly why it is easy to overlook in short experiments.

The clipping prevents catastrophic overfitting, but too many epochs can still slow overall learning.

**Expected results:**
- K=3:  Less efficient (some learning potential wasted per batch)
- K=10: Good balance
- K=20: Risk of policy becoming **too confident on stale data** (= the network's updates are driven by experiences that no longer match what the current policy would encounter, quietly eroding sample efficiency)

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

This experiment uses **ablation study** design (= a method where you remove or vary one component at a time to measure its individual impact — named after the scientific practice of selectively removing tissue to study its function):
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
| **K (epochs)** | **Waste data** (discard rollout before extracting full signal) | K = 4-10 | Overfitting to stale rollout data |
| **n_steps** | Too noisy | 128-2048 | **OOM Memory errors** (uses too much RAM) |
| **batch_size** | Too noisy | 32-256 | **OOM Memory errors** (uses too much RAM) |

These "sweet spots" can shift depending on the environment!

---

## The Key Insight: PPO is Relatively Robust

Compared to earlier algorithms (like DQN without target networks), PPO is relatively
robust to hyperparameter choices. The clipping mechanism provides a natural safety net.

**Real-life example:** A car with **ABS** (Anti-lock Braking System — a safety feature that prevents wheels from locking up during hard braking, keeping the driver in control) brakes vs. without:
- Without ABS (DQN): One wrong turn (bad hyperparameter) and you spin out
- With ABS (PPO): The car corrects itself — reasonable hyperparameters all work okay

This robustness is a major reason PPO is the most popular RL algorithm in practice!

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Ablation study** | Change one thing at a time to see its effect |
| **Clip epsilon ε** | Safety boundary — 0.2 is usually best |
| **Learning rate** | **Step size** — how much the network's weights are adjusted after each batch (think of it as the size of each footstep when walking toward a goal). **2.5e-4 to 3e-4** is scientific notation for 0.00025 to 0.0003 — these are dimensionless multipliers, not time values |
| **Update epochs K** | How many times to reuse each batch — 4-10 is standard |
| **Random Seeds** | Each experiment is repeated with different **random seeds** (= the starting number fed to the random number generator, which controls all random choices in training). Using multiple seeds reveals whether results are consistent or just got lucky |

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
