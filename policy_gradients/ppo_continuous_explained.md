# PPO for Continuous Control: Making BipedalWalker Walk

## Discrete vs. Continuous Actions

So far, every environment we've solved has **discrete** actions:
- CartPole: push LEFT or push RIGHT (2 choices)
- LunarLander: fire nothing / left / main / right (4 choices)

But real-world robots need **continuous** actions:
- A humanoid robot: "how hard to push each joint" (any value from -1 to +1)
- A car: "exactly how much to turn the steering wheel" (any angle from -30° to +30°)
- An arm: "apply exactly 2.3 Newtons in this direction"

**Real-life example:** Typing on a keyboard = discrete (press A, B, C...).
Writing with a pencil = continuous (move hand 2.3 cm to the right, press 40g force...).

---

## The Gaussian Policy for Continuous Actions

For continuous actions, instead of a Categorical distribution (pick from N categories),
we use a **Normal (Gaussian) distribution**:

```
Action ~ Normal(μ, σ)
```

Where:
- **μ (mu, mean)**: The center of the distribution — the action value the network "aims for"
- **σ (sigma, standard deviation)**: The spread — how much randomness / exploration to add

```
        Probability
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Action value
           -1  -0.5   0   0.5   1
                      ↑
                   mean μ
```

**Real-life example:** A skilled archer aims at the center of the target (μ).
Their arrows don't all land in exactly the same spot — there's some spread (σ).
As they practice, they get more accurate (σ decreases) while staying centered on the bullseye.

---

## Our Gaussian Actor-Critic Network

```
State (24 numbers) → [256 neurons] → [256 neurons] →
    ├── Actor: 4 mean values  (μ₁, μ₂, μ₃, μ₄)
    │          + 4 log_std params (shared across all states!)
    └── Critic: 1 value (V(s))
```

The `log_std` (logarithm of the **standard deviation** — a measure of spread or uncertainty)
is a **learnable parameter** — not state-dependent.
This keeps it simple while still letting exploration change during training.

**Why log_std instead of std?** Standard deviation must be positive. Using `log_std` allows
the network to output any real number (positive or negative), then we apply
`exp(log_std)` — the exponential function, which is the inverse of the logarithm — to
recover a guaranteed-positive std. This prevents the std from ever becoming negative or zero.

---

## Computing Log Probability for Continuous Actions

For discrete actions: `log_prob = log(P(action=LEFT))`

For continuous actions, the **Normal distribution** describes a smooth bell-shaped curve
around the mean. A single exact value has probability zero in continuous math, so we use
the curve height at that value, called the **pdf** (probability density function):
```
log_prob = Σᵢ log[Normal(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` means natural logarithm. It turns tiny density values into stable numbers that are
easier for neural networks to optimize. We sum across all action dimensions (4 for
BipedalWalker), because the full action is one 4-number vector.

**Real-life example:** What's the probability of getting exactly 5.732...°C tomorrow?
For continuous weather, you'd look at the Normal distribution curve and see how tall it is
at that exact point. More likely temperatures (near the mean) have higher probability.

---

## BipedalWalker: A Walking Challenge

BipedalWalker-v3 is a 2D robot that must learn to walk without falling:

```
          O (head)
         /│\
        / │ \
       /  │  \
      L   │   R   ← two legs, each with a knee joint
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motors (hip/knee for each leg)
```

**State space (24 numbers):**
- Hull: angle, angular velocity, horizontal velocity, vertical velocity (4 numbers)
- Joints: 4 motors (2 hips, 2 knees) each providing angle and speed, plus 2 ground contact sensors (one for each leg) (10 numbers)
- 10 LIDAR range sensors (distance readings that see the ground ahead) (10 numbers)

**Action space (4 continuous values, each in [-1, 1]):**
The action values control the **torque** (the rotational force applied by the motors) for exactly 4 joints (no actions are applied directly to the hull):
- Leg 1 Hip torque, Leg 1 Knee torque, Leg 2 Hip torque, Leg 2 Knee torque

**Rewards:**
- +300 for reaching the goal (right side)
- -100 for falling over (touching the ground with body)
- Small reward per step of forward progress
- Small penalty for each engine use (reward efficiency)

**Solved when:** Average reward > 300 over 100 episodes

---

## Key Difference from Discrete PPO

Everything is the same EXCEPT:

| | Discrete PPO | Continuous PPO |
|---|---|---|
| **Policy** | Categorical(logits) | Normal(μ, σ) |
| **Sample** | action = sample from {0,1,...,N} | action = μ + σ × noise |
| **log_prob** | log P(action=k) | Σ log Normal(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Not needed | Clamp actions to [-1, 1] |

**Logits** are raw, unnormalized scores for discrete actions. A categorical policy converts
them into probabilities with **softmax** — a function that takes any set of numbers and
squashes them into a valid probability distribution (all values positive, summing to 1).
For example, logits [2.0, 1.0, 0.5] become probabilities [0.59, 0.24, 0.17]. Continuous PPO does **not** use softmax for the action itself,
because the action is not chosen from a fixed menu. Instead, the policy outputs the mean
and standard deviation of a Normal distribution, then samples real-valued torques from it.

**Clamp** means force a value into a valid range. The code uses `action.clamp(-1, 1)` so the
environment never receives a motor command outside its allowed bounds.

**Clip** in PPO means something different: PPO clips the probability ratio inside the loss,
as explained in the [PPO clipping section](./ppo_scratch_explained.md#the-clipping-trick).
Action clamping protects the environment interface; PPO clipping protects the policy update.

---

## Walking from Scratch: What the Agent Learns

**Early training (negative rewards):** The robot flails randomly, falls immediately.
Every episode ends in a crash within seconds.

**Mid training:** The robot discovers that moving legs alternately creates forward progress.
It starts making small, awkward steps — reward becomes less negative.

**Late training:** A smooth, efficient walking **gait** emerges. A gait is a repeated movement
pattern, like alternating left and right steps. The robot adjusts to uneven terrain dynamically by utilizing its LIDAR sensors to adapt its steps in real-time.

**Real-life example:** A baby learning to walk:
1. Falls immediately (negative reward)
2. Takes one step, falls (slightly less negative)
3. Takes a few steps (small positive reward)
4. Walks across the room (large positive reward!)

---

## Why BipedalWalker Needs PPO (Not REINFORCE)

- **BipedalWalker episodes** can be up to 1600 steps (much longer than CartPole!)
- **Rewards are sparse** — forward progress rewards are tiny per step
- **REINFORCE would need** thousands of complete episodes to get useful signal

PPO's n-step updates with [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) let the robot learn from incomplete episodes:
> "Even though I fell after 50 steps, those steps showed SOME forward progress.
> Let me use a 50-step return estimate rather than waiting for episode completion."

---

## Results

After 500 updates (≈ 1 million environment steps):
- The robot makes visible progress from random flailing toward some forward movement
- Consistent improvement in the learning curve
- Full convergence to reward > 300 requires more training (5-10M steps)

The learning curve shows the characteristic "S-curve" of continuous control:
1. Slow initial progress (learning stability)
2. Rapid improvement (gait discovery)
3. Gradual refinement (gait optimization)

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Gaussian policy** | Instead of choosing from a menu, throw a dart at a range of values |
| **μ (mean)** | Where the policy "aims" |
| **σ (std)** | How much randomness / exploration the policy uses |
| **log_std as learnable parameter** | A global exploration rate updated by gradient-based optimization (gradient *ascent* on reward, or equivalently gradient *descent* on the PPO loss) — just like any other network weight |
| **Continuous control** | Controlling real-valued outputs (torques, forces, angles) |

---

## What's Next?

PPO has many **hyperparameters** — settings you choose before training begins (as opposed to
*parameters* like network weights, which are learned automatically). Examples include
`clip_eps`, learning rate, number of epochs, and batch size.

How sensitive is PPO to these choices? `ppo_hyperparams.py` runs experiments
systematically varying each hyperparameter and shows the effect on learning speed and stability.
