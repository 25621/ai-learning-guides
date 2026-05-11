# PPO: Safe and Steady Policy Updates

## The Problem with A2C

Imagine you're learning to balance a broomstick on your finger. After weeks of practice,
you can keep it up for 30 seconds!

Now your coach gives you advice: "Lean your wrist slightly more to the left."

**Good advice → careful change → still balance for 30 seconds ✓**

But what if the coach overreacts and says: "LEAN WAY TO THE LEFT IMMEDIATELY!"
You overcorrect → broomstick falls → you've lost weeks of progress.

This is the A2C problem: **large gradient updates can destroy a good policy**.

**PPO (Proximal Policy Optimization)** is a safety system that prevents this.

---

## The Core Idea: Stay Close to What Was Working

PPO's key constraint:

> **"Don't change the policy too much in a single update."**

Before an update, we have the "old" policy π_old.
After the update, we have the "new" policy π_new.

PPO measures how much the policy changed with the **probability ratio**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1.0: policy unchanged
- r = 1.5: new policy is 50% more likely to take that action
- r = 0.5: new policy is 50% less likely to take that action

**Real-life example:** You're a chef adjusting a recipe.
- r = 1.0: same amount of salt as before
- r = 2.0: double the salt — too extreme!
- r = 0.9: 10% less salt — small, safe change

---

## The Clipping Trick

PPO clips the ratio to stay within [1-ε, 1+ε] (typically ε = 0.2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Let's break this down:

**Case 1: The action was GOOD (A > 0)**

We want to do this action more (r > 1). But we cap how much we increase:
```
if r > 1.2: clip to 1.2, no more incentive to push further
```
This prevents us from swinging TOO far in one direction.

**Case 2: The action was BAD (A < 0)**

We want to do this action less (r < 1). But again, we cap:
```
if r < 0.8: clip to 0.8, no more penalty for going further
```

**Visual:**
```
ε = 0.2, so the safe ratio window is 0.8 to 1.2.

GOOD action (A > 0): increase the action probability, but stop rewarding it after 1.2
ratio r:       0.6      0.8      1.0      1.2      1.4
incentive:      ↑        ↑        ↑        ↑        -
meaning:     too low     ok      old      max     clipped

BAD action (A < 0): decrease the action probability, but stop rewarding it below 0.8
ratio r:       0.6      0.8      1.0      1.2      1.4
incentive:      -        ↓        ↓        ↓        ↓
meaning:     clipped    max      old       ok    too high
```

The `-` marks the flat clipped region. In that region, making the probability ratio even
more extreme does not improve the objective, so PPO has no extra incentive to push farther.

**Real-life example:** A car's speed limiter. You can accelerate, but once you hit 120 km/h,
the limiter kicks in and won't let you go faster. It keeps you safe without stopping
you from moving.

---

## Why This Prevents Catastrophic Updates

A **catastrophic update** is when one large policy change completely destroys everything the
agent has learned — hours of training gone in a single gradient step.

Without clipping: one large gradient step might change the policy drastically.
With clipping: the gradient is zero outside [1-ε, 1+ε], so the policy can only move a little per step.

**Real-life example:** A good surgeon takes small, precise cuts — not large, sweeping ones.
PPO is the "careful surgeon" of RL optimizers.

---

## GAE: Smarter Advantage Estimates

PPO uses **Generalized Advantage Estimation (GAE)** to compute the advantage:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (TD error)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE has a parameter λ (lambda):
- λ = 0: use only one-step TD error (low variance, high bias)
- λ = 1: use full Monte Carlo returns (high variance, low bias)
- λ = 0.95: a good balance between both!

**Real-life example:** Planning a road trip.
- λ=0: only look at the next 5 miles (safe, but might miss a shortcut later)
- λ=1: consider the entire 500-mile journey (more info, but very uncertain)
- λ=0.95: look far ahead but weight nearby roads more heavily ← best balance!

---

## Multiple Epochs: Reusing Data Efficiently

After collecting a batch of experience (rollout), REINFORCE throws it away after ONE update.

PPO reuses each batch for **K epochs** (typically 4-10 passes through the same data):

```
Collect 512 steps × 4 environments = 2048 transitions
Epoch 1: 32 minibatches × update each
Epoch 2: shuffle, 32 more minibatches × update each
Epoch 3: ...
Epoch 4: ...
```

**What is a "minibatch"?** Updating with all 2048 transitions at once is slow and
memory-hungry; updating one transition at a time is noisy. A **minibatch** is a small
chunk in between — here, 2048 ÷ 32 = **64 transitions per minibatch**. We compute one
gradient step per minibatch, so each epoch performs 32 small, stable updates instead of
1 huge one. (This is the same minibatch idea used everywhere in deep learning — see
[mini-batch gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent).)

The clipping ensures these multiple passes don't overshoot — without clipping, multiple
epochs would destroy the policy by pushing it too far!

**Real-life example:** A student has 30 practice problems.
- REINFORCE: do each problem once, learn a little, throw them away
- PPO: do each problem 4 times (different angles each time), clip your changes
  so you don't memorize wrong patterns

---

## The Full PPO Loss

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = clipped policy gradient
L_entropy = entropy bonus (encourages exploration)  
L_critic  = MSE between V(s) and returns
```

Typical coefficients: c₁ = 0.01 (entropy), c₂ = 0.5 (critic)

**Two terms worth unpacking:**

- **Policy gradient** — the "actor" half of the loss. It uses the gradient signal to
  push the policy toward actions with higher advantage and away from actions with lower
  advantage. This is the same core idea introduced in REINFORCE — see the [REINFORCE
  walkthrough](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) for the
  intuition. PPO just adds the clipping wrapper around it.
- **MSE (Mean Squared Error)** — the "critic" half of the loss. The critic V(s) predicts
  the expected return from a state; we compare its prediction to the actual return and
  square the difference: `MSE = mean((V(s) - return)²)`. Squaring punishes large mistakes
  more than small ones and gives a smooth, differentiable signal for training. (Standard
  regression loss — see [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).)

---

## The Results

```
Update  200 | Avg reward: ~120
Update  400 | Avg reward: ~280
Update  800 | Avg reward: ~280-300
```

PPO on CartPole shows steady improvement but tends to plateau around 280-300.
(A **plateau** means the learning curve flattens — reward stops improving even as training
continues. The policy has found a locally good strategy but isn't making further progress.)
This is actually expected — PPO is designed for harder, longer-episode environments.

An interesting observation: **REINFORCE solved CartPole faster!** (500 avg vs 300 avg)

Why? CartPole episodes are short (≤500 steps), so REINFORCE's exact returns are very
accurate. PPO's bootstrapped estimates add unnecessary complexity. PPO truly shines on
environments where waiting for full episodes is impractical (like BipedalWalker).

**What is "BipedalWalker"?** BipedalWalker (specifically `BipedalWalker-v3` in
[Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) is a
classic benchmark RL environment: a 2-legged robot that must learn to walk forward
across uneven terrain without falling. Unlike CartPole's two discrete actions
(LEFT / RIGHT), BipedalWalker has **continuous** actions — four torque values, one for
each leg joint, each a real number in [-1, 1]. Episodes can run for thousands of steps,
which is exactly the regime where PPO's data efficiency and stability pay off.

---

## Key Equations

```
Ratio:      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Clip loss:  L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| **Ratio r(θ)** | How much the policy changed on this action |
| **Clip ε** | The safety boundary — don't change the policy more than this |
| **GAE** | A smart way to estimate advantages by looking ahead multiple steps |
| **Data efficiency** | Each rollout is collected from several parallel environments (decorrelated, stable experience) and then reused for K epochs of minibatch updates — clipping keeps these repeat passes safe |

---

## What's Next?

So far, all our environments have **discrete** actions (push LEFT or RIGHT).

Real robots need to control **continuous** actions — like "apply exactly 0.73 Newtons of force."

`ppo_continuous.py` extends PPO to continuous actions using a **Gaussian policy**,
and tests it on the much harder BipedalWalker-v3 environment!
