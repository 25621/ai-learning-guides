# Behavioral Cloning (BC) 🐒

## What Is It?

Imagine you want to learn to play tennis. You watch hundreds of hours of
recorded Wimbledon matches and simply **copy what the players do**. You
don't think about whether their shot was the *best* shot — you just match
your body position to theirs and swing the same way.

That's behavioral cloning. **No reward. No planning. Just imitation.**

In RL terms: take the dataset of `(state, action)` pairs and train a
neural network to predict the action from the state, exactly like an
image-classification model predicts cat-vs-dog. The "label" is whatever
action the data collector took.

---

## How It Differs From "Real" Offline RL

| Approach | Uses rewards? | Can beat the data? |
|----------|---------------|---------------------|
| **BC**   | ❌ no         | ❌ no — at best, matches the average data quality |
| **CQL** (and friends) | ✅ yes | ✅ yes — can stitch good transitions out of mixed data |

BC is the "supervised-learning view" of RL. It is incredibly simple, often
surprisingly strong, and the universal baseline. **If an offline RL
algorithm can't beat BC on the same dataset, it has done nothing.**

---

## Real-Life Examples

- **Learning to drive from dashcam footage.** Look at the road, predict
  the steering wheel angle the human used. (Self-driving teams have done
  exactly this — see ALVINN, NVIDIA's PilotNet.)
- **Apprentice copying a master chef.** "Whatever the chef does, I do."
  Works great if the chef is great; produces a bad chef if the chef is
  bad.
- **GitHub Copilot.** Auto-complete is trained to predict "what code
  would a human type next?" — pure imitation of source-code logs.
- **Mimicking your older sibling.** Kids do this for years before they
  start reasoning about *why* the older sibling does what they do.

---

## The Math (One Line)

For each `(s, a)` in the dataset, minimise:

```
loss = -log π(a | s)        (cross-entropy)
```

That's it. The policy `π` is just an MLP that outputs action logits;
training is identical to MNIST.

---

## What Our Code Does

The script `behavioral_cloning.py`:

1. **Loads all four datasets** built by `d4rl_dataset.py`
   (`random`, `medium`, `expert`, `medium-replay`).
2. For each dataset, **trains a separate BC policy** for 10,000 gradient
   steps of cross-entropy. The reward column is completely ignored.
3. Every 2,500 steps, **evaluates** the current policy by rolling it out
   greedily in the real CartPole-v1 environment (20 episodes, averaged).
4. Plots:
   - A bar chart: final BC return per dataset.
   - A learning-curve chart: how each BC variant climbs over training.

---

## What You Should See

A typical run prints:

```
Final evaluation returns:
  BC on random          ->    ~20  ± a few   (≈ random play)
  BC on medium          ->   ~150  ± large   (≈ the medium policy)
  BC on expert          ->   ~480  ± small   (≈ the expert policy)
  BC on medium-replay   ->    ~60  ± large   (≈ the AVERAGE of mixed data)
```

The bar plot makes the story obvious: **BC's return tracks the dataset's
average return.** It cannot go above that ceiling because it has no way to
prefer the "good" parts of a mixed dataset over the "bad" parts — both are
equally valid imitation targets.

That's the punchline: **BC inherits the data's ceiling.**

---

## BC vs CQL — The Cleanest Comparison

On the **medium-replay** dataset (the most realistic, mixed-quality case):

| Method | Approx final return | Why? |
|--------|--------------------:|------|
| BC     | ~60   | Imitates the *average* of failed early runs + later good ones |
| CQL    | ~400+ | Uses rewards to prefer high-Q transitions; stitches a good policy out of mixed data |

So CQL **beats the data**, BC **matches the data**. That's the whole
reason offline RL is a research field and not just "do imitation
learning". When data is mixed quality (which real logs always are),
reward-aware methods recover more.

On **expert** data the comparison flips: BC matches expert (~480) and CQL
struggles to do meaningfully better because there's nowhere to go.

This is the famous "data-quality vs algorithm" trade-off:

```
                            EXPERT  data  →  BC wins, CQL ties
   Algorithm sophistication  ↑         
                            MIXED   data  →  CQL clearly beats BC
                            
                            RANDOM  data  →  Everybody fails; need exploration
```

---

## Where BC Lives in Modern RL

- **Pre-training for online RL.** Many modern systems (RT-1, Voyager,
  game-playing bots) start with BC on demonstrations, then fine-tune
  online with PPO/SAC.
- **RLHF.** Step 1 of InstructGPT is supervised fine-tuning — pure BC on
  human-written responses. PPO + reward model come later.
- **DAgger (Ross et al., 2011).** A clever extension: query the expert
  again on states the BC policy visits, fixing the compounding-error
  problem. Often beats vanilla BC by a lot.
- **Decision Transformer (Chen et al., 2021).** A "smart" BC that
  conditions the action prediction on a desired *return-to-go*,
  essentially turning offline RL back into next-token prediction.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Imitation learning** | Umbrella term for "copy the demonstrator"; BC is the simplest member |
| **Compounding error** | A small BC mistake takes you to states the dataset never saw, where mistakes compound |
| **Demonstration data** | Trajectories produced by an expert, used as the BC training set |
| **Data ceiling** | BC's return is bounded by the average return in the dataset |
| **DAgger** | An interactive fix for compounding error |

---

## One-Sentence Summary

> **Behavioral cloning is just supervised learning on (state, action)
> pairs — strong when the data is good, helpless when the data is mixed.**
