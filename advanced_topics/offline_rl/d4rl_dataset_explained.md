# D4RL Benchmark Datasets 📦

## What Is It?

Imagine you want to teach a robot to flip pancakes. Letting it practise on a
real stove for a month would be slow, dangerous and expensive. But you have
ten years of recorded video of chefs flipping pancakes (some good, some bad,
some random). Can you teach the robot from *just that data*, without ever
letting it touch a real pan?

That's **offline reinforcement learning**. The agent learns from a fixed
dataset of past experience — no live environment. The hardest part is that
the agent can never *try out* what it learned until the very end.

To make this fair to study, the research community needed a *standard
dataset*. That's **D4RL**: a collection of pre-recorded transitions for
classic control tasks, released by UC Berkeley in 2020. Every paper trains
on the same bytes, so results are comparable.

---

## What's In a D4RL Dataset?

For each task, D4RL ships **four quality levels**:

| Level | Where the data comes from | Why it matters |
|-------|---------------------------|----------------|
| **random**        | A policy that picks actions uniformly at random | Worst case: can you still learn something useful? |
| **medium**        | A partially-trained policy (about half of expert score) | Realistic: most logged data is mediocre |
| **expert**        | A near-converged policy | Best case: can you match the source policy? |
| **medium-replay** | The *entire replay buffer* used to train the medium policy | Mixed: contains early failures AND later successes |

The "medium-replay" level is the most interesting. It mixes **bad and good**
transitions — exactly what a real-world log looks like (a robot's first
fumblings AND its later refined behaviour, all in one bucket).

---

## Real-Life Examples of Offline Datasets

- **Medical records.** Years of (patient_state, treatment, outcome) tuples.
  You can't randomise treatments on living people, but you can learn a
  better policy from the log.
- **Customer service chat logs.** Millions of (user_message, agent_reply,
  satisfaction) records. Train a better assistant without bothering more
  users.
- **Autonomous-driving fleet data.** Every Tesla / Waymo car uploads its
  drives. The fleet is a giant medium-replay dataset.
- **Recommender systems.** Click logs from last year are a frozen dataset:
  you can't re-show the same ads to the same users.

In all four cases, **you cannot ask the environment for a new sample.** The
dataset is what you have. Forever.

---

## What Our Code Does

The real D4RL datasets are recorded on MuJoCo locomotion tasks
(HalfCheetah, Hopper, Walker2d, Ant). MuJoCo is heavy to install, so we
recreate the **same four-level structure on CartPole-v1** — the standard
beginner environment from earlier phases. The lessons transfer directly.

The script `d4rl_dataset.py`:

1. **Trains a DQN** on CartPole until it solves the task (return ≥ 475).
2. **Snapshots two checkpoints** along the way:
   - "medium" — the moment the recent return crossed 150
   - "expert" — the moment the recent return crossed 475
3. **Snapshots the medium-policy's full replay buffer** — every transition
   it ever saw. That's our "medium-replay" dataset.
4. **Rolls out three new policies** for 10,000 transitions each:
   - `random`   — uniform random
   - `medium`   — the medium checkpoint + ε=0.10 noise
   - `expert`   — the expert checkpoint + ε=0.02 noise
5. **Saves four `.npz` files** in `outputs/`, each with arrays
   `obs / action / reward / next_obs / terminal`.

These four files are the inputs to `cql.py` and `behavioral_cloning.py`.

---

## What You Should See When You Run It

A plain-text summary printed to the console and saved to
`outputs/d4rl_summary.txt`:

```
dataset         |   N    |  mean return  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Plus a histogram (`outputs/d4rl_returns.png`) showing how the four datasets
overlap. The key features to notice:

- **Random** clusters around 20 (the average length of a random CartPole episode).
- **Expert** clusters at the 500 ceiling.
- **Medium** sits in between, with high variance.
- **Medium-replay** has a long left tail — it remembers the early failed runs.

---

## Why The Dataset Matters

Whichever dataset you train your offline algorithm on, you are putting a
*ceiling* on what's possible:

- **From `expert`** — even a dumb algorithm (behavioural cloning) can do well,
  because all the data is good.
- **From `random`** — you need a smart algorithm that can *stitch together*
  rare good transitions. BC will fail completely.
- **From `medium-replay`** — the most realistic and the most interesting.
  Good algorithms (like CQL) can sometimes **beat the average data quality**
  because they extract structure from mixed signals. Dumb algorithms
  (BC) regress to the mean.

We'll see exactly this story in the next two scripts.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Offline RL**         | Train from a fixed dataset; no environment interaction allowed |
| **Behaviour policy**   | The policy that *produced* the dataset |
| **Dataset quality**    | How good the behaviour policy was (random / medium / expert) |
| **Replay buffer**      | The full history of transitions seen during a training run |
| **Distribution shift** | The trained policy wants to take actions the dataset never recorded |

---

## One-Sentence Summary

> **D4RL freezes RL into a supervised-learning-style benchmark: same bytes
> for everyone, no environment cheating, may the best algorithm win.**
