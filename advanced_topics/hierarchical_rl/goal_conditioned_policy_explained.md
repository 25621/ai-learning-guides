# Goal-Conditioned Policy

## The Big Idea: One Policy to Rule Them All {#core-idea}

Imagine you are a delivery driver. You don't need a completely different skill set for every address. You know how to drive, read a map, and navigate traffic — you just plug in *today's destination* and go.

A **goal-conditioned policy** works the same way. Instead of training one agent that can only go to one fixed goal, we train a single agent that accepts any goal as an input and figures out how to get there.

---

## How It Differs From Standard RL {#how-it-differs}

In standard RL (as covered in the earlier phases of the curriculum), the reward function is baked in: "reach cell (7, 7), get +1." The agent learns exactly one thing: how to reach *that* cell.

In goal-conditioned RL, the reward depends on whether the agent reaches *whatever goal it was given this time*. The policy learns:

> **"Given where I am and where I want to be, what should I do?"**

The goal travels *with* the agent, like a destination typed into a navigation app.

---

## The Sparse Reward Problem {#sparse-reward-problem}

Here is the catch: learning from sparse rewards (only +1 at the goal, 0 everywhere else) is brutally hard. Most attempts fail — the agent wanders randomly, never bumps into the goal, and the network gets nothing useful to learn from.

Imagine trying to learn to throw a dart blindfolded. You throw a thousand times and always miss. After a thousand failures, you still have no idea what "a good throw" feels like.

This is where **Hindsight Experience Replay (HER)** comes in.

---

## Hindsight Experience Replay: Failing Forward {#hindsight-experience-replay}

HER's trick is beautifully simple. After a failed episode, HER asks:

> *"Even though you didn't reach your goal… where did you actually end up?"*

It then **replays that same episode**, but pretends the agent's actual final position **was** the goal all along. Suddenly, a failed episode becomes a successful one — for a different goal.

It's like a failed basketball player who keeps shooting for the hoop and missing. HER would say: "Okay, you hit the left wall every time. Congratulations — you're great at hitting the left wall! Let's log those throws as successful left-wall-hitting attempts." Over time the player builds up skill in hitting *any* target, and eventually transfers that to the real hoop.

This turns thousands of "failures" into a rich library of *successful* navigations to many different spots. The agent learns to reach all of them, which generalizes to the real target.

---

## The Real-Life Analogy: Toddler Learning to Stack Blocks {#real-life-analogy}

A toddler trying to put a block in a bucket misses constantly. But each "miss" lands the block *somewhere*. If you replay each miss as "you were trying to put it *right there* — and you did it!", the toddler builds fine motor skill across the whole table. Soon they can place a block anywhere — including in the bucket.

---

## What Our Code Does {#what-our-code-does}

The script `goal_conditioned_policy.py` runs in a **7x7 maze** with walls. At the start of each episode, a random goal cell is chosen. The agent must find it.

The policy takes two inputs at every step:
1. Where the agent currently is
2. Where it wants to go

After each episode (successful or not), HER generates several additional synthetic "successes" by relabeling the actual positions visited as alternative goals.

Training runs for 3,000 episodes with a decaying exploration rate — the agent explores more at first and then increasingly trusts what it has learned.

---

## What the Charts Show {#what-the-charts-show}

![Goal-Conditioned Policy Results](outputs/goal_conditioned_policy.png)

**Left — Success Rate Over Training:** Each episode is either a success (reached the goal) or failure. The curve climbs steadily as the agent's universal navigation skill improves. By the end, the agent reaches any goal almost every time.

**Right — Goal Success Rate Heatmap:** After training, we test the agent on every possible goal cell and color each cell by how often the agent reaches it. Green means the agent reliably reaches that spot; red means it still struggles. A well-trained agent shows mostly green across the whole maze.

---

## Where This Shows Up in the Real World {#real-world-applications}

| Application | The "goal" |
|-------------|------------|
| Robot arm reaching | Target 3-D position |
| Self-driving car | GPS coordinate |
| Language model assistant | User's instruction |
| Video game Non-Player Character | Any waypoint on the map |

Goal-conditioned policies are one of the key building blocks for HIRO (Hierarchical RL with subgoals) — the high-level manager picks a subgoal, and the low-level worker is exactly this kind of goal-conditioned policy.

---

## One-Sentence Summary {#summary}

> **A goal-conditioned policy is one agent that can navigate to any destination — and HER makes learning from failure possible by pretending every missed shot was aimed at wherever it landed.**
