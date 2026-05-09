# Q-Learning Agent for Frozen Lake 🧊

## What Is It? (For Curious Kids!)

Picture a frozen pond with slippery ice. There is a **Start square** and a **Goal square**
with some **Holes** in the middle. If you fall in a hole, you start over!

The ice is slippery, so even if you try to walk right, you might slide up or down instead.
A **Q-Learning agent** is a robot that learns — by trying over and over — how to get from
Start to Goal without falling in!

---

## The Big Idea: A Magic Table of Scores

Q-Learning builds a big table called the **Q-table**. Each row is a square on the ice,
and each column is an action (left, right, up, down). The numbers inside are **scores**:
"How good is it to take this action from this square?"

Every time the robot tries a move:
1. It gets feedback (did it fall? did it reach the goal?)
2. It updates the score in the table using this formula:

> **New Score = Old Score + Learning Rate × (What really happened − What I expected)**

The robot is basically asking: "Was this move better or worse than I thought?"

**Real-life example:** Think of a baby learning to walk. Every time they try a step and fall,
they learn "that step was bad." Every time they succeed, they remember "that worked!" After
many tries, they figure out how to walk. Q-learning does the same thing, but with a table!

---

## What Makes Q-Learning Special: It's Off-Policy!

Here's something clever: when Q-Learning updates its table, it *always assumes it will make
the perfect move next time*, even if during training it sometimes explores random moves.

This means Q-Learning is **off-policy** — it learns about the best strategy even while
doing something different (exploring). It separates "what I do" from "what I learn."

---

## What Our Code Found

We trained for **50,000 episodes** on the slippery 4×4 Frozen Lake:

| Metric | Result |
|--------|--------|
| Greedy evaluation success rate | **73.1%** |
| Milestone target (>70%) | ✓ **PASSED** |

The ice is very slippery, so even the best policy can't win 100% of the time!

The learned Q-table shows the agent figured out: go down and right while avoiding the holes.

---

## Real-Life Examples

- **Self-driving car**: Learning which lanes to take at intersections through trial runs.
- **Recommendation systems**: Learning which movies to suggest based on whether users liked
  previous suggestions.
- **Video game AI**: A character that learns to navigate a maze by trying many paths.

---

## Key Words to Remember

- **Q-table**: The table of "how good is each action in each state"
- **Q(s, a)**: The score for taking action a in state s
- **Reward**: What the agent gets after taking an action (+1 for reaching goal, 0 otherwise)
- **Off-policy**: Learns the optimal strategy even while exploring randomly
- **ε-greedy**: Most of the time do the best known action; sometimes explore randomly
- **Discount factor γ**: How much future rewards are worth (like preferring money now vs later)

The big idea: **Q-Learning builds a "cheat sheet" for every situation, and keeps improving
it until it knows the best move everywhere.**
