# Frozen Lake with a Random Policy 🧊

## What Is Frozen Lake?

Imagine you are playing on a **frozen pond** with your friends.

The ice is mostly safe, but some spots have **holes** — if you step on a hole,
you fall in and the game is over! At one end of the pond is a **present** 🎁.
Your job is to slide from the **start** to the **present** without falling in.

Here is what the frozen lake looks like (4 squares × 4 squares):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (where you begin)
- **F** = Frozen ice (safe!)
- **H** = Hole (fall in, game over 😨)
- **G** = Goal — the present! 🎁

---

## The Tricky Part: Slippery Ice!

On a real frozen pond, when you try to walk *right*, sometimes the ice makes
you slide *up* or *down* instead! That's what makes this hard.

Even if you *want* to go right, the game might slide you somewhere else.
This is called **stochasticity** — a fancy word for "things don't always go
the way you planned."

---

## What Is a Random Policy?

A **policy** is just a plan: "In this situation, I will do THIS action."

A **random policy** means: "I have no plan at all! I will just pick a random
direction every time — up, down, left, or right — like spinning a spinner!"

It's like a baby walking on ice with no idea where the present is.

---

## What Our Code Found

We tried the random policy for **1,000 games**:

| Result | Value |
|--------|-------|
| **Times reached the present** | 11 out of 1,000 (1.1%) |
| **Average steps per game** | 7.5 steps |
| **Fastest game** | 2 steps |
| **Longest game** | 33 steps |

Most of the time, the random walker fell into a hole quickly.
Only 1 in 100 games ended with finding the present!

---

## Why Is This Useful?

Even though the random policy is terrible, it gives us a **baseline** —
a starting point to compare against.

When we later build a *smart* policy (using Q-learning or other algorithms),
we can say: "Our smart agent succeeds 75% of the time — much better than the
random walker's 1%!"

**Real-life example:** Imagine trying to find your classroom in a new school
by randomly turning left or right at every hallway. You might get there
eventually, but it would take a long time! A smart policy is like having a map.

---

## What the Heatmap Shows

In our picture, the **heatmap** shows which squares the random walker visited
most often:

- The **Start** square is visited a lot (every game begins there).
- Squares near the **holes** are visited less (the walker often falls in before
  reaching them).
- The **Goal** is visited very rarely because the random walker almost never
  makes it there.

This tells us something important: the random policy gets stuck near the
beginning and never really explores the whole lake.

---

## Key Words to Remember

- **Policy**: Your plan for what to do in each situation
- **Random policy**: No plan — just pick a random action!
- **Baseline**: A bad result we use for comparison (how much better can we do?)
- **Stochastic**: Things don't always go the way you plan (like slippery ice!)
- **Success rate**: How often did we win? (Here: 1.1% — very low!)

The big idea: **A random policy is a starting point. Real learning means
building a better plan!**
