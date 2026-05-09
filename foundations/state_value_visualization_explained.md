# State-Value Functions 🗺️

## What Is a "State"?

Think about playing a board game. At any moment, you are standing on *one*
square of the board. That square is your **state** — it's where you are
right now.

In our 4×4 grid game, there are 16 squares (states). Each square is a place
the agent can stand on.

---

## What Is a "Value"?

Now here's the magic question: **"If I'm standing on this square right now,
how much treasure can I expect to collect before the game ends?"**

That answer is the **value** of that state!

A square with a **high value** means: "This is a great spot — I'll probably
collect a lot of treasure from here!"

A square with a **low value** means: "Uh oh — from here, things usually go
badly."

**Real-life example:** Imagine you're playing hide-and-seek. If you're hiding
behind a big tree (a great spot), your chance of winning is high — that's a
high-value state! If you're hiding in the middle of an empty room, you'll
probably get found — that's a low-value state.

---

## Our Grid World

Here's the game board we used:

```
S  .  .  .      S = Start
.  H  .  H      H = Hole (reward -1, game ends)
.  .  .  H      G = Goal (reward +1, game ends)
H  .  .  G      . = Empty safe square
```

- If you reach **G** (Goal): you get **+1 point** 🎉
- If you step on **H** (Hole): you get **-1 point** 😢
- Other steps: **0 points**

We used γ (gamma) = 0.99, which means future rewards count almost as much
as immediate rewards. (A candy tomorrow is almost as good as candy today!)

---

## Two Different Plans (Policies)

We tested two policies and computed the value of every square for each:

### Policy 1: Uniform Random
Randomly pick up, down, left, or right with equal chance.

```
Values (Uniform Random Policy):
-0.912  -0.932  -0.912  -0.942
-0.929   (H)   -0.898   (H)
-0.901  -0.801  -0.696   (H)
 (H)   -0.630  -0.104   (G)
```

Almost everywhere is **negative** — the random policy falls into holes so
often that being anywhere is pretty bad!

---

### Policy 2: Biased Toward Goal
Prefer moving right and down (toward the goal), but still sometimes go other
directions.

```
Values (Biased-Toward-Goal Policy):
-0.838  -0.895  -0.814  -0.961
-0.798   (H)   -0.665   (H)
-0.595  -0.143  -0.213   (H)
 (H)    0.254   0.673   (G)
```

Now the squares near the **Goal** have **positive values** (0.254 and 0.673)!
The smart policy makes those squares good places to be.

---

## What the Colors in Our Picture Mean

In our visualization:
- **Green squares** = high value (great places to be)
- **Red squares** = low value (avoid these!)
- **Yellow squares** = somewhere in between

You can see the **gradient** — values get greener as you move toward the Goal
and redder near Holes.

---

## Why Do We Care About Values?

Values are the *foundation* of reinforcement learning! Once you know the value
of every state, you can make smart decisions:

> "I'm at square A. I can go to square B (value = 0.5) or square C (value = -0.3).
> I'll go to B — it has a higher value!"

This is exactly how many RL algorithms (like Q-learning) learn to make good
decisions without being told the rules.

**Real-life example:** Imagine you're picking which line to stand in at the
grocery store. Each line is a "state." The value of that state is how quickly
you'll get through checkout. You look at the lines (observe states) and pick
the one with the highest value (shortest wait + fewest items).

---

## How We Computed the Values

We used **Iterative Policy Evaluation**, which works like this:

1. Start: guess all values are 0.
2. Update: for each square, calculate what the value *should* be based on
   where the policy takes you next.
3. Repeat until the values stop changing (converge).

Mathematically: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(next_state)]**

In plain English: "The value of this square = the average reward I'll get
right now + a little bit of the value of wherever I end up."

---

## Key Words to Remember

- **State**: Where you are right now (one square on the board)
- **Value V(s)**: Expected total reward starting from state s
- **Policy**: Your plan for what to do in each state
- **Discount factor γ**: How much you care about future rewards (0.99 = a lot!)
- **Policy Evaluation**: Computing values for every state under a given policy

The big idea: **Some places are better than others — and the value function
tells you exactly how good each place is!**
