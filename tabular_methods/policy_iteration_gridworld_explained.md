# Policy Iteration for GridWorld 🗺️

## What Is It? (For Curious Kids!)

Imagine you are playing a board game on a **4×4 grid** (like a tiny chessboard). You start
in one corner and need to reach the other corner. Every step costs 1 point (you don't want
to waste steps!), and reaching the goal earns you nothing extra — you just want to get there
as fast as possible.

**Policy Iteration** is how a computer figures out the **best moves for every square**
on the board — all at once!

---

## The Big Idea: Two Steps, Over and Over

Think of it like cleaning your room with a helper:

1. **Step 1 — Figure out how good each square is (Policy Evaluation)**
   Your helper walks around every square and writes down: "If I follow the current plan, how
   many steps will it take me to reach the exit from here?" They do this again and again until
   the numbers stop changing.

2. **Step 2 — Improve the plan (Policy Improvement)**
   Now you look at each square and ask: "Is there a better direction I could go from here?"
   If yes, update the plan!

Repeat Steps 1 and 2 until the plan stops changing — that's the **optimal policy**!

**Real-life example:** Imagine finding the fastest route to school. First you guess a route
and time it (Step 1). Then you look at each street corner and ask "is there a shortcut from
here?" (Step 2). You update your route and repeat until you can't find any more shortcuts!

---

## What Our Code Found

Our 4×4 GridWorld has two terminal states (corners), and the agent pays -1 per step.
Policy iteration converged in just **4 rounds** (139 total evaluation sweeps):

```
State Values V(s):       Optimal Policy:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**The values make perfect sense!** Squares next to a terminal have value -1 (one step away).
Squares two steps away have value -1.9 (= -1 + 0.9 × -1), and so on.

---

## Real-Life Examples

- **GPS navigation**: Figuring out the best turn at *every* intersection on the map.
- **Elevator control**: Which floor should the elevator go to when it has multiple requests?
- **Factory robot**: Planning the most efficient path around a warehouse grid.

---

## Key Words to Remember

- **Policy**: The plan — what action to take in each state
- **Value Function V(s)**: How good it is to be in state s (higher = closer to goal)
- **Policy Evaluation**: Computing how good the current plan is
- **Policy Improvement**: Making the plan better using the value function
- **Optimal Policy**: The best possible plan — can't be improved further

The big idea: **You don't need to try every possible plan! Just keep improving the current
one, and you'll find the best plan in very few rounds.**
