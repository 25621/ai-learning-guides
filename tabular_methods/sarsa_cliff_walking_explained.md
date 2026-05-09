# SARSA for Cliff Walking 🏔️

## What Is It? (For Curious Kids!)

Imagine a **very long hallway** with a **terrible cliff** along one edge. If you fall off the
cliff, you have to go all the way back to the start! Your goal is to walk from one end to the
other as quickly as possible, without falling.

**SARSA** is a robot that learns to walk this hallway by practicing. It learns to take a
*safe path* that avoids the cliff — even if it's a little longer — because it knows it might
accidentally slip close to the edge when exploring!

---

## The Big Idea: Learning From What You Actually Do

SARSA stands for: **S**tate → **A**ction → **R**eward → **S**tate → **A**ction

These are the five pieces of information SARSA uses to learn:

1. **S** — Where am I right now? (current state)
2. **A** — What action did I actually take?
3. **R** — What reward did I get?
4. **S** — Where did I end up?
5. **A** — What action am I *actually going to take next*?

The last "A" is what makes SARSA special! It updates using the action it will *actually take
next* (even if that's a random exploratory move), not the perfect ideal action.

**Real-life example:** Think about learning to ride a bike. If you know you sometimes wobble
randomly (exploration), you stay a little further from parked cars — because you know your
wobbly self might swerve! SARSA does this: it learns a safe path because it accounts for
its own random mistakes.

---

## The Cliff Walking Map

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← CLIFF ← ← ← ← ←
```

- **S** = Start (bottom-left)
- **G** = Goal (bottom-right)
- **C** = Cliff — stepping here = -100 reward, restart!
- Every other step = -1 reward

---

## What Our Code Found

After training SARSA for 500 episodes:

| Result | Value |
|--------|-------|
| Final 50-episode average reward | **-21.6** |
| Optimal (risky) path reward | -13 |

SARSA's learned policy goes **along the top of the grid** — a safe detour! It costs a few
extra steps (-21 instead of -13), but almost never falls off the cliff during training.

---

## Real-Life Examples

- **Nurse administering medicine**: Takes the proven safe protocol (safe path) even if a
  slightly faster method exists, because small mistakes (exploration) could be dangerous.
- **Airline pilots**: Follow strict checklists (safe paths) even when shortcuts might seem
  faster, accounting for human error.
- **Learning to cook**: Start with well-tested recipes (safe), not risky shortcuts.

---

## Key Words to Remember

- **On-policy**: Learns about the policy it's actually using (including its random mistakes)
- **SARSA update**: Uses the *actual* next action, not the theoretically best one
- **Safe path**: A longer path that avoids danger, accounting for exploration mistakes
- **TD control**: Updating values after every single step (not waiting for the whole episode)

The big idea: **SARSA is honest — it learns from what it actually does, not what it wishes
it would do. This makes it cautious and safe near danger!**
