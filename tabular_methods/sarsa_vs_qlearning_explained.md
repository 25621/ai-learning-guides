# SARSA vs Q-Learning: Safe vs Optimal Paths 🐢 vs 🐇

## What Is It?

Two robots both need to walk along a **cliff edge** to reach the goal. Both robots
are still *learning* and sometimes make random moves (oops!).

- 🐢 **SARSA robot**: "I know I sometimes wobble... so I'll walk far away from the cliff
  to be safe, even if it takes longer."
- 🐇 **Q-Learning robot**: "The shortest path hugs the cliff — let's go! (Falls off
  sometimes while learning, but eventually learns the best route.)"

Both robots are smart, but they make a **different tradeoff**: safe-but-slower vs
optimal-but-risky-while-learning.

---

## The Key Difference: What "Next Action" Do You Use?

When updating scores after each step, both algorithms ask:
> "What is the value of the *next state*?"

| Algorithm | Uses the next action... | On-policy? |
|-----------|------------------------|------------|
| **SARSA** | ...that I will *actually take* (maybe random!) | Yes |
| **Q-Learning** | ...that is *theoretically best* (always greedy) | No |

**Real-life example:** Two kids learning to ride bikes.

- **SARSA kid**: Stays close to the grass because *they know* they sometimes wobble randomly.
  They're planning for their actual wobbly self.
- **Q-Learning kid**: Practices in the middle of the path because they're imagining a perfect
  future self who never wobbles. They fall sometimes now, but learn the best path faster.

Both kids eventually learn — but during training, the SARSA kid falls less!

---

## What Our Code Found

Both algorithms ran for **500 episodes** on Cliff Walking with ε=0.1 (epsilon, pronounced
"EP-suh-lon"; here it means 10% random moves):

| Metric | SARSA | Q-Learning |
|--------|-------|------------|
| Avg reward during training (last 50 ep) | **-19.7** | **-51.0** |
| Greedy evaluation (no exploration) | -17 | **-13** |

- **During training**: SARSA gets **much better rewards** because it avoids the cliff
  (accounting for its own random moves)
- **After training** (pure greedy): Q-Learning finds the **shorter optimal path** (-13)!

As ε (epsilon) shrinks toward 0, both algorithms converge to the same optimal policy.

---

## Visual Summary

```
SARSA path (during training):     Q-Learning path (greedy, after training):
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
     (safe detour, top rows)                 (optimal, hugs cliff)
```

---

## Real-Life Examples

- **New surgeon vs experienced surgeon**: New surgeon (SARSA) stays far from risky
  techniques while learning. Experienced surgeon (Q-Learning greedy) uses the most efficient
  technique after mastering it.
- **City driving vs highway route**: SARSA-like planning takes safer residential streets;
  Q-Learning finds the optimal but narrow highway.
- **Student studying**: SARSA-student sticks to well-understood topics during practice.
  Q-Learning-student pushes to the hardest problems (fails more) but learns optimal strategy.

---

## Key Words to Remember

- **On-policy** (SARSA): Learns about what you *actually do*, including random exploration
- **Off-policy** (Q-Learning): Learns about the *best possible* behavior separately from
  what you actually do
- **Safe path**: Longer route that avoids danger, used when exploration causes accidents
- **Optimal path**: Shortest/highest-reward route, found when no exploration happens
- **Exploration-exploitation tradeoff**: The balance between trying new things and using
  what you know

The big idea: **SARSA is safer during training (on-policy), Q-Learning finds the optimal
path faster (off-policy). Which is better depends on whether falling off the cliff matters!**
