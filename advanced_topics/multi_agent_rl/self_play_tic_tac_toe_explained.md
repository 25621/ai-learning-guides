# Self-Play: Teaching an Agent by Letting It Play Itself ♟️

## What Is Self-Play?

Imagine a kid who wants to get really good at chess but has no one to play.
So she plays herself. Left hand vs right hand. Every game, *both* sides try
to win. Every game, *both* sides learn what worked.

That is **self-play**: a single agent acts as both players, and every move
becomes a lesson for whoever moves next. No teacher, no expert opponent.
Just a learner that is also its own ladder.

Self-play sounds like a trick — surely you need a real opponent? — but it is
the engine behind the most famous RL milestones of the last decade:
**AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. They all use
self-play. The reason is simple: as you improve, your opponent improves by
the same amount. The challenge always matches your skill.

---

## Why It Works

Three things make self-play special:

1. **Endless opponents.** You never run out of games. The opponent is always
   present and free.
2. **Curriculum that grows with you.** A beginner can only play other
   beginners. As you get better, so does your shadow — automatically.
3. **Symmetry.** In a zero-sum game (one player's win is the other's loss),
   one set of Q-values describes both sides; you just flip the sign when it
   is the other player's turn. So a *single* Q-table can teach itself.

Tic-tac-toe is the perfect testbed: small enough to fit in a dictionary, but
complex enough that randomly picking moves will almost always lead to a loss against a strategic player.

---

## A Real-Life Analogy

- **Practicing tennis against a wall.** You can't lose to a wall, but you
  can practice your serves. Self-play is doing this on both ends — you are
  the wall *and* the player, and you switch back and forth.
- **A debate club that argues both sides.** Better debaters emerge from
  always defending the opposite view to whatever they personally believe.
  Every argument trains both attack and defense.
- **AlphaGo Zero.** It learned from zero human games. Starting from random
  moves it played millions of games against itself; in a few days it was
  better than every previous Go program, including the one that beat Lee
  Sedol.

---

## What Our Code Does

We learn one Q-table for the *current player to move*:

```
Q[(board, player_to_move)][action] = expected return for that player
```

The training loop is:

1. Start an empty board. `player = X`.
2. Both players act with the **same agent**, using ε-greedy.
3. After each game, walk backwards through every (board, player, action)
   triple in the history and apply the Q-learning update.
4. The reward flips sign across turns: if X wins, every move X made gets
   +1 (or bootstraps value from a future winning state); every move O made gets -1.
5. We slowly decrease (decay) our exploration rate (ε) from 0.2 → 0.02, so the agent commits to its best play
   late in training instead of trying random moves.

Every 2,500 episodes we evaluate the agent against a **random opponent**
(we freeze the learning process so no new updates are made to the Q-table during evaluation, and both sides play greedily). The agent should win or draw
~100% of those games after enough self-play.

### What you should see

After 50,000 self-play episodes:

| Match-up | Expected result |
|----------|-----------------|
| Trained agent vs Random opponent (1000 games) | **~95-99% wins or draws**, virtually 0% losses |
| Trained agent vs Itself (200 greedy games) | **All 200 draws**. Tic-tac-toe is a game that always ends in a tie (draw) if both players play perfectly. The fact that self-play draws every game is a sign of convergence. |

The plot `outputs/self_play_tic_tac_toe.png` shows the agent's win/draw/loss
fractions versus a random opponent over time:
- Win rate starts ~60% (when both players play randomly, the first player has an inherent advantage because they get to place more markers on the board, leading to a baseline win rate of about 60% for player X).
- Climbs to >90%.
- Loss rate falls to nearly 0%.

The script also prints a sample game move-by-move at the end so you can see
the agent play.

---

## Watch Out For These Subtleties

- **Sign flips matter.** A common bug: forgetting that "the opponent
  maximising their value" means *minimising ours* in the bootstrap target.
  The update in our code uses `target = reward - gamma * max(Q[next, opponent])`.
- **Symmetry is not exploited here.** A real implementation would canonicalise
  boards (meaning they would rotate or reflect any board state into a standard, unique 'normal form' so the agent recognizes identical board situations) to share Q-values across 8
  symmetries. We skip this — the state space is small enough to brute-force.
- **The Q-table grows.** After 50k self-play games you will see a few
  thousand state-player keys. That is fine here; for chess or Go you would
  need a neural network instead, which is why **AlphaZero replaces the
  table with a CNN + MCTS**.

---

## Where Self-Play Breaks

- **Non-zero-sum games.** "Both sides happy" is incompatible with
  symmetrical play; you cannot just flip a sign.
- **Asymmetric roles.** If "attacker" and "defender" have different action
  spaces, you need two separate networks.
- **Strategy cycling.** Pure self-play can get stuck in
  rock-paper-scissors-like cycles. AlphaStar fixed this by keeping a large
  *pool* (or "league") of saved past versions of the agent and picking
  opponents from that pool at random, so the agent learns to beat many
  different playstyles rather than just the current one.
- **Reward hacking.** Self-play makes both sides smarter, but only at the
  game *as you defined it*. If your reward system has unintended loopholes (like rewarding a player just for surviving longer instead of winning), both sides will mutually exploit the loophole, leading to bizarre, unhelpful behavior instead of mastering the actual game.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Self-play**      | Same agent plays both sides of a game |
| **Zero-sum**       | One player's gain = the other's loss |
| **Symmetry**       | One Q-table can serve both sides if you flip signs |
| **Population play**| Self-play with *many* past versions of yourself as opponents (AlphaStar) |
| **Curriculum**     | A natural difficulty progression — self-play gets it for free |
| **MCTS**           | Monte-Carlo Tree Search — the planning algorithm AlphaZero pairs with self-play |

---

## One-Sentence Summary

> **Self-play turns improvement into its own ladder: every time you get
> better, your opponent does too — automatically.**

This idea, scaled up with **neural networks** (brain-inspired mathematical
functions that learn patterns from data) and tree search, beat the best humans
at Go, chess, shogi, Dota 2, and StarCraft.
