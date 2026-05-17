# Dyna-Q: Learning Faster by Imagining 🧠

## What Is It? {#what-is-it}

Imagine a kid named Mia learning to navigate her new school. Every day she walks
the hallways and discovers new things: "The library is past the cafeteria,"
"Mr. Smith's room is upstairs near the stairwell."

A **plain Q-learning** student only learns from what she does *today*. If today
she just walked from class to the cafeteria, she only updates her memory about
that one path.

A **Dyna-Q** student is different. After every real walk, she sits down for a
minute and **replays in her head** several past walks she remembers. Each replay
strengthens her mental map. After a few weeks she knows the school inside-out —
not because she walked more, but because she **thought more about what she
already saw**.

That is exactly what Dyna-Q does for an RL agent: it learns from real
experience **and** from imagined experience drawn from a model it builds along
the way.

---

## The Three Ingredients {#the-three-ingredients}

Dyna-Q is "Q-learning + model + planning". One real step does **three** jobs:

1. **Direct RL** — the usual Q-learning update from `(s, a, r, s')`.
2. **Model learning** — write down: "When I did *a* in *s*, I got *r* and ended in *s'*."
3. **Planning** — pick *n* random `(s, a)` pairs from the model's memory and do
   *n* more Q-learning updates, **pretending** those steps just happened.

That third step is the magic. With `n = 50`, every real step in the world causes
**51 updates** to the Q-table. The agent learns ~50x faster — in real-step terms —
than a pure Q-learner.

---

## A Picture of the Loop {#a-picture-of-the-loop}

```
                   ┌────────────────────────────────────┐
                   │                                    │
   real world  ──► take action a ──► observe (r, s')    │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Q-learning      Model[s,a] ← (r,s')   Planning ─┘
         update                            (n imagined updates)
```

The model is just a lookup table:
`(state, action) → (reward, next_state)`. Cheap to build, cheap to query.

---

## Real-Life Examples {#real-life-examples}

- **Chess study.** Grandmasters spend hours replaying their own games and
  master games in their heads. Every replay is "planning" — extra learning from
  experiences that already happened.
- **A musician practising scales.** After playing a tricky bar once, they
  mentally rehearse it ten more times before moving on. The fingers are not
  moving, but the brain is updating.
- **A self-driving car.** While idling at a red light it re-plays the last
  hundred lane-changes in simulation to fine-tune its policy without burning
  tyres.

---

## What Our Code Does {#what-our-code-does}

We use the classic **Dyna Maze** ([Sutton & Barto, Figure 8.2](http://incompleteideas.net/book/the-book.html)): a 6×9 grid with
some walls, a start `S` in the middle-left, and a goal `G` in the top right.

We run three variants, each averaged over 30 random seeds:

| Setting | Planning steps per real step | Meaning |
|---------|------------------------------|---------|
| `n = 0` | 0 | plain Q-learning |
| `n = 5` | 5 | a little imagined practice |
| `n = 50` | 50 | a lot of imagined practice |

The script reports the **average number of real steps per episode** as
training progresses. Fewer steps means the agent has learned a more direct
path to the goal.

### What you should see when you run it {#what-you-should-see-when-you-run-it}

The shortest path on this maze is ~9 steps; with ε-greedy exploration a
well-trained agent averages ~10 steps per episode. Run for 50 episodes and
all three settings converge there — the difference is *how quickly*:

| Setting | Steps per episode (last 10 eps) | What it means |
|---------|--------------------------------:|---------------|
| `n = 0`   | ~10 | Converged — but it took ~30–50 episodes of wandering to get here |
| `n = 5`   | ~10 | Converged within ~10 episodes |
| `n = 50`  | ~10 | Converged within ~3–5 episodes |

The interesting signal is the *learning curve*, not the final number. The plot
saved to `outputs/dyna_q.png` shows three curves diving toward the floor at very
different rates: `n = 50` reaches it in a handful of episodes, while `n = 0` is
still climbing down well into the run. (On a tiny deterministic maze like this,
plain Q-learning does eventually get there — Dyna-Q just needs far fewer real
episodes, which is the whole point on environments where real steps are costly.)

---

## Why It Works So Well on This Maze {#why-it-works-so-well-on-this-maze}

Two reasons:

1. **The environment is deterministic.** Each `(s, a)` always gives the same
   `(r, s')`, so the model is exact after a single visit. Imagined experience is
   as good as real experience.
2. **Real steps are expensive, imagined ones are free.** Each imagined update
   is just a few table look-ups, while a real step requires the agent to walk.
   When real interactions are costly (think: a real robot, a real game), Dyna-Q
   is enormously sample-efficient.

---

## Where Dyna-Q Struggles {#where-dyna-q-struggles}

- **Stochastic environments.** If `(s, a)` can lead to many different `s'`
  values, a "remember last outcome" model lies to you. Fix: store visit counts
  or train a probabilistic model.
- **Non-stationary environments.** If the world changes — for example, a
  doorway that was open suddenly closes, or a shortcut appears — the model
  becomes outdated and gives wrong predictions. **Dyna-Q+** fixes this by
  adding an *exploration bonus*: states that haven't been revisited for a long
  time receive a small extra reward, nudging the agent to go back and check
  whether the world has changed.
- **Large state spaces.** A dictionary keyed on `(s, a)` does not scale to
  millions of states or to continuous states. That is exactly the gap that
  **learned (neural-network) world models** fill — see `world_model.py` next.

---

## Key Words to Remember {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Model**       | Memory of `(state, action) → (reward, next_state)` |
| **Planning step** | Doing a Q-update using imagined data from the model |
| **Direct RL**   | A Q-update using real data |
| **Sample efficiency** | Measures how effectively an AI model or algorithm uses training data to achieve a specific level of performance |
| **Dyna**        | Sutton's architecture that interleaves learning + planning |

---

## One-Sentence Summary {#one-sentence-summary}

> **Dyna-Q learns from doing AND from imagining — and imagining is free.**

This idea, in its modern neural form, powers some of the strongest RL agents
ever built (MuZero, Dreamer, World Models).
