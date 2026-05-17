# Training on Montezuma's Revenge 🏛️🔑

## Why This Game Is Famous (in RL circles) {#why-this-game-is-famous-in-rl-circles}

In 2015, DeepMind's DQN learned to play dozens of Atari games at
super-human level from raw pixels. It made headlines. But buried in the
results table was one game where DQN scored **0** — the same as doing
nothing at all: **Montezuma's Revenge**.

Why? Look at what the game asks of you in the very first room:

1. Climb *down* a ladder.
2. Walk across a ledge.
3. Jump over a rolling skull (mistime it → you die).
4. Climb *up* another ladder.
5. Grab the key.

That's roughly **100 precise button presses**, and the game gives you
**not a single point** until the key is in hand. The reward signal is a
flat, featureless **zero** for the entire opening sequence.

A normal RL agent learns by adjusting toward rewards it actually receives.
If the reward is zero everywhere it can reach, there is *nothing to learn
from* — it's like trying to find the bottom of a perfectly flat valley by
feeling for the downhill direction. So DQN just twitched around the
starting platform forever. Montezuma became *the* benchmark for **hard
exploration**: the game you can only beat if you explore *cleverly*, not
randomly.

The breakthrough came in 2018 with **Random Network Distillation (RND)** —
and the trick was exactly the subject of work item 1: add an **intrinsic
curiosity bonus** so the agent rewards *itself* for reaching novel screens,
and suddenly it has a dense signal pulling it deeper into the level. RND
got a super-human score on Montezuma. (Later: Go-Explore, Agent57, …)

## Real-Life Examples of "Montezuma-style" Sparse Reward {#real-life-examples-of-montezuma-style-sparse-reward}

- **A combination lock / a treasure hunt with cryptic clues.** No partial
  credit. You're at zero until you're suddenly at the prize.
- **Getting a paper accepted, or a startup to profitability.** Months of
  no external reward, then (maybe) a big one.
- **A video game speedrun route.** Dozens of frame-perfect inputs in a row
  with no feedback until the trick either works or doesn't.
- **Escape rooms.** The room tells you almost nothing until you've chained
  several discoveries together.

In all of these, "just try random stuff" is hopeless. You need to
*systematically* explore — and an internal "ooh, that's new, keep going"
signal is what keeps you systematic.

## Why We Don't Actually Train On Pixel Montezuma Here {#why-we-dont-actually-train-on-pixel-montezuma-here}

Doing the *real* thing properly means:

- a convolutional network to see the 210×160 RGB screen,
- frame-stacking (so the agent can tell which way the skull is moving),
- an RND module (two more networks: a fixed random "target" and a trained
  "predictor"),
- and **tens of millions of environment frames** — many GPU-hours.

That's a research project, not a teaching script. So `montezuma_revenge.py`
does two honest things instead:

### 1. It *touches* the real game (if `ale-py` is installed) {#1-it-touches-the-real-game-if-ale-py-is-installed}

It loads `ALE/MontezumaRevenge-v5` through Gymnasium, runs a **uniform-
random agent for 2000 steps**, and reports the total game reward. The
number it prints is almost always **0.0** — the abstract phrase "sparse
reward" turned into a concrete, run-it-yourself fact. If the Atari
package isn't installed, it prints the one-line `pip install` command and
moves on.

### 2. It trains a tabular agent on a *scale model*: `MiniMontezumaEnv` {#2-it-trains-a-tabular-agent-on-a-scale-model-minimontezumaenv}

This is a tiny gridworld with the same *skeleton* as Montezuma's first
room:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = start
#.....D.......#     K = key      D = door (only passable while carrying the key)
#..K..#.......#     T = treasure (the ONLY tile that gives a reward)
###############
```

To win you must: walk to the **key** (~6 moves), pick it up; walk to the
**door** (~4 moves) — which now opens; walk through and reach the
**treasure** (~5 moves). About **15 perfect moves**, with **zero feedback
until the treasure**. The `has_key` flag is part of the agent's state, so
once you grab the key there's a whole second room of *new* states to
discover — just like new screens opening up in the real game.

We then train a plain **tabular Q-learner** twice:

| Agent | Result on MiniMontezuma |
|-------|--------------------------|
| **no curiosity (epsilon-greedy)** | Return stays at **0** for all 1,500 episodes. It never even reaches the key. (Sound familiar? That's DQN on the real game.) |
| **with a prediction-error curiosity bonus** | Reaches the treasure within ~20–25 episodes and then learns the **optimal 15-step route**. (That's the RND idea, shrunk to fit in a Q-table.) |

The figure shows the two learning curves side by side, plus the actual
route the curious agent learned, drawn on the grid (start → key → door →
treasure). The script also prints that route as ASCII frames.

## The Lesson {#the-lesson}

> **"Sparse reward" is not a quirk of one weird Atari game — it's the
> default in any world where success requires a long, specific sequence of
> actions.** A reward-only agent (vanilla DQN) literally cannot get
> started: there is no gradient to follow. A curiosity bonus manufactures
> one — a dense, self-generated "this is new, keep going" signal — and
> that signal is what carries the agent across the desert of zeros to the
> first real reward. Everything after that (RND, Go-Explore, Agent57) is a
> scaled-up, neural-network version of the same move.

## Key Words to Remember {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Hard exploration** | Problems where you only succeed by exploring cleverly; random exploration fails |
| **Sparse reward** | The reward is zero almost everywhere; you get it only after a long correct sequence |
| **Montezuma's Revenge** | The Atari game that classic deep-RL agents (DQN, A3C) scored 0 on — the canonical hard-exploration benchmark |
| **RND (Random Network Distillation)** | The 2018 method that beat Montezuma using a prediction-error curiosity bonus |
| **Go-Explore** | "Remember promising states, return to them, then explore from there" — another Montezuma-cracker |
| **Scale model** | A small, cheap environment that keeps the *structure* of a hard problem so you can study it quickly |

## One-Sentence Summary {#one-sentence-summary}

> **Montezuma's Revenge is the game that taught RL "rewards you never
> receive can't teach you anything" — and the fix, then and now, is a
> curiosity bonus that lets the agent reward itself for exploring until it
> finds the real prize.**
