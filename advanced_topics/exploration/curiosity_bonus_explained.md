# Curiosity Bonus (Intrinsic Motivation) 🧭

## What Is It? {#what-is-it}

Imagine a toddler dropped into a new room. Nobody pays them, nobody claps —
yet they make a beeline for the cupboard they haven't opened, the button
they haven't pressed, the noisy toy in the corner. They are running on an
**internal reward**: *"That looks new. Go check it out."*

A **curiosity bonus** gives a reinforcement-learning agent the same
internal drive. The environment's real reward (the "extrinsic" reward —
points, money, winning the game) is left exactly as it is. We just add a
second, self-generated reward for visiting things the agent finds *novel*
or *surprising*, and train on the sum:

```
reward the agent learns from  =  real reward  +  beta * curiosity bonus
```

`beta` is a knob that starts large (be curious!) and shrinks over time
(stop dawdling, go cash in what you learned).

## Why Bother? The Sparse-Reward Problem {#why-bother-the-sparse-reward-problem}

Normal RL agents learn from rewards they actually receive. That works
great when rewards are everywhere ("+1 every step you stay upright" in
CartPole). It falls apart when the reward is **sparse** — zero, zero,
zero, ... , zero, and then finally a +1 after a long, very specific
sequence of correct actions.

Real examples of sparse reward:

- **Montezuma's Revenge** (the Atari game): your first point arrives only
  after ~100 precise moves — climb down a ladder, dodge a skull, climb up,
  grab a key. Until then the score is a flat zero.
- **A combination lock.** 9,999 wrong codes give you nothing; one gives
  you the prize.
- **Drug discovery / scientific experiments.** Thousands of failed trials,
  then one that works.
- **Writing a long proof or program.** No partial credit until the whole
  thing checks out.

A reward-only agent in these settings is like a student who refuses to
study unless they're paid per correct answer on the final — they never get
started. Curiosity is the bonus that says *"exploring is its own reward,"*
so the agent keeps poking around until it trips over the real prize.

## Two Flavours of Curiosity (both implemented in `curiosity_bonus.py`) {#two-flavours-of-curiosity-both-implemented-in-curiosity_bonuspy}

### 1. Count-based novelty: "I've barely been here" {#1-count-based-novelty-ive-barely-been-here}

The simplest possible novelty signal. Keep a tally `N(s, a)` of how many
times you've taken action `a` in state `s`, and give yourself a bonus that
shrinks as that tally grows:

```
curiosity bonus  =  1 / sqrt( N(s, a) + 1 )
```

First time you try something: bonus = 1.0. After 100 tries: bonus = 0.1.
After 10,000 tries: 0.01. The agent is rewarded for going where it hasn't
been, and the lure naturally fades from well-trodden ground.

**Real-life analogy:** a tourist with a "places I haven't visited" list.
Brand-new neighbourhood? Top priority. The café you've been to fifty
times? Not exciting anymore.

This is the oldest idea in the book (MBIE-EB, UCB). Its weakness: in a
huge or continuous world you never visit the *exact* same state twice, so
the raw count is always 1 — which is why the next flavour exists.

### 2. Prediction-error novelty: "I didn't see *that* coming" {#2-prediction-error-novelty-i-didnt-see-that-coming}

This is the idea behind the famous **ICM** (Intrinsic Curiosity Module,
Pathak et al. 2017) and its cousin **RND** (Random Network Distillation,
Burda et al. 2018). Instead of counting, the agent keeps a little
**model that tries to predict what happens next** — "if I'm here and I do
this, where do I end up?" — and rewards itself by **how wrong the model
was**:

```
curiosity bonus  =  surprise  =  -log P( the state I actually reached | where I was, what I did )
```

- A situation the model has never seen → it predicts badly → big surprise
  → big bonus → "go explore there!"
- A situation the model has seen a hundred times → it predicts perfectly →
  zero surprise → zero bonus → "been there, understood, move on."

**Real-life analogy:** a kid learning how the world works by playing.
Pushing a glass off the table the *first* time is fascinating (it
shattered!). The hundredth time, you already knew it would shatter — not
interesting. Curiosity = the gap between what you expected and what
happened.

In our tabular code the "model" is just a table of transition counts, and
"how wrong it was" is the surprisal `-log P`. Real ICM/RND use neural
networks so the same idea works on raw pixels — but the principle is
identical.

> **Why two versions?** Count-based is dead simple and a great baseline.
> Prediction-error scales to big, never-repeating worlds and gives a
> *sharper* signal: in a deterministic environment, once you've seen a
> transition once the surprise instantly drops to ~0, whereas a count
> bonus fades only slowly as `1/sqrt(N)`. In our experiments the
> prediction-error agent solves MiniMontezuma in a couple dozen episodes;
> the count agent gets there too, just more slowly and less reliably.

## What Our Code Does {#what-our-code-does}

`curiosity_bonus.py` trains a plain **tabular Q-learner** on
`MiniMontezumaEnv` — a tiny two-room gridworld where you must walk to a
**key**, pick it up (now the **door** opens), walk through, and reach the
**treasure**. Reward (+1) appears *only* at the treasure, after ~15
perfect moves. It runs three agents and plots them:

| Agent | What it does on MiniMontezuma |
|-------|-------------------------------|
| **epsilon-greedy (no curiosity)** | Wanders near the start, *never* reaches the key, score stays 0 forever. |
| **count-based bonus** | Reliably finds the key; threads the whole chain to the treasure maybe ~40% of episodes. Works — just a bit noisy. |
| **prediction-error bonus** | First reaches the key *and* the treasure within ~20–25 episodes; as `beta` decays it converges to solving it on every episode. |

The figure shows:
- a learning curve: *P(reach the treasure)* over training,
- a second curve for the intermediate milestone *P(pick up the key)*,
- and **state-visitation heat-maps** of the grid — the no-curiosity agent
  is a tight blob near the start; the curious agents flood *both* rooms.

## The Mechanism in One Picture {#the-mechanism-in-one-picture}

```
            no curiosity                       with curiosity bonus
   reward:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └──── nothing to learn from ──┘     └ + 0.4 0.3 0.9 0.2 ... ┘  (self-made,
                                                  dense, points "toward the new stuff")
   result:  random walk, never finds +1         systematically sweeps the world,
                                                 trips over +1, then the bonus fades
```

The curiosity bonus turns *"I haven't seen this"* into reward, so the
agent **deliberately pushes into unexplored territory** instead of
flailing randomly. And because the bonus shrinks as things become
familiar (and `beta` decays), once the agent has found the real reward it
naturally stops dawdling and starts exploiting.

## A Few Honest Caveats {#a-few-honest-caveats}

- **The "noisy-TV problem".** A prediction-error agent can get hypnotised
  by a source of pure randomness (a TV showing static, dice rolling) — it
  can *never* predict it, so the surprise never fades. ICM's real trick is
  to predict in a *learned feature space* that ignores stuff the agent
  can't control; RND sidesteps it differently. Our deterministic
  gridworld has no noisy TV, so we don't hit this.
- **Curiosity is a means, not an end.** That's why `beta` decays. An agent
  that stays maximally curious forever never settles down to actually
  *win*.
- **Scaling deep exploration is still hard.** A bonus in the reward helps,
  but plain tabular Q-learning is slow to propagate the resulting optimism
  down a long chain (see `compare_exploration.py`). Cracking pixel
  Montezuma needed extra firepower — RND with a neural net, bootstrapped
  DQN, Go-Explore.

## Key Words to Remember {#key-words-to-remember}

| Word | Meaning |
|------|---------|
| **Intrinsic reward** | A reward the agent generates for itself, separate from the environment's reward |
| **Extrinsic reward** | The environment's real reward (points, win/lose) |
| **Sparse reward** | Reward is zero almost everywhere; you only get it after a long correct sequence |
| **Novelty / surprise** | How new or unexpected a state (or transition) is — the thing curiosity rewards |
| **Count-based bonus** | Novelty ≈ `1/sqrt(visit count)` — the classic exploration bonus |
| **ICM** | Intrinsic Curiosity Module: novelty = a forward model's prediction error (in a learned feature space) |
| **`beta`** | The weight on the curiosity bonus; usually annealed toward 0 so the agent eventually exploits |

## One-Sentence Summary {#one-sentence-summary}

> **A curiosity bonus is a self-given reward for novelty — it manufactures
> a dense, "go-explore-over-there" signal that drags the agent through
> sparse-reward worlds it would otherwise never solve, then politely fades
> away once everything is familiar.**
