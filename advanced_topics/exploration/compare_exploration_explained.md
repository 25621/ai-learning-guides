# Comparing Exploration Strategies 🔦

## The One-Sentence Problem

An RL agent has to do two things that pull in opposite directions:

- **Exploit**: do the thing that has worked best so far.
- **Explore**: try something new, in case it's even better.

Lean too far toward exploit and you'll happily settle for a mediocre
routine forever. Lean too far toward explore and you never cash in. *How*
you explore — not just *whether* — is what separates an agent that solves
Montezuma's Revenge from one that scores zero.

This script puts **five** exploration strategies head-to-head on the same
two hard tasks, so you can see their personalities.

## Real-Life Analogy: Picking a Lunch Spot

You've just moved to a new city with 200 restaurants.

- **ε-greedy** = "Go to my current favourite, but once every ten days roll
  a die and pick a *totally random* restaurant." You'll sample widely but
  *aimlessly* — and you'll keep re-rolling places you already hated.
- **Optimistic initialisation** = "Assume *every* restaurant I haven't
  tried is the best in town until proven otherwise." You'll methodically
  work through all 200, crossing each off as reality disappoints you —
  and you'll find the genuinely great ones fast.
- **UCB** = "Prefer my favourite, but give a *bonus* to places I've barely
  tried — the less I know about it, the bigger the bonus." Smart about
  *which* new place to try next, but it judges each restaurant in
  isolation.
- **Count-based reward bonus** = like UCB, but you also *enjoy the novelty
  itself* — a meal at a brand-new place is intrinsically satisfying, and
  that satisfaction shapes your long-term plan of which neighbourhoods to
  wander into.
- **Prediction-error reward bonus** = "I get a thrill from a meal that
  *surprised* me — something I couldn't have predicted." A new place that
  turns out exactly as expected? Meh. One that's wildly different from
  your mental model? Fascinating, and you update your plan to seek more
  like it.

## The Five Strategies (all in `compare_exploration.py`)

### 1. ε-greedy — the default, and it's *dithering*, not exploring

Act greedily, but with probability ε take a uniformly random action. It's
the standard baseline in DQN and friends. Its fatal flaw on hard tasks:
**every step is an independent coin flip.** To stumble down a chain of `N`
correct moves you need the coin to land right `N` times in a row — that's
exponentially unlikely. ε-greedy is *jiggling*, not *exploring*.

### 2. Optimistic initialisation — "innocent until proven boring"

Start *every* Q-value at the largest return that's even possible,
`R_max / (1 − γ)`. Now an action the agent has never tried looks like the
best thing in the world, so the **greedy** policy is forced to go try it;
only after visiting it does the value sag toward the truth. The optimism
about *un*tried regions automatically **propagates through the value
function** (via Q-learning's bootstrap), so the agent is pulled, step by
step, toward the parts of the world it hasn't seen. Almost free, no extra
bookkeeping — and, as you'll see, the strongest *deep* explorer in a small
tabular world.

### 3. UCB-style action selection — bonus in the *choice*, not the *reward*

Pick `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: prefer high-value
actions, but inflate the ones you've rarely tried. Famous from multi-armed
bandits. The catch: the bonus lives only in the **action-selection rule**,
never in the reward — so it does *not* flow through the value function.
UCB is great at "make sure I've tried every action in *this* state" but
weak at "plan a route toward a far-away unexplored region."

### 4. Count-based **reward** bonus — curiosity, the classic version

Add `1/√(N(s,a))` to the **reward** (with a weight `beta` that decays).
Because it's in the reward, Q-learning *does* propagate it: states that
lead toward novel regions become valuable. This is the MBIE-EB / classic
"exploration bonus" idea — and the first half of work item 1.

### 5. Prediction-error **reward** bonus — curiosity, the ICM/RND version

Add `−log P(s'|s,a)` from a tiny learned forward model to the reward
(again with decaying `beta`). The sharpest novelty signal of the five: in
a deterministic world, the surprise of a transition drops to ~0 the moment
you've seen it once, instead of fading slowly like `1/√N`. The tabular
cousin of ICM / RND — the second half of work item 1.

## The Two Test Tasks

- **Task A — MiniMontezuma**: the key→door→treasure gridworld, reward only
  at the treasure (~15 perfect moves away). Tests "can you survive a long
  sparse-reward chain at all?"
- **Task B — DeepSea(N)**: the textbook deep-exploration chain, run at
  lengths `N = 5, 8, 11, 14`. The reward hides behind `N` correct moves,
  each with a tiny immediate cost — so a myopic agent learns to avoid the
  cost and never finds the prize. Tests "does your strategy still work as
  the chain gets *longer*?"

## What Actually Happens (run it and see)

**Task A — MiniMontezuma:**

| Strategy | First treasure | Final solve rate |
|----------|---------------:|-----------------:|
| ε-greedy | never | 0.00 |
| optimistic init | ~episode 1 | 1.00 |
| UCB action selection | ~episode 3 | ~0.95 |
| count reward bonus | ~episode 82 | ~0.41 |
| prediction reward bonus | ~episode 23 | 1.00 |

**Task B — DeepSea, fraction of seeds that found the reward:**

| Strategy | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| optimistic init | 1.0 | 1.0 | 1.0 | 1.0 |
| UCB action selection | 1.0 | 1.0 | 0.0 | 0.0 |
| count reward bonus | 1.0 | 1.0 | ~0.1 | 0.0 |
| prediction reward bonus | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(Numbers wobble a little with the random seeds, but the shape is rock
solid.)*

## The Lessons

1. **ε-greedy is not exploration.** It never solves *either* hard task.
   Random dithering simply does not thread long correct sequences. (Yet
   it's still the default in a lot of code — because on *easy* tasks it's
   good enough and dead simple.)

2. **Real exploration means being optimistic about the unknown — one way
   or another.** Whether you bake the optimism into the *initial values*
   (strategy 2), into the *action choice* (strategy 3), or into a
   *self-generated reward* (strategies 4–5), the common thread is: *make
   the unexplored look attractive*, then let learning carry you there.

3. **On a sparse-reward grid, all four "real" strategies work — and the
   prediction-error bonus gets there fastest**, because it produces the
   crispest "this is new" signal.

4. **On a *deep* chain, where the optimism has to travel a long way, the
   simple champion is optimistic initialisation.** It propagates optimism
   through the value function for free. UCB falls apart first (its bonus
   never enters the value function, so it can't *plan* deep). Reward
   bonuses do better — they *do* propagate — but plain tabular Q-learning
   is slow to push that optimism all the way down a long chain before the
   bonus decays.

5. **That last point is exactly why scaling deep exploration to pixels
   needed extra firepower** — bootstrapped DQN, RND with a real neural net
   (so optimism *generalises* across similar states instead of
   propagating one cell at a time), Go-Explore (literally remember and
   return to promising states). The tabular toys here show you the
   *principles*; the real systems are these same principles plus a network
   that generalises.

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Exploration–exploitation trade-off** | Try new things vs. cash in what you know — the central tension in RL |
| **Dithering** | "Exploration" by adding random noise to actions (ε-greedy, Gaussian noise) — weak on hard tasks |
| **Optimism in the face of uncertainty** | The umbrella principle: treat the unknown as if it's great until you've checked |
| **Optimistic initialisation** | Implement that principle by starting all values at the max possible return |
| **UCB** | Upper Confidence Bound: pick `argmax (value + bonus that shrinks with visit count)` |
| **Deep exploration** | Exploration that requires a long *coherent* sequence of "unusual" actions, not just one |
| **`beta` annealing** | Shrinking the curiosity weight over time so the agent eventually stops exploring and exploits |

## One-Sentence Summary

> **ε-greedy is just noise; every real exploration strategy works by making
> the unexplored look attractive — via optimistic values, an action-choice
> bonus, or a self-generated novelty reward — and the right choice depends
> on whether your reward is merely *sparse* or genuinely *deep*.**
