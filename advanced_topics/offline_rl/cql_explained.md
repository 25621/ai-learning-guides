# Conservative Q-Learning (CQL) 🛡️

## What Is It?

Imagine you're learning to invest money by reading a giant ledger of past
stock trades made by other people. The ledger has buys, sells, and holds —
but **no record of any trade nobody actually made**.

Now imagine an over-confident student looks at the ledger and says:
*"What if someone had bought lottery tickets every Monday? That would have
been an amazing trade!"*

The problem: **the ledger has no data about Monday-lottery-buying**, so the
student is just hallucinating. Yet that hallucinated trade looks great on
paper, so the student's "policy" keeps wanting to do it.

That hallucination problem is **distribution shift**: an offline learner
loves actions the dataset never tested, because there's no data to
contradict the optimism. CQL is the cure.

---

## How Q-Learning Goes Wrong Offline

Normal Q-learning's target is:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

That `max_{a'}` is the danger. When the dataset never recorded action `a'`
in state `s'`, the network just *guesses* a Q-value — and neural networks
tend to **over-estimate** Q for unseen inputs. The target inherits the
over-estimate, the network learns to predict that bigger number, and on
the next step we **extrapolate** (project even further beyond anything the
data supports) even higher. The policy chases a phantom.

If you could keep collecting more data this would self-correct (the
phantom action turns out to be bad in reality). But **in offline RL you
can't collect more data.** The phantom is forever.

---

## CQL's Trick

CQL (Kumar et al., 2020) adds a penalty term to the loss:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Two pieces:

1. **`log Σ_a exp Q(s, a)`** (read: *"log-sum-exp over all actions"*) is a
   **soft maximum** over all actions — a smooth, differentiable
   approximation of `max` that considers every action at once rather than
   hard-selecting one winner. Penalising it shrinks Q-values
   **across the board** (pushing all predictions
   down uniformly) — especially for actions with the *highest* Q, which is exactly
   where the hallucinations live.
2. **`- Q(s, a_dataset)`** rewards high Q on the action the dataset
   actually recorded — protecting in-distribution Q-values from the
   shrinkage above.

Net effect: **Q is pulled down on un-seen actions, pulled up on seen
actions.** The learned Q becomes a *lower bound* on the true Q. The
**`argmax`** policy (the rule that simply picks the action with the highest Q)
stops chasing phantoms.

Full loss:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Where **`Bellman_MSE`** is the standard error from normal Q-learning,
measuring how much the network's current guess disagrees with its own
future guess).

`α` is the conservatism knob. Too small → distribution shift creeps back
in. Too large → the agent is so conservative it never improves beyond the
data.

---

## Real-Life Examples

- **Conservative chess coach.** You can only learn from games already
  played. A reckless coach says "this hypothetical move with no
  precedent could be brilliant!"  CQL is the coach who says "we have no
  data on that — let's stick to moves real players have tried."
- **Restaurant menu choices.** Yelp reviews never cover the
  off-menu items. A naive policy would recommend the off-menu items
  based on hallucinated five-star ratings. CQL recommends only what's
  been ordered enough times to trust.
- **Robot grasping from logs.** The robot has video of grasping cups,
  bottles, and books — but never a knife. CQL refuses to confidently
  recommend "grab the knife by the blade."

---

## What Our Code Does

The script `cql.py`:

1. **Loads the four datasets** built by `d4rl_dataset.py`.
2. **Picks `medium-replay`** as the training set — it's the most realistic
   (mixed quality) and the most damaging to naive methods.
3. **Trains three agents purely offline**, in identical conditions except
   for `α`:
   - `α = 0`   →  naive offline DQN (no penalty — the broken baseline)
   - `α = 1.0` →  mild CQL
   - `α = 5.0` →  strong CQL
4. **Evaluates each every 2,500 gradient steps** by rolling out greedily
   in the real environment (10 episodes). This is the *only* env contact;
   training itself never sees the env.
5. **Plots learning curves** to `outputs/cql.png`.

---

## What You Should See

A typical run prints something like:

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (unstable; often crashes)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

In the learning-curve plot:

- The **red curve** (`α = 0`) climbs early then often **falls off a cliff**
  once distribution-shift hallucinations infect the **Bellman target**
  (the number we use as the "correct answer" when training the Q-network:
  `r + γ · max Q(s', ·)`). When phantom Q-values pollute that target,
  every gradient step makes things worse. The **Bellman loss** (the MSE
  between the Q-network's prediction and the Bellman target) looks fine —
  that's the **treachery** of the problem: the network is perfectly
  consistent with its own wrong beliefs, so the loss gives no warning.
- The **orange curve** (`α = 1.0`) climbs more slowly but **stays up**.
- The **green curve** (`α = 5.0`) is the most stable and usually best.

The Bellman-loss panel shows another tell-tale: naive DQN's loss can stay
small while its policy is terrible, because the network is internally
consistent with its own hallucinations.

---

## Where CQL Sits in the Field

CQL was a *big* deal because it gave a principled, simple fix to
distribution shift. The lineage:

```
DQN (online)
   │
   ▼
Naive offline DQN  ── breaks because of distribution shift
   │
   ▼
CQL (Kumar 2020)    ── add a conservative penalty: Q is a lower bound
   │
   ▼
IQL (Kostrikov 2021)  ── never query Q on un-seen actions in the first place
   │
   ▼
Decision Transformer (Chen 2021)  ── skip Q entirely, treat RL as sequence modelling
                                      (predict the *next action* given past states and
                                       a desired total return, exactly like an LLM
                                       predicts the next word)
```

Each step in this lineage is a different answer to the same question:
**how do I avoid asking my Q-network about things it has never seen?**

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Distribution shift** | The trained policy wants actions outside the data |
| **Out-of-distribution (OOD)** | An (s, a) pair the dataset never recorded |
| **True Q** | The real expected future return for taking action `a` in state `s`, if we could measure it perfectly |
| **Conservative Q** | A learned Q-function that tries to stay below the true Q instead of over-promising |
| **Logsumexp** | A smooth, differentiable approximation of `max` |
| **Alpha (α)** | CQL's conservatism knob — how hard to push Q down on OOD actions |

---

## One-Sentence Summary

> **CQL adds a "pessimism penalty" that punishes high Q-values on actions
> the dataset never tried — so the policy can't fall in love with
> hallucinations.**
