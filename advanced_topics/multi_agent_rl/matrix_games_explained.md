# Matrix Games: The Simplest Multi-Agent World 🎲

## What Is a Matrix Game?

Imagine you and a friend each pick a hand sign — **rock, paper, or scissors** —
*at the same time*. You don't see each other's pick. The winner is decided by
a small table:

|        | Rock | Paper | Scissors |
|--------|:----:|:-----:|:--------:|
| Rock     |  0,0  | -1,+1 | +1,-1 |
| Paper    | +1,-1 |  0,0  | -1,+1 |
| Scissors | -1,+1 | +1,-1 |  0,0  |

That table is the *whole world* of the game. No movement, no time, no map.
Just a one-shot decision. We call this a **matrix game** because the payoff
matrix is the entire environment.

Matrix games are the cleanest place to study **multi-agent RL**, because the
only thing that can change during training is each player's *policy* — the
probability of picking each action.

---

## Why It's "Multi-Agent"

In single-agent RL the environment is fixed: the wind always blows the same
way, the floor never moves. The agent improves and eventually wins.

In a matrix game, your "environment" is *another learning agent*. As they get
smarter, what counts as a good move for you *changes*. This is called
**non-stationarity**, and it is the central problem of multi-agent RL.

> If you keep playing Rock, your opponent will eventually start playing Paper
> always. So you switch to Scissors. So they switch to Rock. So you switch to
> Paper... and so on. The "best move" never stays put.

The classical solution is **mixed strategies**: don't pick any one action
deterministically — randomise in a way the opponent can't exploit.

---

## The Three Games We Play

### 1) Rock-Paper-Scissors (zero-sum)
- One player's gain is the other's loss.
- The **Nash equilibrium** is: each player picks each action with probability
  ⅓.  Any deviation is exploitable.
- We expect our two Q-learners to wobble around ⅓-⅓-⅓ — never perfectly
  steady, because every time one drifts, the other reacts.

### 2) Prisoner's Dilemma (general-sum)
Two suspects are interrogated separately:

|           | Cooperate | Defect |
|-----------|:---------:|:------:|
| Cooperate |   3, 3    |  0, 5  |
| Defect    |   5, 0    |  1, 1  |

- "Defect" beats "Cooperate" no matter what the other does — it is a
  **dominant strategy**.
- Both players are rational → both defect → both get 1, even though
  (Cooperate, Cooperate) was 3 each. Selfish best-response destroys group
  welfare.
- We expect Q-learning to converge cleanly to (Defect, Defect).

### 3) Stag Hunt (coordination)
Two hunters can together bring down a stag (huge prize), or each settle for a
hare (small but safe prize):

|       | Stag | Hare |
|-------|:----:|:----:|
| Stag  | 4, 4 | 0, 3 |
| Hare  | 3, 0 | 2, 2 |

- (Stag, Stag) is **payoff-dominant** — best for both.
- (Hare, Hare) is **risk-dominant** — safe if you don't trust your partner.
- Outcome depends on initial conditions: independent Q-learners often end up
  in the *worse* (Hare, Hare) equilibrium because hares are safer to learn.

---

## Real-Life Examples

- **Pricing in a duopoly.** Two coffee shops on the same street each pick a
  price every morning. The shape of the payoff matrix decides whether they
  end up at a high "cooperative" price (good for them, bad for customers) or
  a low cut-throat price.
- **Network protocols.** Routers and senders choose timing strategies; the
  network's congestion outcome is determined by the matrix-game-like payoff
  of getting through vs. backing off.
- **Bidding in an auction.** Each bidder picks a bid not knowing the others;
  payoffs depend on the entire vector. The Nash equilibrium is a *bidding
  strategy*, not a single number.

---

## What Our Code Does

For each game we:
1. Create two stateless Q-learners (Q is just one number per action — there
   are no states in a 1-shot game).
2. Loop for 20,000 steps. Each step: both agents pick an ε-greedy action
   simultaneously, get a reward, update their Q-values.
3. Track each agent's **empirical action frequency** in a rolling 500-step
   window. Instead of just looking at abstract probabilities, we count what actions they actually chose recently (e.g., "in the last 500 rounds, they played Rock 40% of the time"). This gives us a real-time, practical picture of their changing strategy.
4. Plot the frequencies over time, save to `outputs/<game>.png`, and print
   the final Q-values.

### What you should see

| Game | Expected outcome of the plot |
|------|------------------------------|
| **Rock-Paper-Scissors** | Both players hover near ⅓-⅓-⅓ but jitter visibly. The curves chase each other — classic cycling behaviour. |
| **Prisoner's Dilemma** | Both players' "Defect" frequency rises to ~1.0 quickly. "Cooperate" gets crushed. |
| **Stag Hunt** | Most random seeds settle at (Hare, Hare). Some lucky seeds reach (Stag, Stag) — try changing the seed in the script and watch it flip. |

---

## Where Independent Learning Breaks

Our agents are *independent* — they only see their own reward, never the
opponent's action or Q-values. This is the simplest baseline and it has
limits:

- It **cannot guarantee convergence** in general-sum games.
- It can get stuck in **bad equilibria** (Stag Hunt).
- It **cannot model the opponent**.

Real multi-agent algorithms fix this by explicitly reasoning about the other
learner. Here is what each one does, in plain English:

| Algorithm | Core idea                                                                                                                                                                                                                                                                               | Real-life analogy |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious play** | Keep a running tally of how often your opponent has picked each action. Assume tomorrow they'll do what they've always done — then pick your own best response to that belief.                                                                                                          | Watching an opponent's habits over many chess games and adjusting your opening accordingly. |
| **CFR (Counterfactual Regret Minimisation)** | After every round, ask *"How much did I regret not picking each other action?"* Gradually shift probability toward actions you regret skipping. Used in poker because it handles **imperfect-information** games (you don't see the opponent's cards).                                  | After a poker hand, replaying it and thinking: *"I should have bet more — I'll do that next time."* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Your gradient step accounts for the fact that the opponent is *also* gradient-stepping. You optimise your own update while anticipating the opponent's next update — two steps ahead instead of one.                                                                                    | Negotiating a deal while thinking: *"If I offer X, they'll counter with Y, so I should start with Z."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | Each agent's *critic* (value estimator) is trained with the **global view**: it sees everyone's observations and actions. The *actor* (the policy that gets deployed) still only uses local information — this is the CTDE (Centralized Training with Decentralized Execution) pattern. | A basketball coach who watches the full court (centralized critic) but teaches each player to react only to what they can see (decentralized actor). |

But independent Q-learning is the right first step. You see the
non-stationarity problem hit you in the face, and the fixes make sense
afterwards.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **Payoff matrix** | The table that defines a 1-shot multi-agent game |
| **Nash equilibrium** | A policy profile where no single agent can improve by deviating |
| **Mixed strategy** | A policy that randomises over multiple actions |
| **Non-stationarity** | The environment (= other agents) keeps changing as it learns |
| **Independent learner** | An agent that ignores the existence of other learners |
| **Zero-sum** | One agent's gain is exactly the other's loss |
| **General-sum** | Both agents can win, both can lose, or anything in between |

---

## One-Sentence Summary

> **In matrix games, the "environment" is another learner — so the best move
> keeps moving.**

This is the bedrock idea behind every multi-agent algorithm you will meet
later, from self-play to MADDPG to MARL with communication.
