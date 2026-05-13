# Exploring PettingZoo Environments 🦓

## What Is PettingZoo?

If you have done single-agent RL you have probably used **Gymnasium** (the
successor to OpenAI Gym). Every environment looks the same: `env.reset()`,
`env.step(action) → obs, reward, done, info` — a new *observation* of the world,
a scalar *reward* signal, a *done* flag saying "game over", and an *info*
dictionary for debugging extras. That uniformity is what makes RL libraries work.

**PettingZoo** is exactly the same idea but for *multiple agents*. It is a
zoo of multi-agent environments — all behind one well-defined API:
- **Classic toy problems**: simple environments like Rock-Paper-Scissors to test basic algorithms.
- **Cooperative grid worlds**: agents navigating a grid to achieve a shared goal.
- **Atari multiplayer**: classic competitive games like Pong.
- **MPE (Multi-Particle Environment)**: continuous-space physics environments for complex coordination and competition.

If you can write code that
works on one PettingZoo environment, you can plug into any of the others
with almost no changes.

---

## The Two API Styles

Multi-agent settings are messier than single-agent ones because two agents
can act at the same time, or in turn, or even in arbitrary orders. PettingZoo
solves this with two parallel APIs:

### 1) AEC (Agent-Environment-Cycle)

One agent acts at a time. The environment loops through agents in some
order, and each gets:
- an **observation** — what they see *right now*,
- a **reward** — the payoff earned by the *joint* action in the last full
  round (i.e., what happened as a result of *all* agents acting, not just
  you; in a chess game, for example, your reward reflects the board state
  after your opponent's last move, not just yours),
- a **termination flag** — `True` when the episode ends *naturally* (e.g.,
  checkmate, someone wins),
- a **truncation flag** — `True` when the episode is *cut short* by a time
  limit before a natural ending is reached.

This is natural for **turn-based games** like chess, Go, poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = my_policy(obs, agent)
    env.step(action)
```

### 2) Parallel

All agents observe and act simultaneously every step. `step()` takes a
*dictionary* of actions and returns dictionaries of observations and
rewards.

This is natural for **real-time games** like MPE (Multi-Particle
Environments, where all dot-agents move simultaneously) or multi-agent
gridworlds.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

The two styles are **isomorphic** — structurally equivalent and
interconvertible: any AEC environment can be automatically wrapped to
look like a Parallel one, and vice versa. PettingZoo ships the conversion
wrappers so you only ever have to write code for one style.

---

## Real-Life Analogy

- **AEC = a board game night.** "Alice's turn. Now Bob. Now Carol. Back to
  Alice." Whoever moves next sees the latest board state.
- **Parallel = a multiplayer video game.** All four players are pressing
  buttons simultaneously; the game updates the world 60 times a second.
- **Why uniform APIs matter.** Imagine if every multiplayer video game
  needed its own joystick. PettingZoo is the "universal joystick" of MARL.

---

## What Our Code Does

We build a **PettingZoo-style** environment from scratch: the **Iterated
Coordination Game**. Two agents repeatedly pick channel `0` or `1`:

- Same choice → both get +1
- Different choice → both get -1

The **observation** each agent receives is the previous *joint action* —
what both agents chose last round, packed into a single integer.
Concretely: each agent's last action is one of `{start, 0, 1}` (3 states),
so the pair encodes as `3 × agent_1_state + agent_2_state`, yielding
9 possible integers (0 – 8). Integer 0 is the "start" state — it signals
that no action has been taken yet (the very beginning of an episode).
An episode lasts 25 steps, so the maximum total return is +25 per agent
and the minimum is −25. **Random play scores ≈ 0** because at each step
two independent random agents each pick 0 or 1 with equal probability:
they match 50 % of the time (+1) and differ 50 % of the time (−1), giving
an expected per-step reward of 0.5 × (+1) + 0.5 × (−1) = **0**. Summed
over 25 steps the expected episode return is also 0.

We then:

1. **Demonstrate the AEC interface** with a random rollout — this confirms
   the basic AEC loop: `agent_iter()` yields the agent whose turn it is,
   `last()` reads that agent's current observation and accumulated reward,
   and `step()` delivers their chosen action back to the environment.
2. **Train two independent Q-learners through the Parallel interface**.
   Each agent keeps its own Q-table keyed by the **joint-action
   observation** (the single integer that encodes what *both* agents did
   last round), so it can learn "when we both picked 0 last time, I should
   pick 0 again."
3. **Try to import the real `pettingzoo` library** and roll out one of its
   built-in environments (Rock-Paper-Scissors) with a random policy. If
   PettingZoo isn't installed, we skip this step with a friendly message.

### What you should see

| Stage | Expected |
|-------|----------|
| Random rollout (AEC)            | Mean (average) episode return near **0** — random agents pick channels independently, matching and mismatching in roughly equal measure. |
| Independent Q-learners (Parallel) — first 100 eps | About **0** — still mostly random while agents explore. |
| Independent Q-learners — last 100 eps             | Strongly positive, **+20 to +25** — **coordination has emerged**: both agents have learned to reliably pick the same channel every round. |

The plot `outputs/pettingzoo_coordination.png` shows individual episode
returns (grey) and a rolling **Mean** curve (blue). The mean smooths out noisy
episodes so you can see the trend: the agents move from uncoordinated random
play near ~0 toward stable **coordination** near ~+25. The dashed green line
marks the perfect-coordination ceiling.

If `pettingzoo` is installed, the script also rolls out
`pettingzoo.classic.rps_v2` to prove the script works against the real
library exactly the same way it works on our hand-rolled env. To enable
that section:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Why Build a Custom Env First?

Because **the API is the lesson.** (Understanding how to structure the interaction between multiple agents and the environment is more important than the specific game rules.) Multi-agent RL has many flavours
(turn-based, real-time, cooperative, competitive, mixed), and they all
fit into the AEC / Parallel pattern. Once you have implemented those two
loops, every PettingZoo environment is just a matter of plugging in a
different env constructor — the training code stays the same.

This is exactly how Gymnasium changed single-agent RL: by making the
environment a black box behind a uniform interface.

---

## Where Independent Q-learning Helps and Hurts

Coordination games are *forgiving* — the agents share the reward sign, so
their interests align. Independent learners can solve this happily because any improvement by one agent helps the other.

In **adversarial** games (RPS) independent Q-learning oscillates forever (as one agent adapts, the other changes its strategy to counter, leading to endless chasing).
In **partially-observable** games it can't learn at all because the
"observation" is only one piece of the state (an agent might be penalized for a good action just because it couldn't see what the other agent was doing). PettingZoo includes both
kinds of environment so you can see these failure modes for yourself.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **PettingZoo**     | The Gymnasium of multi-agent RL — a library of standardised MARL environments |
| **AEC**            | Agent-Environment-Cycle: one agent acts per step (turn-based) |
| **Parallel API**   | All agents act simultaneously each step |
| **MPE**            | Multi-Particle Environment, a popular cooperative/competitive testbed shipped with PettingZoo (often involving moving dots navigating physics-based tasks). |
| **CTDE**           | Centralised Training, Decentralised Execution — train with a global view (access to all states), deploy with only local obs (each agent acts on its own limited vision). |
| **Independent Q-learning** | Each agent runs vanilla Q-learning (the standard, unmodified Q-learning algorithm), ignoring that other learners exist. |

---

## One-Sentence Summary

> **PettingZoo gives every multi-agent environment the same shape — so the
> code you write today still works tomorrow on a totally different game.**

Once the two API styles are second nature, you can step up to MADDPG
(centralised critic for continuous-control agents), QMIX (value mixing for
cooperative teams), MAPPO (multi-agent PPO), or any other modern MARL
algorithm — the environment side of your code never has to change.
