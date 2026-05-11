# Exploring PettingZoo Environments 🦓

## What Is PettingZoo?

If you have done single-agent RL you have probably used **Gymnasium** (the
successor to OpenAI Gym). Every environment looks the same: `env.reset()`,
`env.step(action) -> obs, reward, done, info`. That uniformity is what
makes RL libraries work.

**PettingZoo** is exactly the same idea but for *multiple agents*. It is a
zoo of multi-agent environments — board games, classic toy problems,
cooperative grid worlds, Atari multiplayer, even MPE (Multi-Particle
Environment) — all behind one well-defined API. If you can write code that
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
- an observation (what they see *right now*),
- a reward (from the previous action by *another* agent),
- termination and truncation flags.

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

This is natural for **real-time games** like MPE or multi-agent gridworlds.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: my_policy(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

The two styles are isomorphic: any AEC environment can be wrapped to look
parallel, and vice versa. PettingZoo provides the wrappers.

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

The observation is the previous joint action (encoded as one of 9 integers,
including a "start" symbol). An episode lasts 25 steps, so the maximum total
return is +25 per agent, the minimum is -25, and random play scores ~0.

We then:

1. **Demonstrate the AEC interface** with a random rollout. Confirms that
   the iteration / `last()` / `step()` dance works.
2. **Train two independent Q-learners through the Parallel interface**.
   Each agent has its own Q-table indexed by the joint-action observation.
3. **Try to import the real `pettingzoo` library** and roll out one of its
   built-in environments (Rock-Paper-Scissors) with a random policy. If
   PettingZoo isn't installed, we skip this step with a friendly message.

### What you should see

| Stage | Expected |
|-------|----------|
| Random rollout (AEC)            | Mean episode return near **0** — random agents don't coordinate. |
| Independent Q-learners (Parallel) — first 100 eps | About **0** — still random while exploring. |
| Independent Q-learners — last 100 eps             | Strongly positive, **+20 to +25** — coordination has emerged. |

The plot `outputs/pettingzoo_coordination.png` shows individual episode
returns (grey) and a rolling-mean curve (blue) climbing from ~0 to ~+25,
with a dashed green line at the perfect-coordination ceiling.

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

Because **the API is the lesson.** Multi-agent RL has many flavours
(turn-based, real-time, cooperative, competitive, mixed), and they all
fit into the AEC / Parallel pattern. Once you have implemented those two
loops, every PettingZoo environment is just a matter of plugging in a
different env constructor — the training code stays the same.

This is exactly how Gymnasium changed single-agent RL: by making the
environment a black box behind a uniform interface.

---

## Where Independent Q-learning Helps and Hurts

Coordination games are *forgiving* — the agents share the reward sign, so
their interests align. Independent learners can solve this happily.

In **adversarial** games (RPS) independent Q-learning oscillates forever.
In **partially-observable** games it can't learn at all because the
"observation" is only one piece of the state. PettingZoo includes both
kinds of environment so you can see these failure modes for yourself.

---

## Key Words to Remember

| Word | Meaning |
|------|---------|
| **PettingZoo**     | The Gymnasium of multi-agent RL — a library of standardised MARL environments |
| **AEC**            | Agent-Environment-Cycle: one agent acts per step (turn-based) |
| **Parallel API**   | All agents act simultaneously each step |
| **MPE**            | Multi-Particle Environment, a popular cooperative/competitive testbed shipped with PettingZoo |
| **CTDE**           | Centralised Training, Decentralised Execution — train with a global view, deploy with only local obs |
| **Independent Q-learning** | Each agent runs vanilla Q-learning, ignoring that other learners exist |

---

## One-Sentence Summary

> **PettingZoo gives every multi-agent environment the same shape — so the
> code you write today still works tomorrow on a totally different game.**

Once the two API styles are second nature, you can step up to MADDPG,
QMIX, MAPPO, or any other modern MARL algorithm — the environment side
of your code never has to change.
