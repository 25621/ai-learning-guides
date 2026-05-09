# Target Network: Stabilizing the Bullseye 🎯

## The Moving Goalpost Problem

Imagine you're trying to hit a bullseye with a bow and arrow. You shoot, look at where
your arrow landed, and adjust your aim for next time. Simple, right?

Now imagine the bullseye MOVES every time you shoot! Every arrow you shoot slightly
changes where the bullseye will be for the next shot. You'd never improve — you'd be
chasing a target that's always running away.

That's exactly the problem with DQN without a target network!

---

## Why Q-Targets Keep Moving

In DQN, the target for each update is:
> target = reward + γ × max(Q(next_state))

The Q-values on the right side come from... the same network we're training!

So every time we update the network (to make Q-values better), we also change the
targets. It's a feedback loop:

1. Update the network → Q-values change
2. Q-values change → targets change
3. Targets change → update the network differently
4. Repeat forever — unstable!

**Real-life example:** Trying to weigh yourself on a scale that changes its readings
every time you step on it. You'd never know your true weight!

---

## The Solution: Freeze the Bullseye! ❄️

The **Target Network** is a COPY of the main Q-network that gets frozen in place.

- **Online network** (`qnet`): Updated every training step — learns quickly
- **Target network** (`target_net`): Frozen copy — only updated every 100 steps

We use the FROZEN target to compute targets:
> target = reward + γ × max(Q_TARGET(next_state))

The target stays still for 100 steps! That gives the online network a stable goal to
aim for. Then we copy the online weights into the target, freeze again, and repeat.

**Real-life example:** Think of a student and a teacher. The teacher gives homework
(the target). The student learns and improves. After 100 lessons, the teacher UPDATES
the homework to be harder. The teacher doesn't change every single minute — that
would be too chaotic!

---

## The Full DQN Recipe 🍕

The complete DQN algorithm (experience replay + target network) is:

```
1. Initialize online network Q and target network Q_target (same weights)
2. Create replay buffer (memory box)

Every environment step:
  a. Pick action using ε-greedy with Q
  b. Store (state, action, reward, next_state) in buffer

Every 4 steps:
  c. Sample random mini-batch from buffer
  d. Compute targets using Q_TARGET (frozen!)
  e. Update Q to minimize loss

Every 100 steps:
  f. Copy Q weights → Q_TARGET (sync target)
```

This is the exact algorithm from the DeepMind DQN paper (2015)!

---

## What the Comparison Shows

When you run `dqn_target_network.py`, you'll see:

**Without target network (DQN + replay only):**
- Training might be "okay" but with periodic collapses
- Q-values can diverge (explode or oscillate)
- Learning is less predictable

**Full DQN (replay + target network):**
- More consistent upward learning
- Q-values stay in a reasonable range
- Faster convergence to the solved threshold (195+ on CartPole)

---

## The "Deadly Triad" ☠️

In reinforcement learning, combining three things creates instability:

1. **Function approximation** (neural network instead of table) ← we use this
2. **Bootstrapping** (using Q-values to estimate Q-values) ← we use this
3. **Off-policy learning** (Q-learning uses max, not the actual policy) ← we use this

All three together = the "deadly triad." DQN tames this with:
- Experience replay → breaks correlations
- Target network → breaks the feedback loop

It doesn't fully solve the problem, but it makes it manageable!

---

## Key Vocabulary

| Word | Meaning |
|------|---------|
| **Target Network** | A frozen copy of the Q-network used only for computing targets |
| **Online Network** | The Q-network being actively trained |
| **Sync** | Copying online network weights into the target network |
| **Feedback Loop** | When the output of a system feeds back to change the input (can cause instability) |
| **Deadly Triad** | The combination of function approximation + bootstrapping + off-policy that causes instability |

---

## Real-World Impact

In 2015, DeepMind published their DQN paper showing an AI that could play 49 Atari
games at superhuman level — using JUST these two tricks (replay + target network).

Before this, people thought you couldn't train neural networks with RL because of the
instability. DeepMind proved them wrong, and it kicked off the deep RL revolution!

Next, we'll apply this full DQN recipe to Atari Pong — a real video game with raw pixels
as input!
