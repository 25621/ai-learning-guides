# Deep Q-Network (DQN) from Scratch 🧠

## The Problem with Linear

Remember our linear formula from before?

> Score = w₁ × cart_position + w₂ × cart_speed + w₃ × pole_angle + w₄ × pole_speed

This works okay for CartPole, but what about a video game where you see thousands of
pixels? You can't write a simple recipe for that!

We need something that can look at complicated situations and figure out the best action.
That something is a **neural network**.

---

## What is a Neural Network?

Think of your brain. Millions of tiny cells called neurons talk to each other. When you
touch something hot, neurons send signals: "HOT! → Pull hand away NOW!" Each neuron
passes information along, and together they make a smart decision.

A **neural network on a computer** works the same way:

```
Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
[cart pos]  →    [128 neurons]  →  [128 neurons]  →  [push LEFT score]
[cart speed] →   [  ...       ]    [  ...       ]    [push RIGHT score]
[pole angle] →
[pole speed] →
```

Each arrow has a **weight** (how strong that connection is). There are thousands of
these weights — and the network learns ALL of them!

**Real-life example:** A chef at a restaurant tastes your food and adjusts hundreds of
ingredients at once. Each taste bud is like a neuron, and together they tell the chef
"add more salt" or "less pepper." Training the network is like the chef learning over
thousands of meals.

---

## DQN = Deep Q-Network

**DQN** (Deep Q-Network) was invented by DeepMind in 2013. They took the old Q-learning
formula and swapped out the Q-table for a neural network!

Instead of:
> Q-table[state][action] = score

We have:
> Q-network(state) → [score_for_left, score_for_right]

The network takes the state as input and outputs Q-values for ALL actions at once.
This is much more efficient than computing them separately!

---

## This Script: The "Naive" Version

This script shows DQN **without** any special tricks. It just:
1. Sees the state
2. Asks the network "how good is left? how good is right?"
3. Does the action with the higher score
4. Gets a reward, updates the network

**This is intentionally unstable!** Think of it like a student who immediately forgets
their previous lessons every time they learn something new. The network updates after
every single step, which causes chaos.

**Real-life example:** Imagine learning to cook by changing your entire recipe after
every single bite. You might go from "too salty" to "no salt at all" to "way too salty"
and never settle on the right amount. That's what happens here!

---

## What You'll See

When you run `dqn_cartpole.py`:
- The scores might jump around a lot (unstable learning)
- Sometimes the agent gets really good, then forgets everything
- The loss plot shows wild swings

**This is expected!** It shows WHY we need improvements — experience replay and target
networks. Those come next!

---

## The ε-Greedy Trick 🎲

The robot doesn't always pick the best action. Sometimes it picks randomly!

Why? Because if it always picks what seems best, it might never discover better options.

> With probability ε (epsilon): pick a RANDOM action (explore!)
> With probability 1-ε: pick the BEST known action (exploit!)

We start with ε = 1.0 (100% random) and slowly decrease to ε = 0.01 (1% random).
This way, the robot explores a lot at first, then focuses on what it learned.

**Real-life example:** When visiting a new city, you might try random restaurants at
first (explore). After a while, you go back to your favorites (exploit). But you still
occasionally try something new just in case there's a hidden gem!

---

## Key Vocabulary

| Word | Meaning |
|------|---------|
| **Neural Network** | Layers of connected math neurons that learn from data |
| **Deep** | More than one hidden layer (hence "deep" learning) |
| **DQN** | Deep Q-Network — uses neural net instead of Q-table |
| **ε-Greedy** | Strategy: explore randomly sometimes, exploit best knowledge other times |
| **Instability** | The network keeps "forgetting" because updates interfere with each other |

---

## What's Missing (and Why It Matters)

This naive DQN has two big problems:

1. **Correlated updates**: Every experience comes in order (step 1, step 2, step 3...).
   If step 5 was bad, ALL nearby updates get confused together.
   
2. **Moving target**: After every update, the network changes. But the next update uses
   the SAME network to compute what the target should be. It's like shooting at a moving
   bullseye!

These problems are solved by **Experience Replay** and **Target Networks** in the next
scripts. Together, they turn DQN from a wobbly beginner into a champion game player!
