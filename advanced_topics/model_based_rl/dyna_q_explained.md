# Dyna-Q: Learning From Real Life AND From Your Imagination

## The Big Idea

Imagine you are learning to find candy in a big playground. There are walls, swings,
and a hidden treasure box (🍬) somewhere.

There are two ways to learn the playground:

1. **Walk around it for real.** Every time you take a step, you learn a tiny bit
   about which way is correct. This is slow — you can only take so many real steps
   in one day.

2. **Close your eyes and remember.** "When I went past the swing and turned left,
   I bumped into a wall." You can practice this memory *in your head*, over and
   over, without taking any real steps.

**Dyna-Q does BOTH.** Take one real step, then close your eyes and practice 50
imagined steps in your head. You learn 51 times faster than just walking!

---

## The Three Things Dyna-Q Does Every Step

Every time the agent takes a real step in the world:

### 1. Direct Learning (the real step)
> "I just took a step. Was it good or bad? Let me update my notes."

This is plain **Q-learning** — adjust the value of the action you took based on
what happened.

### 2. Model Learning (build a memory book)
> "I'll write down: from spot A, if I move LEFT, I end up at spot B and get
> reward 0."

The **model** is just a little notebook: `(state, action) → (next_state, reward)`.
For a deterministic maze, this is just remembering what happened the last time.

**Real-life example:** A toddler learns "if I push the red button, the music starts."
That's a tiny world model in their head.

### 3. Planning (close your eyes and replay)
> "Now let me pretend I'm in spot A again and pick LEFT 50 times in my head,
> each time updating my notes as if it were real."

This is the **planning** part — we use the model to generate **imagined** experience
and learn from it.

```
n = 0  →  no imagining  →  just plain Q-learning
n = 5  →  5 imagined steps per real step
n = 50 →  50 imagined steps per real step (super fast learning)
```

---

## Real-life Examples

| Real life | Dyna-Q part |
|-----------|-------------|
| Walking home from school | Real step in the environment |
| Remembering "after the red house I turn left" | Model learning |
| Going to bed and replaying the walk in your head | Planning (imagined steps) |
| Suddenly knowing the route in the morning | Better Q-values from planning |

Athletes use this all the time — they call it **mental rehearsal**. A gymnast
practices a routine in her head a hundred times before a competition. Her brain
builds a model of the routine and "plans" through it.

---

## The Maze in Our Code

```
         G    ← Goal (treasure!)
   #     #
   #     #
   #     #
         #
      #
      #
S     #       ← Start
```

It is a 8×10 maze with some walls (`#`). The agent starts at `S`, must reach `G`.

- Each step: reward 0
- Reach the goal: reward +1
- Run into a wall: you stay in place

---

## What the Plot Shows

Three lines, three agents:

| Line | What it is | What happens |
|------|------------|--------------|
| 🔴 n=0 | Plain Q-learning | Episode 1 takes thousands of steps. Slowly improves. |
| 🟡 n=5 | Dyna-Q, 5 plans per step | Drops fast within ~10 episodes. |
| 🟢 n=50 | Dyna-Q, 50 plans per step | Finds the goal almost optimally by episode 3. |

The vertical axis is on a **log scale** — the green agent is *many times faster*,
not just a little bit.

---

## Why It Works

Think about what happens when the agent finally reaches the goal for the first time:
- **Without planning:** only one Q-value (the very last action) gets updated.
- **With planning:** the model now knows that big reward, and every imagined
  rollout can carry that reward backward through every state we have seen.
  After 50 imagined replays, the "good news" of the goal spreads to many earlier
  states.

> **Planning = a way to extract MORE value from each real-world experience.**

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| Model | A notebook of what we have seen happen |
| Direct RL | Update from the real step |
| Planning | Update from imagined steps using the model |
| n (planning steps) | How many times we close our eyes after each real step |
| Sample efficiency | "Learning a lot from a little real experience" |

---

## What's Next?

In `dyna_q.py` the model is just a hash table — perfect because the maze is
small and deterministic. But the real world is huge and noisy. So next we
trade the notebook for a **neural network** that learns to *predict* what
happens next. That is a **world model**, and it is in `world_model.py`.
