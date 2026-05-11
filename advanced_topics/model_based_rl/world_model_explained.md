# World Models: Teaching a Robot to Daydream

## The Big Idea

You know how when you close your eyes, you can imagine eating an ice cream cone?
You can almost taste it. You can even imagine the cone falling on the floor and
feel a little sad.

You didn't really eat the ice cream. You didn't really drop it. But your brain
**predicted** what would happen.

That predicting brain is called a **world model**. In this script, we teach a tiny
robot brain (a neural network) to do the same thing for a balancing game called
**CartPole**.

---

## What is CartPole?

A cart has a long pole standing up on top, and it can wiggle around. The cart can
push **LEFT** or **RIGHT**. The goal: keep the pole standing up as long as possible.

The state of the world is just four numbers:
1. Where the cart is
2. How fast the cart is moving
3. The angle of the pole
4. How fast the pole is tipping

The action is just one of two choices: `LEFT` (0) or `RIGHT` (1).

---

## What a World Model Does

The world model is a function:

```
   (current state, action chosen)  →  (next state, reward, did we fall?)
```

In other words: **"If I'm here and I push left, where will I be a moment later?"**

After learning, the model can run *in the robot's imagination* without ever
touching the real game.

### Real-life examples of world models

| Situation | What the world model predicts |
|-----------|------------------------------|
| You stack one more block on a tower | "I think it will fall!" |
| You throw a ball | "It will land near the swing" |
| You pour milk into the cup | "It will overflow at this rate" |
| You step on ice | "I will slip" |

Your brain has world models for thousands of things. They are how you plan ahead
without trying things first.

---

## How We Train the Model

### Step 1: Collect experience (the data)
The robot plays CartPole using **random actions** — just for fun, no skill. We
record every single moment:

```
(state, action, next_state, reward, did_we_fall)
(state, action, next_state, reward, did_we_fall)
...
20,000 of these
```

### Step 2: Teach the neural network
The neural network has three little "heads":

| Head | Predicts |
|------|----------|
| Δ-state head | how the four numbers will change |
| Reward head  | the reward we will get |
| Done head    | the chance that the pole will fall |

We show the network thousands of examples and ask: *"Given the state and action,
guess the next state. How close were you?"* We adjust the network to be a bit
closer next time.

### Step 3: Test the imagination

This is the most fun part. We do an experiment:
1. Reset the **real** CartPole.
2. Pick 20 random actions.
3. Play them in the **real** game and write down where the cart ends up.
4. Play the **same** 20 actions in the **imagined** game (using the model).
5. Compare the real ending to the imagined ending.

If the imagined ending is close to the real one, the model is good!

---

## Why Predicting Many Steps is Hard

Predicting **one** step ahead is easy. Predicting **20 steps** ahead is hard
because errors compound:

```
Step 1: model is off by 0.01
Step 2: starts from the wrong place → off by 0.02
Step 3: → off by 0.05
...
Step 20: → off by 0.5 (uh oh)
```

This is why the plot shows three bars — the error grows as we predict further
into the future.

**Real-life example:** Predicting tomorrow's weather is pretty accurate.
Predicting next month's weather... not so much. Same problem.

---

## What the Plot Shows

The plot has two parts:

**Left:** The training loss going down — the model is getting better at predicting.

**Right:** Three bars showing how wrong the model is at horizons 1, 5, and 20 steps.
- 1 step: very small error (good imagination!)
- 5 steps: small but growing error
- 20 steps: error is bigger — but the imagination is still useful

---

## What's Saved

After training, we save the model brain to a file called `world_model.pt`. The
next script (`model_based_planning.py`) will load it and use the imagination to
actually **play CartPole well**.

---

## Key Takeaways

| Concept | Plain English |
|---------|---------------|
| World model | A predictor of what happens next |
| Transition | One real (state → next state) pair we collected |
| Δ-state | The CHANGE in the state — easier to predict than the new state directly |
| Compounding error | Mistakes pile up the further you imagine |
| Horizon | How many steps ahead we imagine |
| Open-loop rollout | Imagining a whole sequence without checking reality |

---

## What's Next?

We have an imagination. Now we need to USE it. The next file,
`model_based_planning.py`, shows how to use the world model to pick **good
actions** in the real game — without doing any more learning. This is called
**model predictive control (MPC)**, and it is one of the oldest and most useful
ideas in control theory and modern model-based RL.
