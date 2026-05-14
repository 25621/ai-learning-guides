# Reward Modeling: Teaching A Computer What People Prefer

## The Big Idea

A reward model is a small judge. You show it two answers to the same
question, tell it which one a person liked better, and over time it learns
to give a higher score to answers that people would prefer.

Why do we need such a judge? Because most of what we want from a language
model is hard to write down as a math formula. There is no single equation
for "helpful," "polite," or "well written." But people can almost always
point to the better of two options. The reward model turns those simple
"this one is better" votes into a score that a learning algorithm can use.

## A Real-Life Analogy

Imagine teaching a friend to bake brownies.

You do not hand them a 50-page rule book on "what makes a good brownie."
Instead, you taste two batches and say:

"This one is better."

After a few rounds of this, your friend starts to notice patterns. Maybe
the fudgier one always wins. Maybe the over-baked one always loses. Your
friend builds a mental scoring system from your comparisons.

A reward model does exactly this, but with numbers. It does not need to
know *why* the chosen answer is better. It just needs lots of "this beats
that" examples and it gradually learns a score that lines up with the
preferences.

## How The Learning Works (Intuition Only)

Each example is a triple: a prompt, a **chosen** response, and a
**rejected** response. We want the model to give a higher score to the
chosen one than to the rejected one - by any margin.

The training nudge is simple in spirit:

- Score of chosen too low? Push it up.
- Score of rejected too high? Push it down.
- Already in the right order with a clear gap? Leave them alone.

That nudge is called the Bradley-Terry loss, and it is the standard recipe
in modern RLHF systems.

## What The Experiment Shows

We trained a reward model on 2,000 synthetic preference pairs. The plot
below shows three views of the same training run.

![Reward model training](outputs/reward_modeling.png)

- **Left** - the loss falls quickly. The model is becoming more confident
  about its rankings.
- **Middle** - preference accuracy climbs to nearly 100%. On almost every
  pair, the chosen response gets a higher score than the rejected one.
- **Right** - the score distributions for chosen vs rejected responses pull
  apart. At the start they overlapped; after training, chosen responses
  sit clearly to the right.

That separation is the whole point. A later step (PPO or DPO) can now use
this score as a target to optimize toward.

## Where This Sits In The RLHF Pipeline

The roadmap describes RLHF as "aligning models with human preferences."
The reward model is step one of three:

1. **Reward model (this file)** - convert preference votes into a score.
2. **PPO fine-tuning** - push the language model toward higher scores
   while staying close to its original behavior.
3. **DPO** - a newer shortcut that skips the reward model entirely.

So reward modeling is the bridge between *human judgment* and
*machine optimization*. Get this bridge wrong and every downstream step
gets steered off course.

## Why This Matters Outside The Lab

The same idea shows up in many places:

- **Recommendation systems** learn what you like from clicks, skips, and
  time spent watching.
- **Search engines** learn ranking from "which result did you click."
- **Restaurants** learn the popular dishes from repeat orders, not from
  customers writing essays about what they liked.

Whenever it is easier to *compare* than to *rate*, a reward model is the
right tool.

## One-Sentence Summary

**A reward model is a learned judge that turns "this one is better"
preferences into a numeric score the rest of RLHF can optimize.**
