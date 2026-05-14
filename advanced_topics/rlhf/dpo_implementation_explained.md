# DPO: Skipping The Judge And Going Straight To The Source

## The Big Idea

Classic RLHF has two stages: first train a reward model, then use PPO to
chase its scores. DPO (Direct Preference Optimization) asks a clever
question:

*If the reward model is just an intermediate step, can we skip it?*

It turns out: yes. DPO trains the language model directly from preference
pairs, with no separate judge, no PPO sampling loop, and no KL coefficient
to tune. It uses one elegant formula and behaves like supervised learning.

This makes DPO simpler to run, more stable, and faster - which is why it
has rapidly become the default choice for many open-source aligned models.

## A Real-Life Analogy

Suppose you are coaching a student to write essays.

The PPO approach is: hire a teacher to grade essays, then have the student
write essay after essay and adjust based on the teacher's grades.

The DPO approach is: show the student two essays at a time and say,
"this one is better - lean toward writing like this one, away from that
one." No teacher in the middle. The student adjusts directly from
comparisons.

Both can work. DPO usually finishes faster because nobody has to train and
maintain a separate teacher.

## How The Learning Works (Intuition Only)

DPO uses the same preference pairs as reward modeling: prompt, chosen,
rejected. For each pair, it asks two questions:

1. Has the model become **more likely** to produce the chosen response
   than the reference model would have been?
2. Has the model become **less likely** to produce the rejected response
   than the reference model would have been?

The training pushes both numbers in the right direction at once. Crucially,
the reference model is always there in the comparison - it plays the same
role as the KL penalty in PPO. The model is allowed to shift, but the
shifts are always *relative to* the starting point.

A subtle and beautiful result from the DPO paper is that this single loss
function is mathematically equivalent to "train a reward model, then run
PPO with a KL penalty." Same destination, simpler journey.

## What The Experiment Shows

We trained a policy directly on 2,000 preference pairs for 300 epochs.

![DPO training](outputs/dpo_implementation.png)

- **Left** - the DPO loss drops as the model learns to prefer chosen over
  rejected responses.
- **Middle** - preference accuracy (how often the policy assigns a higher
  implicit reward to the chosen response) climbs to about 99%.
- **Right** - the implicit reward margin grows. DPO never names a "reward,"
  but the gap between log-probabilities of chosen vs rejected, scaled by
  beta, can be read as one. It widens steadily, meaning the model becomes
  more confident in its preferences.

Notice how clean this looks compared to PPO. There is no sampling loop, no
exploration noise, and no separate reward model running. Each epoch is a
pure supervised-style update over the preference dataset.

## Where This Sits In The RLHF Pipeline

DPO is an *alternative* to steps two of the classic pipeline:

- **Classic:** preferences → reward model → PPO → aligned model.
- **DPO:** preferences → aligned model. (Done.)

The catch is that DPO trains on a fixed preference dataset. PPO, because
it samples fresh responses each round, can in principle explore further.
In practice DPO wins for most "align on a curated preference dataset" use
cases.

## Why This Matters Outside The Lab

The "skip the middle measurement" pattern is everywhere:

- A coach correcting a swimmer's form by demonstrating side-by-side rather
  than scoring each lap with a stopwatch.
- A photographer editing two photos at a time, picking the better one,
  instead of building a "good photo" scoring rubric.
- A hiring manager comparing two résumés rather than scoring each one
  against a 30-point checklist.

When you only need to *rank*, you do not need an absolute scale. DPO is
that insight applied to language models.

## One-Sentence Summary

**DPO turns preference pairs directly into a better model, with no reward
model in the middle - simpler than PPO, and often just as good.**
