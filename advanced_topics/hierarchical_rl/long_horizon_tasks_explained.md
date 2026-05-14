# Long-Horizon Tasks: Solving The Faraway Reward Problem

## The Big Idea

A **long-horizon task** is a task where the reward is far away from the early
actions that make it possible.

This is hard for RL because the agent may need to do many correct things before
anything good happens. If it only gets a reward at the very end, it may never
discover which early choices mattered.

In the foundations phase, rewards helped the agent learn. In long-horizon
tasks, the problem is that the useful reward can be too delayed.

## A Treasure Hunt Analogy

Imagine a treasure hunt:

1. Find the key.
2. Use the key to open the door.
3. Walk through the door.
4. Reach the treasure.

If a child only gets praise after finding the treasure, they may give up before
learning that the key mattered.

But if we also say "good, you found the key" and "good, you opened the door,"
the task becomes easier to learn. The final goal is the same, but the path has
meaningful milestones.

That is the hierarchical idea: break a distant goal into useful subgoals.

## What The Experiment Shows

The task is a simple chain:

start -> key -> door -> treasure

The flat learner only cares about the final treasure. The hierarchical learner
gets help from the natural milestones: key first, door next, treasure last.

![Long-horizon task result](outputs/long_horizon_tasks.png)

The green curve rises because the milestone-guided learner finds the full
solution. The red curve stays low because the flat learner rarely connects the
early key action to the faraway treasure reward.

The bottom panel shows the learned route through the milestones.

## How This Connects To The Roadmap

The README builds from simple RL toward advanced topics. Long-horizon tasks are
one reason that advanced methods are needed.

Basic Q-learning can work when rewards are frequent and the task is short. But
when success requires a long chain of preparation, the agent benefits from
structure:

- options can represent reusable routines
- goals can tell lower-level policies what to reach
- milestones can make faraway success easier to discover

This is why Hierarchical RL appears in the advanced section.

## Why This Matters

Many important tasks are long-horizon tasks.

Real examples:

- a robot cleaning a kitchen
- a game character preparing for a boss fight
- a delivery robot navigating a building
- a software agent completing a multi-step workflow

The final reward may come late, but the agent still needs to learn which early
steps are useful.

Hierarchy gives the agent stepping stones.

## One-Sentence Summary

**Long-horizon tasks are hard because the reward is far away; hierarchy helps by
turning one distant win into a chain of understandable milestones.**
