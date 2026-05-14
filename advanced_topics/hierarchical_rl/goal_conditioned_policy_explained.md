# Goal-Conditioned Policies: One Skill, Many Destinations

## The Big Idea

A normal policy answers:

"What should I do from here?"

A **goal-conditioned policy** answers:

"What should I do from here, given where I want to end up?"

That small change is powerful. The agent does not need a separate policy for
every destination. It learns one flexible behavior that changes depending on the
goal.

## A Road Trip Analogy

Imagine using a map app.

The app does not have one brain for "drive to the grocery store" and a totally
different brain for "drive to the library." It uses the same navigation skill,
but the route changes when the destination changes.

That is the heart of goal-conditioned RL.

The current location matters, but the destination matters too.

## What The Experiment Shows

The grid has many possible goals. During training, the desired goal changes from
episode to episode. The agent learns to read the goal as part of the problem.

![Goal-conditioned policy result](outputs/goal_conditioned_policy.png)

The left chart shows that success becomes reliable. The two route pictures use
the same starting area but different goals. The policy changes its route because
the destination changed.

This is the key visual proof: the agent did not memorize one fixed path. It
learned a reusable "go to the requested place" skill.

## How This Connects To The Roadmap

Earlier phases focus on the basic RL loop:

- the agent observes a state
- chooses an action
- receives a reward
- improves its policy

Goal-conditioned RL keeps that loop, but gives the agent a clearer question:

"Reach this goal."

That connects naturally to the README's advanced Hierarchical RL section.
Higher-level systems often create goals for lower-level policies. For example,
a high-level planner might say "go to the door," then "go to the key," then
"go to the exit." A goal-conditioned lower-level policy can carry out each of
those instructions.

## Why This Matters

Many real tasks are not one-task-only problems.

A robot should not need a separate brain for every object in a room. A game
agent should not need a separate brain for every item on the map. A navigation
agent should not need a separate brain for every address.

Goal-conditioned policies make behavior reusable.

Real examples:

- "pick up that cup"
- "walk to that room"
- "move the cursor to that button"
- "drive to that charging station"

The goal changes, but the underlying skill is shared.

## One-Sentence Summary

**A goal-conditioned policy is like a navigation app: the skill stays the same,
but the destination tells it what route to take.**
