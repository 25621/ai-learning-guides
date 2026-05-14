# Option-Critic: Learning Helpful Habits

## The Big Idea

Hierarchical RL is about learning at more than one level.

In earlier parts of this roadmap, the agent usually chose one tiny action at a
time: move up, move down, move left, move right. That works in small worlds, but
it can feel like planning a road trip by thinking about every single footstep.

The **Option-Critic** idea gives the agent reusable habits called **options**.
An option is like a mini-plan:

- when to start using it
- what small actions to take while using it
- when to stop and let the higher-level agent choose again

A human version is "walk to the elevator." That phrase hides many tiny actions:
turn, step forward, avoid people, press the button, wait, enter. You do not
think about each muscle movement separately. You use a familiar routine.

## A Manager Analogy

Imagine a manager running a small store.

The manager does not say:

"Move your left foot, move your right foot, pick up the box, turn 10 degrees..."

Instead, the manager says:

"Restock the shelf."

The employee handles the smaller steps. When the shelf is full, that job ends
and the manager gives a new instruction.

In Option-Critic:

- the **manager** is the high-level policy choosing an option
- the **employee** is the option's internal policy
- the **end of the task** is the option's termination rule

The useful part is that the agent learns these pieces together.

## What The Experiment Shows

The agent starts in one room and needs to reach a goal in another room. At the
beginning, it wanders. Over time, it learns a better routine for crossing the
rooms and reaching the goal.

![Option-Critic learning curve and route](outputs/option_critic.png)

The left chart shows the agent needing fewer steps as training continues. The
middle panel shows which option the agent prefers in different places. The right
panel shows the route it takes after learning.

The important lesson is not that the grid is hard. The important lesson is that
the agent is no longer only learning "which single action now?" It is also
learning "which routine should control the next stretch?"

## How This Connects To The Roadmap

The README introduces **agents, states, actions, rewards, policies, and value
functions** in the foundations phase. Option-Critic keeps all of those ideas but
adds one more layer:

- a normal policy chooses actions
- a hierarchical policy can choose options
- each option then chooses actions for a while

This is the same direction as the advanced topics in the roadmap: once basic RL
works, we ask how to make it handle longer, messier problems.

## Why This Matters

Flat RL can be wasteful because it repeatedly relearns tiny patterns. If an
agent discovers a useful routine, it should be able to reuse it.

Real examples:

- a robot learns "go to the charger"
- a game agent learns "collect health"
- a household assistant learns "clear the table"
- a navigation agent learns "leave the building"

Each routine may contain many small actions, but the higher-level planner can
think in larger chunks.

## One-Sentence Summary

**Option-Critic helps an agent learn its own reusable habits, so it can plan
with meaningful chunks instead of only tiny actions.**
