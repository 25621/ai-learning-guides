# Option-Critic Architecture

## The Big Idea: Working in Chapters, Not Word by Word

Imagine you are writing a novel. You don't plan every single word before you begin. Instead, you think in **chapters**: "Chapter 1 introduces the hero. Chapter 2 is the quest. Chapter 3 is the showdown." Within each chapter, you figure out the details as you go.

That is exactly how the Option-Critic architecture thinks about decisions.

---

## What Is a "Flat" Agent?

A normal RL agent (like the ones from Phase 3 and 4 of the curriculum) decides one action at a time, every single step. It's like a GPS that recalculates the entire route from scratch every time you move one meter. It works, but it's exhausting and slow to learn.

---

## What Is an "Option"?

An **option** is a **named skill** — a mini-policy that the agent can run for several steps in a row before handing control back.

Think of it like a manager delegating to specialists:

| Who | What they do |
|-----|-------------|
| **Manager (meta-policy)** | Decides *which* specialist to send on a job |
| **Specialist A** | Expert at navigating the top-left room |
| **Specialist B** | Expert at crossing doorways |
| **Specialist C** | Expert at charging toward the goal |
| **Specialist D** | Backup generalist |

The manager picks a specialist. The specialist works until they decide they're done (this is called **termination**). Then the manager picks again.

---

## The Three Moving Parts

Every option has three components — think of them as the specialist's **job description**:

1. **Initiation**: When can this specialist be called on? *(e.g., "Specialist A only activates near the top-left room.")*
2. **Intra-option policy**: What does the specialist do while they're working? *(e.g., "Walk toward the top-left corner.")*
3. **Termination**: When does the specialist hand back control? *(e.g., "Stop when you've reached a doorway.")*

The beauty of Option-Critic is that all three are **learned automatically** — you don't hand-craft the specialists. The algorithm figures out that it's useful to have one option for each room, or one for rushing to the goal, all on its own.

---

## A Day in the Life of an Option-Critic Agent

1. Agent enters a new room (state).
2. **Manager** looks at the room and picks an option — say, Option 2.
3. **Option 2's specialist** takes over: walks toward the doorway, step by step.
4. At some point, Option 2 says "I'm done here" (termination).
5. **Manager** wakes up, picks a new option for the new situation.
6. Repeat.

Compare this to the flat agent: flat agent agonizes over every single step. Option-Critic delegates whole stretches of behavior, letting each specialist get good at its narrow job.

---

## Why Does This Help?

In a maze, the agent needs to reach a goal that may be 30–50 steps away. With flat learning, every step on the path is equally "invisible" until the reward finally arrives at the end — that signal has to travel backwards through dozens of steps.

With options, the path breaks into **sub-tasks**. Each sub-task gets its own mini-reward signal (reaching the doorway, entering the next room). Learning propagates through shorter segments. **The agent learns faster on problems that require many steps.**

This is the core idea behind all of [Hierarchical RL](../../README.md#hierarchical-rl) — and Option-Critic is one of its cleanest implementations.

---

## What Our Code Does

The script `option_critic.py` puts an Option-Critic agent into a **7x7 gridworld** with a fixed goal. The agent starts anywhere in the grid and must navigate to the goal cell.

The agent has four options and must simultaneously learn:

- A policy for each option (where to walk)
- When to terminate each option (termination condition)
- A meta-policy for choosing between options

The reward uses **potential-based shaping** — the agent gets a small bonus each step it moves closer to the goal, on top of +1 for reaching it. This dense feedback makes learning stable enough to see the options working within 2,500 episodes.

No human ever tells it what each option should do. The algorithm discovers which areas of the grid each option specializes in.

---

## What the Charts Show

![Option-Critic Learning Curves](outputs/option_critic.png)

**Left — Shaped Return:** Higher return means the agent is reaching the goal more reliably *and* taking shorter paths (the shaping gives a bonus per step closer). The curve rising then stabilizing shows the options learning to coordinate.

**Right — Steps to Goal:** Fewer steps means the agent found a more efficient path. The downward trend shows the options maturing into coherent skills that guide the agent more directly toward the goal.

The smoothed curves show the general trend across 100-episode windows — some noise is normal in RL, especially when multiple components (options, termination, meta-policy) are learning simultaneously.

---

## One-Sentence Summary

> **Option-Critic teaches an agent to work in skills rather than single steps — a manager picks which specialist runs, each specialist does its job, and the whole system learns together from the same reward signal.**
