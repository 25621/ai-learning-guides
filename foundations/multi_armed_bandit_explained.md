# The Multi-Armed Bandit Problem 🎰

## What Is It? (For Curious Kids!)

Imagine you are at a birthday party and there are **10 different candy jars**.
Each jar has candy inside, but some jars have *yummy* candy and some jars have
*not-so-yummy* candy. You don't know which jar is the best — you have to try them!

Every time you reach into a jar, you get some candy. Your job is:

> **Get as much yummy candy as possible!**

That's the Multi-Armed Bandit problem! Instead of candy jars, scientists call
them "arms" (like arms on a slot machine). Each arm gives you a prize, but the
prizes are different each time.

---

## The Big Question: Should I Try New Jars or Stick With My Favorite?

This is the hardest part! Let's say you tried Jar #3 and it was pretty good.
Now you have a choice:

- **Exploit**: Keep picking Jar #3 because you already know it's good.
- **Explore**: Try a new jar — maybe Jar #7 is even *better*!

If you only ever pick the first jar you liked, you might miss the super-yummy
jar. But if you *always* try new jars, you never use what you already learned!

**Real-life example:** Think about your favorite restaurant. You always order
chicken nuggets (exploit!), but maybe the pizza is even better. If you never
try anything new, you'll never know!

---

## The Epsilon-Greedy Strategy

A smart way to solve this is called **epsilon-greedy** (epsilon is just the
Greek letter ε, said like "ep-sih-lon"):

1. **Most of the time (say 90%)**: Pick the jar you *think* is the best.
2. **Sometimes (say 10%)**: Pick a *random* jar to explore!

The 10% exploring trips help you discover better jars. The 90% exploiting
trips let you use what you already learned.

---

## What Our Code Found

We tested 10 arms (candy jars) with 200 different kids, 1000 picks each:

| Strategy | % of Time Choosing the Best Jar |
|----------|----------------------------------|
| **Never explore (ε=0)** | 14.5% — got stuck early, never found the best! |
| **Explore 1% of the time (ε=0.01)** | 37.6% — slowly found the best jar |
| **Explore 10% of the time (ε=0.10)** | **74.2%** — learned quickly, picked the best most of the time! |

**Lesson**: A little bit of exploration goes a long way!

---

## Real-Life Examples

- **Netflix recommendations**: Should Netflix show you a movie you'll probably
  like (exploit) or suggest something new (explore)?
- **Doctor choosing a treatment**: Use the treatment that usually works (exploit)
  or try a new one that might be even better (explore)?
- **A bee finding flowers**: Should it keep visiting the flowers it knows have
  nectar, or fly to a new field?

---

## Key Words to Remember

- **Arm**: One of the choices (like a candy jar)
- **Reward**: What you get when you pick an arm (like candy)
- **Exploit**: Use what you already know is good
- **Explore**: Try something new to learn more
- **Epsilon (ε)**: The chance you explore instead of exploit

The big idea: **You have to balance trying new things with using what you know!**
