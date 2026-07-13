# Sample Efficiency Study

## Key Insight

[Sample efficiency](/shared/glossary/#sample-efficiency) asks how much reward an algorithm buys per unit of environment experience, and it cleanly splits the two camps of deep RL: [off-policy](/shared/glossary/#off-policy) methods like [SAC](/shared/glossary/#sac) replay every transition from a [buffer](/shared/glossary/#experience-replay) many times, so they reach good policies in far fewer samples, while [on-policy](/shared/glossary/#on-policy) methods like [PPO](/shared/glossary/#ppo) must throw data away after each update and make up for it with massive parallelism and fast simulation. Running PPO and SAC on the same [MuJoCo](/shared/glossary/#mujoco) task and plotting [return](/shared/glossary/#return) against both samples used and wall-clock time makes the trade-off concrete: SAC usually wins on samples, while PPO often wins on wall-clock time when the simulator is cheap and you can run thousands of environments at once.

---

## What's in this directory

| File | Role |
|------|------|
| `sample_efficiency.py` | Runs SAC and PPO on the *same* task, 3 [seeds](/shared/glossary/#seed) each, and plots the same runs against two different x-axes. |

```bash
python3 sample_efficiency.py     # ~6 min on 12 hyperthreads
```

Neither algorithm is reimplemented here. SAC is
[`cc_lib.py`](../26-ddpg-on-pendulum/cc_lib.py) from project 26; PPO is
[`ppo.py`](../22-ppo-from-scratch/ppo.py) from project 22, configured for continuous
control. That reuse is what makes the comparison trustworthy: both are code the guide
has already tested, so any difference below is the *algorithms* differing, not two
implementations of varying quality.

## Two questions that sound like one

"Which algorithm is more efficient?" hides two completely different questions, and they
have different answers:

| the question | the resource you are short of | who cares |
|---|---|---|
| Which learns more **per environment step**? | *samples* | a real robot, a medical trial — anything with a physical clock and parts that wear out |
| Which learns more **per second**? | *seconds* | a fast simulator you can fork across 64 cores |

A paper reporting only the first is answering a question its readers may not have
asked. So this project plots **the same runs** against both x-axes and lets the
disagreement show.

All of it follows from one structural asymmetry — the
[update-to-data ratio](/shared/glossary/#update-to-data-ratio):

- **SAC** does **one gradient update per environment step**. Every step is expensive,
  but each sample is reused many times out of the replay buffer.
- **PPO** collects 2,048 steps, does a few dozen updates on that batch, then **throws
  it away**. Every step is cheap, but each sample teaches it far less.

## The result

![sample vs wall-clock](outputs/sample_vs_wallclock.png)

Same six runs, same task ([HalfCheetah](/shared/glossary/#halfcheetah), where a random
policy scores about `-280`). Only the x-axis changes.

```
HalfCheetah-v5  (threshold = 800)
  algo     env steps    seconds   final return   steps/s
  SAC         24,000        203           1969       121
  PPO        235,520        210            900      1124
```

**Per sample, SAC is 10× better.** It reaches the threshold in 24,000 steps; PPO needs
235,520. On a physical robot collecting one step per control tick, that is the
difference between an afternoon and two weeks.

**Per second, they arrive at the same moment.** SAC crosses the threshold at 203
seconds, PPO at 210. A 3% gap — a tie.

The explanation is the last column: **PPO runs 9.3× more environment steps per second**
(1,124/s vs 121/s), because it is not paying for a gradient update on every single
step. SAC's 10× sample advantage and PPO's 9.3× speed advantage very nearly *cancel*,
and the race to a working policy ends in a dead heat.

Do not memorize that cancellation — it is a coincidence of this machine. Understand
*why* it happens: the two ratios are the same trade, made in opposite directions. **SAC
spends compute to save samples. PPO spends samples to save compute.** Which one wins
depends entirely on which of the two is scarce for *you*, and that is a fact about your
problem, not about the algorithms.

## Where each one actually wins

Two things the table does not say, and both matter.

**SAC keeps going.** Look at the right-hand panel *past* the threshold. The two are
tied at 200 seconds, but SAC climbs on to `1969` while PPO sits at `900`, rising
slowly. Equal time to a *mediocre* policy is not equal time to a *good* one. (SAC's
curve also shows a sharp dip around 250 seconds — one seed briefly collapsing, then
recovering. That is SAC being SAC: more sample-efficient, and less placid than PPO,
which plods up its curve almost boringly.)

**PPO's advantage is the one that scales.** PPO managed 1,124 steps/s on a *single CPU
core*, and its rollout collection is
[embarrassingly parallel](/shared/glossary/#vectorized-environment) — give it 64 cores
and it collects roughly 64× faster, sliding its whole wall-clock curve to the left. SAC
cannot do this: its bottleneck is the gradient update, and those are strictly
*sequential* (one update per step, each depending on the last). Handing SAC 64 cores
barely helps. **That is the real reason large-scale RL runs on PPO** despite being far
less sample-efficient — its scarce resource is the one you can simply buy more of.

## What to take away

> **SAC is 10× more sample-efficient. PPO is 9.3× faster per sample. Choose the one
> whose scarce resource matches yours.**

If samples cost money, wear out hardware, or take real-world time — a robot arm, a
recommender experimenting on live users, anything physical — you want the off-policy,
replay-buffer, high-[UTD](/shared/glossary/#update-to-data-ratio) family: SAC,
[TD3](/shared/glossary/#td3), and their descendants. If samples are nearly free and you
have cores to burn, PPO's simplicity and parallelism win, and its large appetite for
samples is not a flaw you need to fix.

Neither is "better". They answer different questions — and the most common mistake in
applied RL is inheriting another team's algorithm along with their answer, without
noticing that your constraint was never the same as theirs.
