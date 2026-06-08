# Hand-Trace Bellman Backups

## Key Insight

A [Bellman backup](/shared/glossary/#bellman-operator) replaces each state's current value estimate with "reward now plus discounted value of where you land next," and applying it over and over is guaranteed to converge because the backup is a [contraction mapping](/shared/glossary/#contraction-mapping) — every pass shrinks the gap to the true [value function](/shared/glossary/#value-function) by at least a factor of the [discount factor](/shared/glossary/#discount-factor) γ. Doing ten backups by hand on a three-state [MDP](/shared/glossary/#mdp) — a toy world with only three possible situations the agent can be in, small enough that you can write down and update every state's value on paper — makes that abstract guarantee concrete: you literally watch the numbers stop moving. This is the mechanism underneath [value iteration](/shared/glossary/#value-iteration) and, in disguise, underneath every value-based deep-RL algorithm.
