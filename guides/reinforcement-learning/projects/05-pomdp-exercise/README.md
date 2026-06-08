# POMDP Exercise

## Key Insight

Every algorithm in this phase secretly assumes the [Markov property](/shared/glossary/#markov-property): that the current state holds everything you need to act well, so the past can be forgotten. When the agent can only see part of the state — here, its grid row but not its column — that assumption breaks and the problem becomes a [POMDP](/shared/glossary/#pomdp), where the best possible *memoryless* [policy](/shared/glossary/#policy) is provably worse than one that remembers. Building this on purpose teaches the failure mode behind a huge fraction of real-world RL bugs: the "state" you fed the agent was never actually Markov.
