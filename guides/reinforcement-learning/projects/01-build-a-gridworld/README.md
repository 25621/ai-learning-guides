# Build a Gridworld

## Key Insight

A reinforcement-learning problem is fully described by five pieces — states, actions, [transition probabilities](/shared/glossary/#transition-function), a [reward function](/shared/glossary/#reward-function), and a [discount factor](/shared/glossary/#discount-factor) — bundled together as an [MDP](/shared/glossary/#mdp). A [gridworld](/shared/glossary/#gridworld) is the smallest environment where you can write all five down by hand and read them straight off a picture, so it turns abstract symbols into something you can point at. Building one yourself forces you to decide concretely what counts as a *state*, what the transition function does when you walk into a wall, and where the reward actually lives — the three spots where a beginner's mental model is usually fuzziest.
