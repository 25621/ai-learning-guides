# Noisy-TV Experiment

## Key Insight

The [noisy-TV problem](/shared/glossary/#noisy-tv-problem) is the Achilles' heel of every prediction-error exploration method, and this project reproduces it on purpose. You drop a "television" into the environment that displays fresh random static every step; because the static is truly unpredictable, a curiosity agent that rewards itself for prediction error — like [RND](/shared/glossary/#rnd) or the [ICM](/shared/glossary/#icm) — earns a large [intrinsic reward](/shared/glossary/#intrinsic-reward) *every single time it looks at the screen*, so it sits and stares forever instead of exploring. Why it matters: it draws the sharp line between *novelty* (something genuinely new to learn) and mere *stochasticity* (randomness you can never learn), and it explains why later methods measure surprise in a learned, controllable feature space rather than over raw pixels.
