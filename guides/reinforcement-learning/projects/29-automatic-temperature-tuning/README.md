# Automatic Temperature Tuning

## Key Insight

[SAC](/shared/glossary/#sac) balances reward against exploration with a single knob, the entropy temperature α, and the algorithm is painfully sensitive to it: set it too high and the agent acts almost randomly forever, too low and it [collapses](/shared/glossary/#maximum-entropy-rl) to a brittle near-deterministic [policy](/shared/glossary/#policy). [Automatic temperature tuning](/shared/glossary/#automatic-temperature-tuning) removes the guesswork by treating α as something to learn — you pick a *target entropy* (how random you want the policy to be on average) and adjust α by gradient descent so the policy's actual [entropy](/shared/glossary/#entropy-regularization) is driven toward that target. This one change is what let SAC use a single configuration across wildly different tasks instead of re-tuning α by hand for every robot.
