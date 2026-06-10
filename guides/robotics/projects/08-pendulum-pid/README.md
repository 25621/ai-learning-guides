# Pendulum PID

## Key Insight

A [pendulum](/shared/glossary/#pendulum) is the smallest system where you can watch every term of a [PID controller](/shared/glossary/#pid) earn its keep: the proportional term pulls the pole back toward the upright [setpoint](/shared/glossary/#setpoint), the derivative term damps the swinging so it stops overshooting, and the integral term erases any steady lean left by gravity or a miscalibrated model. This project stabilizes the pole *once it is already near upright* — holding a balance point, which PID does well — rather than the harder swing-up from hanging, where a weak motor must pump energy in over several swings and plain PID is not enough on its own. Plotting the [step response](/shared/glossary/#step-response) — how the angle reacts when you suddenly command a new target — is how you read off, in one picture, whether your gains are too sluggish, too twitchy, or just right.
