# Domain Randomization Study

## Key Insight

[Domain randomization](/shared/glossary/#domain-randomization) is the primary technique for bridging the [sim-to-real](/shared/glossary/#sim-to-real) transfer gap, training policies in a simulator whose physical parameters are dynamically varied to prevent overfitting to simulated physics. By measuring the policy's success rates in a held-out test simulation with extreme physics values, this study quantifies the performance drop caused by the [reality gap](/shared/glossary/#reality-gap) and how randomization cures it. The key is that by exposing the policy to a wide distribution of masses, frictions, and latencies during training, the robot learns a robust control strategy that generalizes to physical hardware without requiring system identification.
