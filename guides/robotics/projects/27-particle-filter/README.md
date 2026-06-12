# Particle Filter

## Key Insight

When a robot is completely lost or faces ambiguous sensor readings, it cannot rely on single-hypothesis estimators like the [Kalman Filter](/shared/glossary/#kf). A [particle filter](/shared/glossary/#particle-filter) represents the robot's belief using a swarm of weighted samples, allowing it to track multiple candidate locations simultaneously across a known map. As the robot moves and senses the environment, the resampling step continuously duplicates particles in high-probability areas and discards those in impossible locations, eventually collapsing the multi-modal distribution down to the correct unique position.
