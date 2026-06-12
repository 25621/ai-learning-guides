# AMCL on a Known Map

## Key Insight

Localize a mobile robot within a known occupancy grid map using [Adaptive Monte Carlo Localization (AMCL)](/shared/glossary/#adaptive-monte-carlo-localization-amcl). By representing the robot's possible positions as a set of particles, the algorithm combines noisy wheel [odometry](/shared/glossary/#odometry) with [lidar](/shared/glossary/#lidar) scans to estimate the robot's pose. As the robot moves and detects features, the [particle filter](/shared/glossary/#particle-filter) resamples to converge on the true location, dynamically adjusting the particle count to balance estimation accuracy with real-time computational constraints.
