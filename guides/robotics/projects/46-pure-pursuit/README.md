# Pure Pursuit

## Key Insight

Implement a [pure pursuit](/shared/glossary/#pure-pursuit) path-tracking controller to steer a [differential drive](/shared/glossary/#differential-drive) robot along a pre-planned reference path. By finding a target point on the path at a defined look-ahead distance, the robot can compute its target angular velocity and follow curves smoothly. Tuning the look-ahead distance reveals the fundamental trade-off in path tracking: a short look-ahead tracks paths tightly but risks steering instability, while a long look-ahead provides smooth trajectories at the expense of cutting corners.
