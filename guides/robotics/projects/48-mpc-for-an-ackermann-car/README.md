# MPC for an Ackermann Car

## Key Insight

Control an [Ackermann steering](/shared/glossary/#ackermann-steering) car around a racetrack using a [kinematic bicycle model](/shared/glossary/#kinematic-bicycle-model) within a [Model Predictive Control (MPC)](/shared/glossary/#mpc) framework. Unlike simpler robots, a car-like vehicle has a minimum turning radius and cannot slide sideways, introducing [nonholonomic constraints](/shared/glossary/#nonholonomic-constraint) to the path tracking challenge. The MPC solver handles these constraints by optimizing steering inputs over a rolling time horizon, minimizing deviation from the centerline while respecting physical limits.
