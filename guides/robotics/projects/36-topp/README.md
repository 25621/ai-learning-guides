# TOPP

## Key Insight

A geometric path only dictates *where* the robot travels, not *when* or *how fast* it should move, leaving joint velocities and accelerations undefined. [TOPP (Time-Optimal Path Parameterization)](/shared/glossary/#topp) calculates the optimal velocity profile along a pre-planned path, ensuring the robot travels as fast as possible without violating its physical joint-velocity and acceleration limits. This project implements TOPP to parameterize a geometric path, demonstrating how velocity profile limits prevent actuator damage while maximizing motion efficiency.
