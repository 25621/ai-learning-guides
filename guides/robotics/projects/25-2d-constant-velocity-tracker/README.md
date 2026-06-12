# 2D Constant-Velocity Tracker

## Key Insight

Tracking a moving object requires predicting its motion while correcting for noisy position measurements. A [Kalman Filter (KF)](/shared/glossary/#kf) uses a [constant-velocity model](/shared/glossary/#constant-velocity-model) to balance the physical expectation of momentum against incoming sensor updates. Tuning the ratio of process noise (how much the object's speed actually changes) to measurement noise (sensor error) shapes the steady-state gain, determining whether the filter responds immediately to changes or filters out high-frequency jitter.
