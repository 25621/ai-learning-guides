# EKF Localization

## Key Insight

Mobile robots must track their positions using internal motion estimates combined with external sensor readings. An [Extended Kalman Filter (EKF)](/shared/glossary/#ekf) localization algorithm handles nonlinear kinematics—such as wheel rotations translating to 2D coordinates—by linearizing the motion and measurement equations around the current state estimate. This allows the robot to fuse wheel [odometry](/shared/glossary/#odometry) with landmark range and bearing measurements to limit the compounding [drift](/shared/glossary/#drift) of [dead reckoning](/shared/glossary/#dead-reckoning).
