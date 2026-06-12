# VIO MVP

## Key Insight

Fusing high-rate inertial measurements with low-rate visual features produces a [state](/shared/glossary/#state) estimator that is both high-frequency and low-drift. This visual-inertial odometry ([VIO](/shared/glossary/#vio)) system uses an error-[state](/shared/glossary/#state) [Kalman Filter](/shared/glossary/#kf) to estimate the nominal motion using the [IMU](/shared/glossary/#imu) while using camera feature tracks to estimate and correct the IMU's sensor biases. Fusing these complementary modalities solves the scale ambiguity of monocular cameras and the rapid divergence of pure inertial navigation, providing the core navigation backbone for drones and AR/VR headsets.
