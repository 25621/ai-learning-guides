# Quadrotor Min-Snap

## Key Insight

Generate and track a [minimum-snap trajectory](/shared/glossary/#minimum-snap-trajectory) through a sequence of waypoints for a [quadrotor](/shared/glossary/#quadrotor) drone. Because a quadrotor's position and [yaw](/shared/glossary/#yaw) are [differentially flat](/shared/glossary/#differential-flatness), its 3D flight paths can be planned directly as smooth polynomials, bypassing full rotational dynamics. Minimizing snap—the fourth derivative of position over time—produces trajectories that avoid sudden accelerations, allowing the drone's motors to track the path precisely without saturating.
