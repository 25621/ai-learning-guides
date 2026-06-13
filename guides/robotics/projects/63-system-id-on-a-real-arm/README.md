# System ID on a Real Arm

## Key Insight

[System identification](/shared/glossary/#system-identification) bridges the gap between theoretical models and real hardware by fitting physical parameter values—such as joint friction, link masses, and motor torque constants—from experimental trajectories. By exciting the arm's joints with custom trajectories like [chirp signals](/shared/glossary/#chirp), we gather input-output data to optimize these parameter estimates. This prevents control issues like overshoot or steady-state error that occur when running [impedance control](/shared/glossary/#impedance-control) on inaccurate default values.
