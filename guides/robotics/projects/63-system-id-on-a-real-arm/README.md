# System ID on a Real Arm

## Key Insight

[System identification](/shared/glossary/#system-identification) bridges the gap between theoretical models and real hardware by fitting physical parameter values—such as joint friction, [link masses](/shared/glossary/#link-mass), and motor torque constants—from experimental trajectories. By exciting the arm's joints with custom trajectories like [chirp signals](/shared/glossary/#chirp), we gather input-output data to optimize these parameter estimates. This prevents control issues like [overshoot](/shared/glossary/#step-response) or [steady-state error](/shared/glossary/#step-response) that occur when running [impedance control](/shared/glossary/#impedance-control) on inaccurate default values.
