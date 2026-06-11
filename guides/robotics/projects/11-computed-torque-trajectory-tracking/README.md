# Computed-Torque Trajectory Tracking

## Key Insight

[Computed-torque control](/shared/glossary/#computed-torque-control) uses a model of the arm's own physics — the [manipulator equation](/shared/glossary/#manipulator-equation) — as a [feedforward](/shared/glossary/#feedforward-control) term that cancels its nonlinear dynamics in advance, so the leftover error behaves like a simple, decoupled linear system that a light [PID](/shared/glossary/#pid) correction can clean up. Tracking a sinusoidal joint trajectory on a 6-DoF arm and comparing PID-only against feedforward-plus-PID makes the lesson concrete: the model-based prediction does the heavy lifting, and feedback only mops up what the model got slightly wrong. This is why a *good enough* dynamics model, not an ever-higher feedback gain, is what separates a wobbly tracker from a crisp one.
