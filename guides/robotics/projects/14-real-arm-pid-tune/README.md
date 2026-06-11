# Real-Arm PID Tune

## Key Insight

A simulator hides the messiest part of real motors: friction. On a physical hobby arm the joints stick before they slip ([stiction](/shared/glossary/#stiction)) and drag once moving, so [PID](/shared/glossary/#pid) gains that looked perfect in simulation buzz or sag on hardware. This project has you tune the joint PIDs by feel or with a recipe like [Ziegler-Nichols](/shared/glossary/#ziegler-nichols), measure how much torque each joint loses to friction, and add a [friction-compensation](/shared/glossary/#friction-compensation) [feedforward](/shared/glossary/#feedforward-control) term that pre-cancels that drag — the single change that most often turns a sloppy real arm into a precise one.
