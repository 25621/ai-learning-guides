# MPC for a Unicycle

## Key Insight

[Model Predictive Control (MPC)](/shared/glossary/#mpc) drives a robot by re-solving a short look-ahead optimization at every step — predict where each candidate sequence of controls would take you, keep the best, apply only the first control, then re-plan with fresh measurements. Here the prediction model is a [unicycle](/shared/glossary/#unicycle-model) (a point that can only roll forward and turn, never slide sideways — the [kinematic bicycle](/shared/glossary/#kinematic-bicycle-model) is the same idea with an explicit steering angle), and chasing a figure-8 forces the controller to honor that no-sideways-motion constraint while still tracking a curvy path. Writing the optimization in [CasADi](/shared/glossary/#casadi), which supplies the automatic derivatives and hands the problem to a numerical solver, is what makes each re-plan fast enough to run inside the control loop.
