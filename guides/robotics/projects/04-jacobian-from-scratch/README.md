# Jacobian From Scratch

## Key Insight

The [Jacobian](/shared/glossary/#jacobian) is the matrix that turns "how fast each joint is turning" into "how fast the hand is moving and rotating," and it is the single most reused object in robot control. This project builds it two independent ways — analytically from the [kinematics](/shared/glossary/#kinematics), and numerically by nudging each joint a hair and measuring how the [end-effector](/shared/glossary/#end-effector) responds (a [finite difference](/shared/glossary/#finite-difference)) — then demands the two agree to six decimal places, a cross-check that exposes the sign and frame mistakes a single method would quietly hide. Once you trust your Jacobian, velocity control, force control, and [inverse kinematics](/shared/glossary/#inverse-kinematics) all reduce to a few lines of linear algebra.
