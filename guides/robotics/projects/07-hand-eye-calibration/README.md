# Hand-Eye Calibration

## Key Insight

A camera bolted to a robot arm is useless until you know the exact [transform](/shared/glossary/#homogeneous-transform) between the camera's frame and the arm's frame, and you cannot measure that offset with a ruler. [Hand-eye calibration](/shared/glossary/#hand-eye-calibration) recovers it by moving the arm to many poses, recording how a fixed [tag](/shared/glossary/#apriltag) appears to shift in the camera, and solving the resulting `AX = XB` equation, where the unknown `X` is the rigid camera-to-hand offset. Get it right and the arm can grasp whatever the camera sees; get it wrong by even a degree and every grasp drifts, which is why reporting the residual error is part of finishing the project.
