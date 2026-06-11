# Object 6-DoF Pose

## Key Insight

Knowing an object is "a mug" is not enough to grasp it — the robot needs its full [6-DoF pose](/shared/glossary/#6-dof-pose): where it sits and how it is turned. [6-DoF pose estimation](/shared/glossary/#6-dof-pose-estimation) trains a network to predict that pose directly from an image or [point cloud](/shared/glossary/#point-cloud), and fine-tuning it on your own small object set is far cheaper than gathering enough data to train one from scratch. Success is scored with [ADD-S](/shared/glossary/#add-s), the average 3D distance between the object's points under the predicted pose versus the true pose — computed with a nearest-point match so that symmetric objects (a featureless cylinder looks identical when flipped) are not unfairly penalized for a "wrong" rotation that is actually indistinguishable.
