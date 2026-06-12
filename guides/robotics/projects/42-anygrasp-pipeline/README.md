# AnyGrasp Pipeline

## Key Insight

Deploying [AnyGrasp](/shared/glossary/#anygrasp) on a physical manipulator creates a robust 3D grasping pipeline that works directly with raw [point clouds](/shared/glossary/#point-cloud) from a depth camera. The network processes the scene to perform [6-DoF pose estimation](/shared/glossary/#6-dof-pose-estimation) for thousands of candidate grasps, scoring each based on surface geometric features. Executing these grasps on a real arm demonstrates how data-driven models bypass the need for prior CAD models, enabling reliable picking of arbitrary, highly cluttered objects.
