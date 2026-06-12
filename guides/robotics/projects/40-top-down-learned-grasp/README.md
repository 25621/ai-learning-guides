# Top-Down Learned Grasp

## Key Insight

For unstructured environments where object geometry is unknown, robots use a [grasp-quality network](/shared/glossary/#grasp-quality-network) trained on visual data to predict the success of candidate grasps. By feeding [depth maps](/shared/glossary/#depth-map) into the network, the robot can evaluate hundreds of candidate [antipodal grasps](/shared/glossary/#antipodal-grasp) across an object's surface in real-time. This data-driven approach allows the [gripper](/shared/glossary/#gripper) to successfully grasp novel, arbitrary objects without needing explicit 3D computer models of each item.
