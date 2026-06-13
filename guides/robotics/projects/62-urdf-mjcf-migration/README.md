# URDF to MJCF Migration

## Key Insight

A robot's [URDF](/shared/glossary/#urdf--mjcf--usd) defines its visual and kinematic structure but often lacks the precise [inertia](/shared/glossary/#inertia) properties, contact parameters, and clean [collision meshes](/shared/glossary/#collision-mesh) needed for dynamic simulation. Migrating to [MJCF](/shared/glossary/#urdf--mjcf--usd) allows [MuJoCo](/shared/glossary/#mujoco)'s physics solver to accurately compute contact forces, joint limits, and friction. Converting and auditing these files ensures that simulation behaviors match reality, preventing simulator instabilities or phantom forces.
