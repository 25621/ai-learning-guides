# Cloth Folding

## Key Insight

Manipulating soft, flexible objects like cloth is a major challenge in robotics because they can wrinkle, fold, and stretch in infinite ways. [Deformable manipulation](/shared/glossary/#deformable-manipulation) tasks like cloth folding bypass complex 3D shape models in favor of pick-and-place primitive actions guided by visual feedback. By capturing top-down images of the workspace, a robot can plan grasp and release points on the cloth, measuring its folding accuracy by comparing the overlap between the final cloth shape and a target folded shape.
