# Open-Vocab Grasping

## Key Insight

Classical grasping needs a 3D model of every object in advance; [open-vocabulary perception](/shared/glossary/#open-vocabulary-perception) removes that limit by letting you name the target in plain language — "the red cup" — and having the system find it without ever being trained on cups specifically. The pipeline pairs a [VLM](/shared/glossary/#vlm) (or [CLIP](/shared/glossary/#clip)-style model) that grounds the phrase to an image region with [SAM (Segment Anything Model)](/shared/glossary/#sam-segment-anything-model), which carves out that object's exact pixels, then fits a simple top-down [antipodal grasp](/shared/glossary/#antipodal-grasp) — two contact points on opposite sides so the [gripper](/shared/glossary/#gripper) can pinch the object. The power here is generalization: the same code grasps a cup, a stapler, or a banana, because nothing in it is hard-coded to a fixed list of objects.
