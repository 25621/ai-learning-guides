# AprilTag Pose

## Key Insight

An [AprilTag](/shared/glossary/#apriltag) is a printed black-and-white square — a [fiducial marker](/shared/glossary/#fiducial-marker) — whose pattern encodes an ID the robot can detect instantly and unambiguously, making it the cheapest reliable way to drop a known landmark into a scene. Because the tag's real-world size and flat square shape are known in advance, locating its four corners in the image is enough to solve for its full [6-DoF pose](/shared/glossary/#6-dof-pose) — position and orientation — using the [Perspective-n-Point](/shared/glossary/#pnp-perspective-n-point) algorithm, which back-solves the camera geometry that maps the 3D corners to their 2D pixels. Projecting the tag's coordinate axes back onto the image is the honest visual check: if the drawn axes sit squarely on the tag, the recovered pose is right.
