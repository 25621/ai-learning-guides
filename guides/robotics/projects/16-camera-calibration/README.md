# Camera Calibration

## Key Insight

Before a camera can tell a robot *where* something is, you must measure the camera's own optics: its [focal length and principal point](/shared/glossary/#camera-intrinsics) and the way its lens bends straight lines into gentle curves, called [lens distortion](/shared/glossary/#lens-distortion). [Camera calibration](/shared/glossary/#camera-calibration) recovers these numbers by photographing a checkerboard of known square size from many angles and solving for the [pinhole camera model](/shared/glossary/#pinhole-camera-model) that best explains where every corner landed. The score you report — [reprojection error](/shared/glossary/#reprojection-error), the average pixel gap between where the model predicts a corner should appear and where it actually did — must drop below half a pixel, because every later step (depth, pose, grasping) inherits this error and can never be more accurate than the calibration beneath it.
