# Stereo Depth

## Key Insight

A single camera cannot see depth, but two cameras a known distance apart can — exactly how your two eyes do it. The same object appears shifted sideways between the left and right images, and that shift, called [disparity](/shared/glossary/#disparity), is large for near objects and small for far ones. [Stereo vision](/shared/glossary/#stereo-vision) measures the disparity for every pixel and converts it to distance by [triangulation](/shared/glossary/#triangulation) — intersecting the two lines of sight in 3D — producing a [depth map](/shared/glossary/#depth-map) you can lift into a [point cloud](/shared/glossary/#point-cloud). The catch this project teaches: depth precision falls off with the square of distance, so a rig accurate to a millimeter up close may be off by centimeters across the room.
