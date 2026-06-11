# 2D LiDAR SLAM

## Key Insight

Robots operating in unknown environments must construct a map while simultaneously tracking their position within it. This 2D [LiDAR](/shared/glossary/#lidar) [SLAM](/shared/glossary/#slam) system pairs a front-end that matches scan points using algorithms like [Iterative Closest Point (ICP)](/shared/glossary/#iterative-closest-point-icp) to estimate incremental motion with a back-end that maintains a [pose graph](/shared/glossary/#pose-graph). By detecting [loop closures](/shared/glossary/#loop-closure)—recognizing when the robot has returned to a previously visited location—the system injects spatial constraints that correct accumulated [drift](/shared/glossary/#drift), aligning the entire map and trajectory.
