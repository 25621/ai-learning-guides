# Visual Odometry

## Key Insight

[Visual odometry (VO)](/shared/glossary/#visual-odometry) estimates how a camera moved through the world using nothing but its own video, by tracking [features](/shared/glossary/#feature) — distinctive corners and textures — from frame to frame and back-solving the camera motion that explains how they shifted. Because each step's motion is measured relative to the previous frame, small errors compound into [drift](/shared/glossary/#drift): travel 100 meters and your estimated path may be off by several, even though every single step looked accurate. This is why reporting drift over a fixed distance is the standard scorecard, and why a full SLAM system adds [loop closure](/shared/glossary/#loop-closure) — recognizing a previously visited place — to snap the accumulated error back down.
