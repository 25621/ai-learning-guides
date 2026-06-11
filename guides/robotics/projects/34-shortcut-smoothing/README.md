# Shortcut Smoothing

## Key Insight

Because sampling-based planners like [RRT](/shared/glossary/#rrt) select random directions, the raw paths they produce are jagged, jerky, and inefficient for real motors to execute. Shortcut smoothing is a simple post-processing technique that randomly picks two distant points along the planned path and attempts to connect them with a straight line in [C-space](/shared/glossary/#c-space). This project implements this shortcutting pass to show how iteratively replacing jagged segments with collision-free shortcuts reduces overall path length and removes unnecessary joint movements.
