# RRT in 2D

## Key Insight

In environments with complex or continuous obstacles, grid-based path search becomes computationally intractable as dimensions rise. The [RRT (Rapidly-exploring Random Tree)](/shared/glossary/#rrt) algorithm overcomes this by building a search tree that grows rapidly toward randomly sampled configurations, avoiding the need to pre-discretize the entire space. This project grows a 2D tree around obstacles to demonstrate how sampling-based planning explores large spaces quickly, while highlighting its tendency to produce jerky, suboptimal paths that require post-processing.
