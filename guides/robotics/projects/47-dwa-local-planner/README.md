# DWA Local Planner

## Key Insight

Integrate a local obstacle-avoidance controller based on the [Dynamic Window Approach (DWA)](/shared/glossary/#dynamic-window-approach-dwa) with an [A* search](/shared/glossary/#a-star-search) global planner. While the global planner finds a static route to the target, the local planner dynamically selects velocity commands that avoid obstacles by projecting the robot's [state](/shared/glossary/#state) forward in time. This layered navigation stack ensures the robot can navigate complex rooms while reacting immediately to dynamic obstacles.
