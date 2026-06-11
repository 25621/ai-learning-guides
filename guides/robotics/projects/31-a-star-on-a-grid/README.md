# A* on a Grid

## Key Insight

Finding the shortest path on a 2D grid using [Dijkstra's algorithm](/shared/glossary/#dijkstras-algorithm) is slow because it searches in every direction equally, wastefully visiting cells that lead away from the target. The [A* search](/shared/glossary/#a-star-search) algorithm solves this by guiding the search with a heuristic function, such as the Manhattan distance, which estimates the remaining distance to the goal. This project implements A* and compares it against Dijkstra's algorithm to show how an admissible heuristic dramatically cuts the number of visited states while still guaranteeing an optimal, collision-free route.
