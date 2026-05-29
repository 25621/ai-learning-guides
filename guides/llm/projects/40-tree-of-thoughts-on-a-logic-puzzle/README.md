# Tree-of-Thoughts on a Logic Puzzle

---

> Explore several lines of reasoning at once, and prune the dead ends.

---

## Key Insight

[Tree-of-Thoughts](/shared/glossary/#tree-of-thoughts) treats reasoning as a search: it branches into multiple partial solutions, scores them with a heuristic (here a [process reward model](/shared/glossary/#process-reward-model)), and expands only the promising branches. This project applies that search to a logic puzzle.

## Why This Matters

Hard problems often need backtracking — abandoning a wrong path and trying another. A tree search makes that explicit, letting the model recover from an early mistake instead of committing to its first guess.
