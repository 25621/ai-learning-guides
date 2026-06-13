# Long-Horizon Eval

## Key Insight

Evaluating robot policies on short tasks can lead to a false sense of security because small execution errors compound exponentially over extended sequences. In [long-horizon autonomy](/shared/glossary/#long-horizon-autonomy), a single failure in a multi-step sequence (such as a 50-step assembly task) invalidates the entire run, making recovery behaviors critical for success. Developing a rigorous [evaluation harness](/shared/glossary/#evaluation-harness) that specifically measures how error rates compound across sequential phases allows engineers to identify and resolve fragile points in the control pipeline.
