# Aspect-Ratio Bucketing

## Key Insight

Most training pipelines center-crop every image to a square, which throws away the edges of tall portraits and wide landscapes — so the model never learns to compose anything but squares. [Aspect-ratio bucketing](/shared/glossary/#aspect-ratio-bucketing) fixes this by sorting images into groups by shape and building each [batch](/shared/glossary/#batch) from a single group, since a batch must share one tensor [shape](/shared/glossary/#shape). After training this way the model generates correctly-framed portraits and landscapes on demand, and you can visibly see composition improve on non-square test prompts.
