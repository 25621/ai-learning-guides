# Hand-counted FLOPs

## Key Insight

Counting [FLOPs](/shared/glossary/#flops) by hand reveals where the computational weight of a [transformer](/shared/glossary/#transformer) model actually lies. By calculating the cost of [matrix multiplications](/shared/glossary/#matmul) in the [attention](/shared/glossary/#attention) and [MLP](/shared/glossary/#mlp) blocks, we see that the vast majority of operations are simple linear projections. This mathematical exercise sets the baseline for analyzing hardware efficiency and resource utilization.
