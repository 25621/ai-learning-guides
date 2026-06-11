# Factor-Graph Practice

## Key Insight

Modern state estimation frames [SLAM](/shared/glossary/#slam) not as a step-by-step filtering problem, but as a global optimization problem over a [factor graph](/shared/glossary/#factor-graph). Using [GTSAM](/shared/glossary/#gtsam), we represent robot poses and sensor measurements as nodes and constraint factors, solving for the entire trajectory at once. To prevent incorrect measurements like false loop closures from ruining the map, robust cost kernels are applied to downweight outlier constraints, ensuring the optimizer converges to the correct trajectory despite noisy data.
