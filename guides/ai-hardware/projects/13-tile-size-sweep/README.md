# Tile Size Sweep

## Key Insight

Tuning the tile size in [tiling](/shared/glossary/#tiling) algorithms balances memory reuse with resource constraints. By loading data into fast on-chip [shared memory](/shared/glossary/#shared-memory) or [registers](/shared/glossary/#registers), a tiled [matrix multiplication](/shared/glossary/#matmul) reduces slow reads from off-chip [HBM](/shared/glossary/#hbm). However, choosing larger tiles to increase data reuse consumes more shared memory per block, which can exceed hardware limits, reduce [occupancy](/shared/glossary/#occupancy), and lower overall performance.
