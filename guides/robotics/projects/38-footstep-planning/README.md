# Footstep Planning

## Key Insight

For legged robots walking on rough or discontinuous terrain, planning continuous whole-body trajectories is highly complex and prone to collision. [Footstep planning](/shared/glossary/#footstep-planning) simplifies this by first planning a discrete sequence of stable foot placements on a footstep lattice using an [A* search](/shared/glossary/#a-star-search). This project implements footstep planning for a planar biped crossing stepping stones to demonstrate how discrete graph search can solve complex locomotion problems before continuous joint trajectories are generated.
