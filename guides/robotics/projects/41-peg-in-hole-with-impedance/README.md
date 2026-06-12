# Peg-in-Hole with Impedance

## Key Insight

Robotic insertion tasks with tiny clearances, such as [peg-in-hole](/shared/glossary/#peg-in-hole) assembly, will fail and jam under rigid position control due to small alignment errors. By employing software [compliance](/shared/glossary/#compliance) via [impedance control](/shared/glossary/#impedance-control), the robot acts as a virtual [spring-damper](/shared/glossary/#damper) system that yields to contact forces rather than fighting them. Combining this compliance with search strategies like spiral force-search allows the robot to locate the hole, align the parts, and complete the insertion smoothly and safely.
