# Impedance Control

## Key Insight

Position control fights the world: tell the arm to be at a spot and it will push arbitrarily hard to get there, snapping anything in the way. [Impedance control](/shared/glossary/#impedance-control) instead commands a *virtual spring-damper* between the [end-effector](/shared/glossary/#end-effector) and a moving reference, so the arm pushes back gently in proportion to how far it has been displaced — you set the spring's stiffness, and that directly sets how much the arm yields under contact, its [compliance](/shared/glossary/#compliance). Pushing on the simulated arm and watching it give, then tuning the virtual stiffness from soft to stiff, is the core skill behind every contact-rich task: insertion, polishing, and working safely next to people.
