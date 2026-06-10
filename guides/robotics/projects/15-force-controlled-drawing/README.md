# Force-Controlled Drawing

## Key Insight

Drawing on a curved surface you have not measured exactly is impossible with pure position control — aim the tip a hair too deep and you snap the pen, a hair too shallow and it lifts off the paper. The fix is to stop commanding *where* the tip is and start regulating *how hard* it presses: [force control](/shared/glossary/#force-control), realized here with [impedance control](/shared/glossary/#impedance-control), makes the arm soft (highly [compliant](/shared/glossary/#compliance)) along the direction into the surface so the pen rides the unknown curve at a steady, gentle force, while still tracking the drawing path in the other directions. This split — stiff where you know the geometry, soft where you do not — is the central idea of every contact task, and why a robot can write on a surface it cannot see perfectly.
