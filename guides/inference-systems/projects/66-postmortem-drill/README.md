# Postmortem Drill

---

> "Practice the outage on a calm day so the real one is survivable."

---

## Key Insight

This project injects a real failure — a crashed replica or a thrashing [KV cache](/shared/glossary/#kv-cache) — then runs the incident end to end and writes a [postmortem](/shared/glossary/#postmortem): a blameless account of what broke, how it was detected, and what will keep it from happening again.

## Why This Matters

Systems fail; teams that rehearse failure recover faster and calmer when it happens for real. A good postmortem turns one painful outage into permanent lessons, so the same problem does not bite twice.
