# Distill 7B → 1B

---

> Train a small student to imitate a big teacher.

---

## Key Insight

This project trains a 1B student model to imitate the output distribution of a stronger 7B teacher on a specific task domain — knowledge [distillation](/shared/glossary/#distillation) — and reports how much of the teacher's quality the much smaller student is able to retain.

## Why This Matters

Serving a 1B model costs a fraction of serving a 7B, so if the student can keep most of the teacher's behavior on the tasks you care about, distillation is the cheapest way to bring a capable model into a tight latency or budget — though it only works for skills the teacher already has, not for new abilities the teacher lacks.
