# Eval Harness

## Key Insight

A robust [evaluation harness](/shared/glossary/#evaluation-harness) automates regression testing for robot controllers and planning policies by running hundreds of diverse simulation tasks nightly. By seeding variations in object placement, physics parameters, and initial states, the harness calculates a statistical pass-rate dashboard. This continuous evaluation catches subtle bugs or performance regressions before the code is deployed onto real physical hardware, ensuring high reliability across diverse operating conditions.
