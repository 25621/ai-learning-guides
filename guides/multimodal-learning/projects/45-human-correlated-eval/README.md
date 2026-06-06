# Human-Correlated Eval

## Key Insight

[LLM-as-judge](/shared/glossary/#llm-as-judge) grading is only trustworthy if it actually agrees with humans, so you have to *measure* that agreement rather than assume it. This project collects both human ratings and LLM-judge ratings on the same 100 outputs and computes an [inter-rater agreement](/shared/glossary/#inter-rater-agreement) score (a correlation or Cohen's kappa): a high score means the cheap automatic judge can stand in for expensive human review, a low one means it cannot. The subtlety the project exposes is that an LLM judge carries consistent biases — it tends to favor longer answers and ones written in its own style — so lining up with a single human rater is not enough; you gather *three* human and *three* model ratings per output and check whether the model agrees with people about as well as people agree with each other.
