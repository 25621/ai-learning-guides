# Eligibility Traces

## Key Insight

[Eligibility traces](/shared/glossary/#eligibility-traces) give every recently visited state a fading "credit tag," so that when a [TD error](/shared/glossary/#td-error) finally arrives it updates not just the current state but all the states that led up to it, each in proportion to how recently it was seen. The trace-decay parameter λ slides smoothly between the two extremes you already know: λ = 0 is plain one-step [TD(0)](/shared/glossary/#temporal-difference-learning), and λ = 1 recovers [Monte Carlo](/shared/glossary/#monte-carlo-method), with intermediate values usually learning fastest. *Replacing* traces cap each state's tag at 1 rather than letting repeat visits pile it higher, which stops a state visited in a loop from earning unrealistically large credit; sweeping λ ∈ {0, 0.5, 0.9, 1.0} lets you watch the [bias–variance](/shared/glossary/#bias-variance-tradeoff) dial turn in real time.
