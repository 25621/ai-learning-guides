# Hallucination Eval

## Key Insight

Ask a [VLM](/shared/glossary/#vlm) "is there a dog in this image?" and it will often answer "yes" even when there is no dog — a [hallucination](/shared/glossary/#hallucination), where the model produces a fluent, confident claim the pixels do not support. This project builds a tiny [benchmark](/shared/glossary/#benchmark) of *paired* trick questions — half about an object that is present, half about one that is absent — so a model cannot score well just by always guessing "yes"; counting how often it falsely confirms absent objects versus correctly spots present ones turns a vague worry into concrete [precision and recall](/shared/glossary/#precision-and-recall) numbers you can compare across several [open](/shared/glossary/#open-model) VLMs. The design lesson is that a hallucination test must balance true and false cases, because a one-sided test secretly rewards a model that simply agrees with every question.
