# Hard-Negative Mining

## Key Insight

In [contrastive](/shared/glossary/#infonce) training most of the negative examples in a [batch](/shared/glossary/#batch) are *easy* — a photo of a dog against the caption "a downhill ski race" is so obviously wrong the model already scores it low and learns nothing from it. This project is about [hard negatives](/shared/glossary/#hard-negatives) instead — the near-miss mismatches the model gets *almost* right — and measuring what they actually buy you: you *mine* them (for example, searching the dataset for the wrong caption with the highest [cosine similarity](/shared/glossary/#cosine-similarity) to each image), train on them, and compare against plain random negatives. The payoff shows up directly as a jump in [cross-modal retrieval](/shared/glossary/#cross-modal-retrieval) accuracy — measurable proof that not all negatives teach the model equally.
