# Hard-Negative Mining

## Key Insight

In [contrastive](/shared/glossary/#infonce) training most of the negative examples in a [batch](/shared/glossary/#batch) are *easy* — a photo of a dog against the caption "a downhill ski race" is so obviously mismatched that the model already scores it low and learns nothing from it. [Hard negatives](/shared/glossary/#hard-negatives) are the mismatches the model currently gets *almost* right — a dog photo against "a wolf in the snow" — and feeding it more of those is what actually sharpens the boundary it draws in the shared space. Mining them (for example, searching the dataset for the wrong caption with the highest [cosine similarity](/shared/glossary/#cosine-similarity) to each image) and comparing against plain random negatives lets you measure the payoff directly as a jump in [cross-modal retrieval](/shared/glossary/#cross-modal-retrieval) accuracy.
