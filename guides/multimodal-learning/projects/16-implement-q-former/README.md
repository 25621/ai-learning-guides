# Implement Q-Former

## Key Insight

Building a small [Q-Former](/shared/glossary/#q-former) by hand demystifies how a sprawling image gets squeezed into a *fixed* handful of tokens a language model can read. You create 16 learned *query* vectors and let them [cross-attend](/shared/glossary/#cross-attention) to a [frozen](/shared/glossary/#frozen) image encoder's patch features, so each query walks away with one compact summary of what it cares about — 16 notes instead of a whole gallery tour. Training it on [COCO](/shared/glossary/#coco) captions (predict the caption from those 16 image tokens) teaches the queries to capture caption-relevant content, and the choice of 16 over, say, 256 is the whole point: the image becomes a *constant*, small number of tokens regardless of resolution, which is what keeps feeding images to an [LLM](/shared/glossary/#llm) affordable.
