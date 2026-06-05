# Compare Projectors

## Key Insight

The [projector](/shared/glossary/#projector) is the only trained bridge in a [LLaVA](/shared/glossary/#llava)-style [VLM](/shared/glossary/#vlm), and this project races three choices for it on one downstream task: a single linear layer, a two-layer [MLP](/shared/glossary/#mlp), and a [Q-Former](/shared/glossary/#q-former). The real axis of comparison is detail-preservation versus token budget and speed: a linear or MLP projector keeps one token per image patch (maximum detail, but many tokens for the LLM to chew through), while a Q-Former distills the whole image into a small fixed set of learned query tokens (far fewer tokens and a faster LLM, at the cost of a tighter information bottleneck). Reporting quality *and* speed side by side makes the lesson land: there is no universal winner — the right projector depends on whether your task needs every patch or can survive a compressed summary.
