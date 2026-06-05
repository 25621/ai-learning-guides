# Speech LLM

## Key Insight

A Speech LLM reuses the exact recipe behind [LLaVA](/shared/glossary/#projector) — freeze a pretrained encoder, freeze a pretrained [LLM](/shared/glossary/#llm), and train only a small [projector](/shared/glossary/#projector) between them — but swaps the vision encoder for an audio encoder, so the language model can "hear." The projector's whole job is to map audio features into the LLM's word-vector space; once aligned, the LLM can caption sounds or answer questions about them using its existing language ability. Training on [AudioSet](/shared/glossary/#audioset) captions is what teaches that bridge, and because only the projector updates, the hard, five-star part of this project is curating good (audio, caption) data rather than the modeling itself.
