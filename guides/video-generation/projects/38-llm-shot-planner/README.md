# LLM Shot Planner

## Key Insight

A minute-long story is too much for any single [text-to-video (T2V)](/shared/glossary/#t2v) generation, so this project borrows the film world's solution: a [shot list](/shared/glossary/#shot-list). A small [large language model (LLM)](/shared/glossary/#llm) acts as a "director," expanding a one-line prompt like *"a knight rescues a princess"* into a structured JSON plan — a sequence of shots, each with its own description — that you then generate one at a time and stitch together. This is the planning half of [hierarchical generation](/shared/glossary/#hierarchical-generation): the LLM decides *what happens in what order* before any pixels are made, which is the only way to get a video whose events follow a sensible narrative rather than wandering. You evaluate the result on coherence — do the shots actually tell the intended story, and does the knight stay the same knight across them?
