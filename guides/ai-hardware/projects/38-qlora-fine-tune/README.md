# QLoRA Fine-Tune

## Key Insight

Fine-tuning large language models on consumer-grade hardware is made possible by combining parameter-efficient methods with model compression. By loading a 7B parameter model in 4-bit [NF4](/shared/glossary/#nf4) precision and training [LoRA](/shared/glossary/#lora) adapters on top of it, [QLoRA](/shared/glossary/#qlora) drastically reduces the peak GPU memory footprint. Measuring memory consumption during training highlights how [gradient checkpointing](/shared/glossary/#gradient-checkpointing) and [page optimizers](/shared/glossary/#paged-optimizers) further optimize resource allocation without sacrificing model accuracy.
