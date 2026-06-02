# Class-Conditional DDPM

## Key Insight

By default a [DDPM](/shared/glossary/#ddpm) draws a random sample from everything it learned — you cannot ask it for a specific digit or object. [Class conditioning](/shared/glossary/#class-conditioning) fixes that by feeding the desired label into the network alongside the noisy image, so the same model can be steered to generate exactly the class you want. This project injects the label through the network's [normalization](/shared/glossary/#normalization) layers — a learned, per-class scale-and-shift applied to the [activations](/shared/glossary/#activations), an approach often called AdaGN (adaptive group normalization) — training on labeled [CIFAR-10](/shared/glossary/#cifar-10) and then verifying you can summon any of the ten classes on demand. It is the smallest possible step toward the conditioning that, once scaled up to text prompts, becomes modern text-to-image generation.
