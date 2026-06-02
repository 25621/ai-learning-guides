# Masked-Token Model

## Key Insight

Generating image [tokens](/shared/glossary/#token-visualaudio) strictly one at a time is slow. A [MaskGIT](/shared/glossary/#maskgit)-style masked-token model speeds this up by predicting *many* tokens at once: it starts from a grid where most tokens are hidden, fills in the ones it is most confident about, then repeats — refining the whole image in a handful of passes instead of hundreds. The analogy is solving a crossword by first writing in the answers you are sure of, which then make the remaining blanks easier to guess. This project implements such a parallel decoder over the same [VQ-GAN](/shared/glossary/#vq-gan) tokens and compares its sampling speed and quality against the row-by-row [transformer](/shared/glossary/#transformer) from the previous project.
