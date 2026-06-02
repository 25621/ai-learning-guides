# FSQ Tokenizer

## Key Insight

[FSQ](/shared/glossary/#fsq) (Finite Scalar Quantization) is a surprisingly simple way to make image [tokens](/shared/glossary/#token-visualaudio) without learning a [codebook](/shared/glossary/#codebook) at all. Instead of looking up the nearest entry in a trained table, it just rounds each number in the latent to the nearest value on a fixed grid — like snapping every measurement to the nearest tick on a ruler. Because there is no codebook to train, FSQ sidesteps [codebook collapse](/shared/glossary/#codebook-collapse) entirely and is far easier to get working. This project implements FSQ and compares it to a learned [VQ-VAE](/shared/glossary/#vq-vae) on reconstruction [FID](/shared/glossary/#fid): the learned VQ-VAE is usually a little better, but in many settings the much simpler FSQ matches it closely — that is what "holds its own" means here.
