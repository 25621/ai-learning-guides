# Tiny PixelCNN

## Key Insight

[PixelCNN](/shared/glossary/#pixelcnn) is an [autoregressive model](/shared/glossary/#autoregressive-model): it builds an image one pixel at a time, predicting each new pixel from the pixels it has already drawn — like writing a sentence word by word. This project trains a small version on [MNIST](/shared/glossary/#mnist) digits and samples from it row by row, so you feel both its strengths and its weakness firsthand. The quality is surprisingly good and the math is clean, but sampling is painfully slow because every pixel has to wait for the one before it. That slowness is exactly why later approaches like diffusion, which generate all pixels in parallel, eventually took over.
