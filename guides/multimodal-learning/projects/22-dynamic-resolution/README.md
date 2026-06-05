# Dynamic Resolution

## Key Insight

A plain [VLM](/shared/glossary/#vlm) squashes every image down to one fixed square (e.g. 336×336), which throws away the fine print in a document or a dense chart; [AnyRes](/shared/glossary/#anyres) fixes this by tiling the picture at its native aspect ratio into several sub-images, encoding each tile separately, and handing all of their [image tokens](/shared/glossary/#token-visualaudio) to the LLM. Because more tiles means more tokens means more detail preserved, this is exactly the change that lifts [OCR (Optical Character Recognition)](/shared/glossary/#ocr-optical-character-recognition)-heavy benchmarks — tasks where the answer hides in small text the squashed-down image literally cannot resolve. The trade-off to verify is cost: each extra tile adds [image tokens](/shared/glossary/#token-visualaudio) the LLM must process, so AnyRes buys accuracy on detail-dense images at the price of a longer, slower sequence.
