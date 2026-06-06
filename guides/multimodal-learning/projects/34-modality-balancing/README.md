# Modality Balancing

## Key Insight

When one [transformer](/shared/glossary/#transformer) learns text, image, and audio together, the modality with the most [tokens](/shared/glossary/#token-visualaudio) quietly takes over: its share of the [next-token-prediction](/shared/glossary/#next-token-prediction) loss is the largest, so the gradient mostly improves that one modality while the others stall. [Modality balancing](/shared/glossary/#modality-balancing) is the fix — oversample the rare modality's data, or scale up its loss term, until each modality's loss falls at a comparable rate. Deliberately starving one modality and watching its loss flat-line teaches the single most common failure of [native multimodal](/shared/glossary/#native-multimodal) training, and exactly why "just throw all the data in together" is not enough.
