# Tiny Chameleon

## Key Insight

Once an image is just a row of [discrete tokens](/shared/glossary/#token-visualaudio), you can splice it into a sentence and train a single [transformer](/shared/glossary/#transformer) on the mixed stream with one ordinary [next-token-prediction](/shared/glossary/#next-token-prediction) loss — the [early-fusion](/shared/glossary/#fusion-earlymiddlelate) recipe behind [Chameleon](/shared/glossary/#chameleon) and other [native multimodal](/shared/glossary/#native-multimodal) models. Interleaving image tokens with [COCO](/shared/glossary/#coco) caption text in one shared [vocabulary](/shared/glossary/#vocabulary) means the model never sees a seam between "looking" and "reading"; it just predicts the next code, whether that code is a word-piece or a patch of pixels. The lesson is liberating: you do not need a separate vision tower or a special fusion module at all — tokenize everything and let one plain language-model objective do the work.
