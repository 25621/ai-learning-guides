# Grounding Head

## Key Insight

[Grounding](/shared/glossary/#grounding) means making a [VLM](/shared/glossary/#vlm) point at *where* something is, not just say *that* it is there; this project adds it the simplest possible way — extend the model's [vocabulary](/shared/glossary/#vocabulary) with [special tokens](/shared/glossary/#special-tokens) like `<box>` plus a small set of tokens that stand for quantized coordinates, so a bounding box becomes just a few extra tokens the model emits inside its normal text stream. The elegance is that no new architecture or loss is needed: predicting "the cat is at `<box>` 0.1 0.2 0.4 0.6" is the very same [next-token prediction](/shared/glossary/#next-token-prediction) the [LLM](/shared/glossary/#llm) already performs, so it learns spatial output for free with the objective it was built around. Coordinates are quantized into a fixed grid of bins (rather than predicting raw floats) precisely so each one collapses to a single discrete token the existing vocabulary can hold.

## Data Preparation Pipeline

To teach a model this spatial mapping, the training data must be formatted to match standard [instruction tuning](/shared/glossary/#instruction-tuning) structures:

1. **Normalization**: Convert absolute pixel coordinates into relative coordinates between `0.0` and `1.0`.
2. **Quantization**: Map these continuous floats into discrete bins (e.g., 1000 bins) so they align with the newly added coordinate tokens in the [vocabulary](/shared/glossary/#vocabulary).
   * *Example:* `[x1: 0.15, y1: 0.25, x2: 0.45, y2: 0.65]` becomes the discrete sequence `<box> 0.15 0.25 0.45 0.65`.
3. **Formatting**: Wrap the quantized coordinates in a standard conversational JSON structure so the model learns to emit them naturally in response to user prompts:
   ```json
   {
     "image": "cat_photo_001.jpg",
     "conversations": [
       { "role": "user", "content": "Where is the cat in this image?" },
       { "role": "assistant", "content": "The cat is located at <box> 0.15 0.25 0.45 0.65." }
     ]
   }
   ```

## How Frontier Models Handle Grounding

Modern frontier models streamline this process by moving away from glued-together components (frozen encoder + projector) toward [Native Multimodal](/shared/glossary/#native-multimodal) architectures:

* **Training (Native Multimodal):** Images and text are processed into a shared embedding space from the start. The model is instruction-tuned to predict coordinate tokens unconditionally alongside normal text using standard [next-token prediction](/shared/glossary/#next-token-prediction), learning the pattern of relating visual features to text tokens without architectural hacks.
* **Inference Pipeline:**
  * **Prefill:** The model ingests the user's text and the entire sequence of image tokens simultaneously, computing the visual context and storing it in the [KV cache](/shared/glossary/#kv-cache).
  * **Decode:** The model autoregressively generates the response ("The cat is located at..."). When it is time to output spatial tokens, it cross-attends to the visual KV cache. [Greedy decoding](/shared/glossary/#greedy-decoding) is typically used for the coordinate tokens to prevent positional hallucinations.
