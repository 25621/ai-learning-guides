"""With/without negative-prompt comparison, scored with CLIP.

A negative prompt is not a filter — it replaces the empty string in CFG's
unconditional branch, so every step extrapolates AWAY from "ugly, blurry,
low quality" the same way it extrapolates toward the prompt.

The recorded run uses 4 prompts x 2 conditions at a small step count so it
finishes in minutes on CPU; the guide's full protocol (50 prompts, FID-CLIP,
aesthetic scoring) is the same loop with bigger constants — see the README.

Run:
    python negative_prompts.py            # ~5 min on a multicore CPU
"""

import csv
import sys
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-run-sd-inference"))
from contact import contact_sheet, stack_sheets  # noqa: E402

MODEL_ID = "segmind/tiny-sd"
SIZE = 384
NEGATIVE = "ugly, blurry, low quality, deformed, distorted, bad anatomy"
PROMPTS = [
    "a portrait photo of an elderly fisherman, detailed face",
    "a bowl of ramen on a wooden table, food photography",
    "a castle on a hill in morning fog",
    "a hummingbird hovering next to a red flower",
]


def main():
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    rows, sheets = [("prompt", "clip_no_negative", "clip_with_negative")], []
    results = []
    for i, prompt in enumerate(PROMPTS):
        pair = []
        for negative in (None, NEGATIVE):
            t0 = time.time()
            img = pipe(prompt, negative_prompt=negative, height=SIZE, width=SIZE,
                       num_inference_steps=12, guidance_scale=7.5,
                       generator=torch.Generator().manual_seed(100 + i)).images[0]
            pair.append(img)
            print(f"[{i}] negative={'yes' if negative else 'no '} "
                  f"{time.time() - t0:.0f}s", flush=True)
        sheets.append(contact_sheet(pair, ["no negative", "with negative"]))
        results.append((prompt, pair))
    stack_sheets(sheets).save(out_dir / "pairs.png")

    # CLIP score: cosine similarity between prompt and image embeddings.
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        for prompt, (img_plain, img_neg) in results:
            inputs = proc(text=[prompt], images=[img_plain, img_neg],
                          return_tensors="pt", padding=True)
            out = clip(**inputs)
            t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            v = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            sims = (v @ t.T).flatten().tolist()
            rows.append((prompt[:40], f"{sims[0]:.4f}", f"{sims[1]:.4f}"))
            print(f"{prompt[:40]:42s} plain {sims[0]:.4f} | negative {sims[1]:.4f}")
    with open(out_dir / "clip_scores.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {out_dir}/pairs.png and clip_scores.csv")


if __name__ == "__main__":
    main()
