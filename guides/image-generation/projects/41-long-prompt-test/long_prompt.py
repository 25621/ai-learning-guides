"""What actually happens to a long prompt in SD 1.x: CLIP's tokenizer cuts
it at 77 tokens, silently.

Two experiments:
1. Mechanical: tokenize a ~110-token prompt whose star subject is mentioned
   LAST, and decode back exactly what the text encoder sees. The subject is
   simply gone.
2. Behavioral: generate with the subject-last prompt vs the same content
   reordered subject-first, same seed. The model can only follow what
   survived the tokenizer.

Run:
    python long_prompt.py            # ~2 min on a multicore CPU
"""

import sys
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-run-sd-inference"))
from contact import contact_sheet  # noqa: E402

MODEL_ID = "segmind/tiny-sd"
SIZE = 384

FILLER = (
    "a highly detailed photograph taken at golden hour with soft warm light, "
    "shot on a full frame camera with a 50mm lens at f/1.8, shallow depth of "
    "field, rich color grading, cinematic composition, rule of thirds, gentle "
    "film grain, atmospheric haze in the background, professional photography, "
    "award winning, ultra sharp focus on the subject, beautiful bokeh"
)
SUBJECT = "a bright purple elephant standing in a shallow river"

PROMPT_SUBJECT_LAST = f"{FILLER}, the subject is {SUBJECT}"
PROMPT_SUBJECT_FIRST = f"{SUBJECT}, {FILLER}"


def main():
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    tok = pipe.tokenizer

    # --- 1) the mechanical truncation, made visible -----------------------
    report = []
    for name, prompt in (("subject LAST", PROMPT_SUBJECT_LAST),
                         ("subject FIRST", PROMPT_SUBJECT_FIRST)):
        full_ids = tok(prompt).input_ids
        kept_ids = tok(prompt, truncation=True,
                       max_length=tok.model_max_length).input_ids
        kept_text = tok.decode(kept_ids, skip_special_tokens=True)
        dropped = tok.decode(full_ids[len(kept_ids):], skip_special_tokens=True)
        report += [
            f"=== {name} ===",
            f"total tokens: {len(full_ids)} | kept: {len(kept_ids)} "
            f"(limit {tok.model_max_length})",
            f"WHAT THE MODEL SEES: {kept_text}",
            f"SILENTLY DROPPED:    {dropped or '(nothing)'}",
            "",
        ]
    (out_dir / "truncation_report.txt").write_text("\n".join(report))
    print("\n".join(report), flush=True)

    # --- 2) the behavioral consequence ------------------------------------
    images, labels = [], []
    for label, prompt in (("subject past token 77 (lost)", PROMPT_SUBJECT_LAST),
                          ("subject first (kept)", PROMPT_SUBJECT_FIRST)):
        t0 = time.time()
        img = pipe(prompt, height=SIZE, width=SIZE, num_inference_steps=12,
                   guidance_scale=7.5,
                   generator=torch.Generator().manual_seed(3)).images[0]
        images.append(img)
        labels.append(label)
        print(f"{label}: {time.time() - t0:.0f}s", flush=True)
    contact_sheet(images, labels).save(out_dir / "subject_position_ab.png")
    print(f"wrote {out_dir}/truncation_report.txt and subject_position_ab.png")


if __name__ == "__main__":
    main()
