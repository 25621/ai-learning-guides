"""Outpainting = inpainting where the 'hole' is everything OUTSIDE the image.

Recipe: paste the original onto a wider canvas, pre-fill the new border by
mirroring the image edges (a much better starting point for the known-region
latents than gray), mask the border as "to be generated", and run project
37's blended-latent inpainting loop on the whole canvas.

Run (uses project 37's DIY pipeline):
    python outpaint.py            # ~3 min on a multicore CPU
"""

import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-run-sd-inference"))
sys.path.insert(0, str(HERE.parent / "37-img2img-and-inpainting"))
from contact import stack_sheets  # noqa: E402
from diy_pipeline import SIZE, RawSD, timed  # noqa: E402

EXTEND = 128  # pixels added on each side; must be a multiple of 8


def build_canvas(base: Image.Image):
    """Wider canvas with mirrored borders as the pre-fill."""
    w, h = base.size
    canvas = Image.new("RGB", (w + 2 * EXTEND, h))
    canvas.paste(base, (EXTEND, 0))
    left = ImageOps.mirror(base.crop((0, 0, EXTEND, h)))
    right = ImageOps.mirror(base.crop((w - EXTEND, 0, w, h)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w + EXTEND, 0))
    return canvas


def main():
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    sd = RawSD()

    base, s = timed(
        sd.txt2img,
        "a photo of a lighthouse on a rocky cliff above the ocean, dramatic sky",
        seed=21,
    )
    print(f"base: {s:.0f}s", flush=True)

    canvas = build_canvas(base)

    # Latent-space mask: 1 where content is generated (the borders).
    lat_w, lat_h = canvas.size[0] // 8, canvas.size[1] // 8
    ext = EXTEND // 8
    mask = torch.zeros(1, 1, lat_h, lat_w)
    mask[..., :ext] = 1.0
    mask[..., -ext:] = 1.0

    result, s = timed(
        sd.inpaint, canvas, mask,
        "a wide panoramic photo of a lighthouse on a rocky cliff above the "
        "ocean, coastline stretching into the distance, dramatic sky",
        steps=15, seed=6,
    )
    print(f"outpaint: {s:.0f}s", flush=True)

    # Center the original over the result for the before/after figure.
    padded = Image.new("RGB", result.size, (252, 252, 251))
    padded.paste(base, (EXTEND, 0))
    stack_sheets([padded, result]).save(out_dir / "outpainting.png")
    print(f"wrote {out_dir}/outpainting.png (top: original, bottom: outpainted)")


if __name__ == "__main__":
    main()
