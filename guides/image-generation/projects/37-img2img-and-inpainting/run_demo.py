"""Demonstrate the hand-built img2img and inpainting loops.

Run:
    python run_demo.py            # ~5 min on a multicore CPU
"""

import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "36-run-sd-inference"))
from contact import contact_sheet  # noqa: E402

from diy_pipeline import SIZE, RawSD, timed  # noqa: E402


def main():
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    sd = RawSD()

    # A base image to work on, from the vanilla loop.
    base, s = timed(sd.txt2img, "a photo of a red barn in a green field, blue sky",
                    seed=4)
    base.save(out_dir / "base.png")
    print(f"base image: {s:.0f}s", flush=True)

    # --- img2img: strength sweep -----------------------------------------
    prompt = "a van gogh style oil painting of a barn in a field, swirling sky"
    images, labels = [base], ["input (photo)"]
    for strength in (0.3, 0.5, 0.7, 0.9):
        img, s = timed(sd.img2img, base, prompt, strength=strength, seed=11)
        images.append(img)
        labels.append(f"strength {strength}")
        print(f"img2img strength {strength}: {s:.0f}s", flush=True)
    contact_sheet(images, labels).save(out_dir / "img2img_strength_sweep.png")

    # --- inpainting: replace a sky region --------------------------------
    lat = SIZE // 8
    mask = torch.zeros(1, 1, lat, lat)
    mask[:, :, : lat // 2, lat // 4 : 3 * lat // 4] = 1.0  # top-middle box

    # Visualize the mask on the base image.
    vis = base.copy()
    px = vis.load()
    for y in range(SIZE // 2):
        for x in range(SIZE // 4, 3 * SIZE // 4):
            r, g, b = px[x, y][:3]
            px[x, y] = (min(255, r + 90), g // 2, b // 2)

    # Object insertion needs a strong push: the surrounding context says
    # "empty sky", so give the balloon more steps and a higher guidance.
    result, s = timed(
        sd.inpaint, base, mask,
        "a giant colorful striped hot air balloon floating in the sky "
        "above a red barn in a green field",
        steps=20, guidance=9.0, seed=12,
    )
    print(f"inpaint: {s:.0f}s", flush=True)
    contact_sheet([base, vis, result], ["original", "mask (red = regenerate)",
                                        "inpainted"]).save(out_dir / "inpainting.png")
    print(f"wrote {out_dir}/img2img_strength_sweep.png and inpainting.png")


if __name__ == "__main__":
    main()
