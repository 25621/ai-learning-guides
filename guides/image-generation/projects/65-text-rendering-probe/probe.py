"""Text-rendering probe: why generator accuracy collapses as the requested
string gets longer.

Rendering legible text was image generation's most stubborn failure mode for
years. One big reason is purely statistical and unavoidable: a model that draws
a *single* character correctly with probability p still needs k independent
successes to get a k-character string exactly right, so whole-string accuracy
falls like p^k. This probe isolates that compounding law on digit strings.

We render each character with a single-digit class-conditional DDPM (per-character
accuracy ~80%), lay k of them side by side into a "sign", OCR the sign back with
the MNIST classifier, and report exact-string accuracy vs. length. Even at 80%
per character, four-digit strings are right less than half the time — the
multiplicative wall, before you even add the joint-rendering and spelling errors
that afflict real models.

    python probe.py --data-dir data      # ~7 min on CPU

Reuses the phase-5/9 conditional DDPM stack and project 58's classifier.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
sys.path.insert(0, str(HERE.parent / "28-class-conditional-ddpm"))
sys.path.insert(0, str(HERE.parent / "58-caption-ablation"))

from diffusion import GaussianDiffusion  # noqa: E402
from train import EMA, infinite  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402
from mnist_classifier import load_or_train, read_digits  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 200
MAXLEN = 4


def mnist_labeled(data_dir, bs=64):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


def train_char_model(data_dir, out, steps=1500, seed=0):
    if out.exists():
        m = ConditionalUNet(num_classes=10, in_ch=1, image_size=28, attn_resolutions=(7,))
        m.load_state_dict(torch.load(out)["ema"]); return m.eval()
    torch.manual_seed(seed)
    model = ConditionalUNet(num_classes=10, in_ch=1, image_size=28, attn_resolutions=(7,))
    ema = EMA(model, 0.995)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batches = infinite(mnist_labeled(data_dir))
    print(f"char model params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    for step in range(1, steps + 1):
        x0, y = next(batches)
        loss = diffusion.loss(model, x0, model_kwargs={"y": y})
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
        if step % 300 == 0:
            print(f"  step {step}/{steps} | loss {loss.item():.4f} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema.shadow.state_dict()}, out)
    return ema.shadow.eval()


@torch.no_grad()
def render_chars(model, diffusion, labels, seed=0, chunk=256):
    """Sample one 28x28 digit image for each requested class label, in chunks
    so a large probe set doesn't blow up memory or a single forward pass."""
    torch.manual_seed(seed)
    y_all = torch.tensor(labels, dtype=torch.long)
    outs = []
    for s in range(0, len(y_all), chunk):
        y = y_all[s:s + chunk]
        x = torch.randn(len(y), 1, 28, 28)
        imgs, _ = diffusion.p_sample_loop(model, x.shape, model_kwargs={"y": y}, x_init=x)
        outs.append(imgs)
    return torch.cat(outs, dim=0)


def probe(model, diffusion, clf, n_per=45, seed=1):
    rng = np.random.default_rng(seed)
    # Build all requested strings, flatten every character into one big batch,
    # render once, then reassemble per string.
    strings = {k: [[int(rng.integers(0, 10)) for _ in range(k)] for _ in range(n_per)]
               for k in range(1, MAXLEN + 1)}
    flat_labels, index = [], {}
    for k in range(1, MAXLEN + 1):
        index[k] = []
        for s in strings[k]:
            span = list(range(len(flat_labels), len(flat_labels) + k))
            flat_labels.extend(s)
            index[k].append(span)
    imgs = render_chars(model, diffusion, flat_labels, seed)
    preds, _ = read_digits(clf, imgs)
    preds = preds.numpy()

    results, examples = {}, {}
    for k in range(1, MAXLEN + 1):
        exact = char_ok = chars = 0
        ex = []
        for s, span in zip(strings[k], index[k]):
            got = [int(preds[j]) for j in span]
            char_ok += sum(int(g == c) for g, c in zip(got, s))
            chars += k
            is_exact = got == s
            exact += int(is_exact)
            if len(ex) < 6:
                ex.append((torch.cat([imgs[j, 0] for j in span], dim=1), s, got, is_exact))
        results[k] = dict(exact=exact / n_per, per_char=char_ok / chars, n=n_per)
        examples[k] = ex
    return results, examples


def string_of(lst):
    return "".join(str(x) for x in lst)


def plot_curve(results, path):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(HERE.parent / "24-ddpm-on-mnist"))
    import plot_style as ps
    ks = sorted(results)
    exact = [results[k]["exact"] for k in ks]
    per_char = [results[k]["per_char"] for k in ks]
    p1 = per_char[0]
    pred = [p1 ** k for k in ks]
    fig, ax = ps.new_axes(6.8, 4.4)
    ax.plot(ks, per_char, "-o", color=ps.SERIES[1], label="per-character accuracy")
    ax.plot(ks, exact, "-o", color=ps.SERIES[2], label="exact-string accuracy")
    ax.plot(ks, pred, "--", color=ps.INK_MUTED,
            label=f"$p^k$ compounding ($p$={p1:.2f})")
    ax.set_xticks(ks); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Exact-string accuracy collapses with length even at ~80% per character",
              "requested string length (digits)", "accuracy", path)


def plot_examples(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(examples)
    fig, axes = plt.subplots(len(ks), 6, figsize=(11, 1.05 * len(ks) + 1), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for r, k in enumerate(ks):
        for c in range(6):
            ax = axes[r][c]; ax.axis("off")
            if c >= len(examples[k]):
                continue
            img_, want, got, ok = examples[k][c]
            a = ((img_.clamp(-1, 1) + 1) * 127.5).byte().numpy()
            ax.imshow(a, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{string_of(want)}→{string_of(got)}", fontsize=8,
                         color="#1baf7a" if ok else "#e34948")
    fig.suptitle("Requested sign → what OCR read back (green = exact match)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-per", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    diffusion = GaussianDiffusion(T=T, schedule="linear", device="cpu")

    clf = load_or_train(args.data_dir,
                        HERE.parent / "58-caption-ablation/checkpoints/classifier.pt")
    print("training the single-character (digit) renderer ...")
    model = train_char_model(args.data_dir, CKPT / "char.pt", args.steps)

    results, examples = probe(model, diffusion, clf, args.n_per)

    plot_curve(results, OUT / "length_curve.png")
    plot_examples(examples, OUT / "examples.png")

    lines = ["length,exact_accuracy,per_char_accuracy,n"]
    for k in sorted(results):
        r = results[k]
        lines.append(f"{k},{r['exact']:.3f},{r['per_char']:.3f},{r['n']}")
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    print("\ntext-rendering accuracy by string length:")
    for k in sorted(results):
        r = results[k]
        print(f"  len {k}: exact {r['exact']:.0%} | per-char {r['per_char']:.0%}")
    print(f"\nwrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
