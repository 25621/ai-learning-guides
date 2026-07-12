"""A tiny sparse autoencoder (SAE) on GPT-2's residual stream.

The residual stream of a transformer is dense and *polysemantic*: a single
activation dimension lights up for many unrelated concepts, because the model
packs far more features than it has dimensions (superposition). An SAE tries to
undo that packing. It learns an over-complete dictionary — many more features
than the 768 dimensions it reads — under a sparsity penalty, so that any given
activation is reconstructed from just a handful of features. With luck those
features are *monosemantic*: each one fires for a single human-readable concept.

Pipeline:
  1. Run GPT-2 over ~1,500 Wikipedia paragraphs, collect the layer-6 residual
     stream at every token position (~50k activation vectors of width 768).
  2. Train an SAE (768 -> 6144 -> 768) with an L1 penalty on the codes.
  3. Measure reconstruction quality and sparsity (L0 = features per token).
  4. Interpret features by their top-activating tokens in context.

    python sae.py          # ~7 min on CPU (collect + train + interpret)
    python sae.py --plot   # redraw figures from outputs/*.csv

CPU-only. GPT-2 is frozen; only the SAE is trained.
"""

import csv
import html
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
sys.path.insert(0, str(HERE.parent / "43-minimal-rag"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
CKPT = HERE / "checkpoints"
CKPT.mkdir(exist_ok=True)

MODEL = "gpt2"
LAYER = 6                 # residual stream after block 6 (of 12)
D = 768                   # gpt2 residual width
EXPANSION = 4             # dictionary is 4x over-complete (3072 features)
N_FEATURES = D * EXPANSION
MAX_TOKENS = 40_000
L1 = 3e-3                 # sparsity penalty weight


# --------------------------------------------------------------------------- #
# 1. Collect residual-stream activations, keeping token ids + context so we can
#    later read a feature's meaning off the tokens that fire it.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect():
    # Reuse a previous collection if present (it is the slow, deterministic part).
    cache = CKPT / "acts.npz"
    if cache.exists() and (CKPT / "para_tokens.json").exists():
        acts = np.load(cache)["acts"]
        if acts.shape[0] >= MAX_TOKENS:
            acts = acts[:MAX_TOKENS]
            print(f"reusing cached {acts.shape[0]} activations", flush=True)
            return acts

    from transformers import AutoModel, AutoTokenizer
    from rag_lib import build_corpus

    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODEL, output_hidden_states=True)
    model.eval()

    paras, _ = build_corpus(n_paragraphs=1500, n_questions=1, seed=0)
    texts = [p["text"] for p in paras]

    acts = []          # (n, 768) float32 activation rows
    tok_ids = []       # token id at each row
    para_of = []       # which paragraph each row came from
    pos_of = []        # position within that paragraph
    para_tokens = []   # token id list per paragraph (for context windows)

    bs = 16
    for b in range(0, len(texts), bs):
        chunk = texts[b:b + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=96)
        out = model(**enc)
        hs = out.hidden_states[LAYER]                 # (B, T, D)
        mask = enc.attention_mask.bool()
        for j in range(len(chunk)):
            ids = enc.input_ids[j][mask[j]]
            h = hs[j][mask[j]]
            pi = b + j
            para_tokens.append(ids.tolist())
            for pos in range(len(ids)):
                acts.append(h[pos].numpy())
                tok_ids.append(int(ids[pos]))
                para_of.append(pi)
                pos_of.append(pos)
        if len(acts) >= MAX_TOKENS:
            break

    acts = np.asarray(acts[:MAX_TOKENS], dtype=np.float32)
    print(f"collected {acts.shape[0]} activations of width {acts.shape[1]}",
          flush=True)
    np.savez(CKPT / "acts.npz",
             acts=acts,
             tok_ids=np.array(tok_ids[:MAX_TOKENS]),
             para_of=np.array(para_of[:MAX_TOKENS]),
             pos_of=np.array(pos_of[:MAX_TOKENS]))
    # save context token lists + save the tokenizer name for decoding later
    import json
    (CKPT / "para_tokens.json").write_text(json.dumps(para_tokens))
    return acts


# --------------------------------------------------------------------------- #
# 2. The SAE. Decoder columns are kept unit-norm so the L1 penalty measures true
#    sparsity (a feature can't dodge the penalty by shrinking its code and
#    growing its decoder vector). A per-feature bias is subtracted before
#    encoding, the standard Anthropic recipe.
# --------------------------------------------------------------------------- #
class SAE(torch.nn.Module):
    def __init__(self, d=D, m=N_FEATURES):
        super().__init__()
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.W_enc = torch.nn.Parameter(torch.randn(d, m) * (1.0 / d ** 0.5))
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        # tie decoder init to encoder transpose, then keep it unit-norm
        self.W_dec = torch.nn.Parameter(self.W_enc.data.t().clone())
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, c):
        return c @ self.W_dec + self.b_dec

    def forward(self, x):
        c = self.encode(x)
        return self.decode(c), c


def train_sae(acts, steps=2500, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(12)
    X = torch.from_numpy(acts)
    sae = SAE()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        x = X[idx]
        recon, c = sae(x)
        mse = F.mse_loss(recon, x)
        l1 = c.abs().sum(1).mean()
        loss = mse + L1 * l1
        opt.zero_grad(); loss.backward(); opt.step()
        sae._normalize_decoder()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                l0 = (c > 0).float().sum(1).mean().item()
                var = ((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
                fvu = var.item()               # fraction of variance unexplained
            hist.append((step, mse.item(), l0, 1 - fvu))
            print(f"step {step:4d}  mse {mse.item():.3f}  L0 {l0:5.1f}  "
                  f"var-explained {1 - fvu:.3f}", flush=True)
            torch.save(sae.state_dict(), CKPT / "sae.pt")   # checkpoint often
    torch.save(sae.state_dict(), CKPT / "sae.pt")
    with open(OUT / "train.csv", "w") as f:
        f.write("step,mse,l0,var_explained\n")
        for s, m, l0, ve in hist:
            f.write(f"{s},{m:.4f},{l0:.2f},{ve:.4f}\n")
    return sae


# --------------------------------------------------------------------------- #
# 3. Interpret features: for a set of live features, find the tokens that fire
#    them hardest and print a little context window around each.
# --------------------------------------------------------------------------- #
def interpret(acts):
    import json
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    z = np.load(CKPT / "acts.npz")
    tok_ids, para_of, pos_of = z["tok_ids"], z["para_of"], z["pos_of"]
    para_tokens = json.loads((CKPT / "para_tokens.json").read_text())

    sae = SAE(); sae.load_state_dict(torch.load(CKPT / "sae.pt")); sae.eval()
    with torch.no_grad():
        codes = sae.encode(torch.from_numpy(acts)).numpy()      # (N, m)

    fire_rate = (codes > 0).mean(0)                              # per feature
    live = np.where(fire_rate > 1e-4)[0]
    dead = N_FEATURES - len(live)
    print(f"\nlive features: {len(live)} / {N_FEATURES}  (dead: {dead})",
          flush=True)

    def context(row, span=6):
        pi, pos = int(para_of[row]), int(pos_of[row])
        toks = para_tokens[pi]
        lo, hi = max(0, pos - span), min(len(toks), pos + span + 1)
        parts = []
        for k in range(lo, hi):
            piece = tok.decode([toks[k]])
            if k == pos:
                piece = f"[[{piece.strip()}]]"
            parts.append(piece)
        return "".join(parts).replace("\n", " ").strip()

    # Rank live features by how selective they are (high max activation, low
    # fire rate) — these are the crisp, interpretable ones, not the always-on
    # "bias" features.
    max_act = codes.max(0)
    selectivity = max_act[live] / (fire_rate[live] * len(codes) + 1)
    order = live[np.argsort(-selectivity)]

    report = []
    chosen = []
    seen_top_tokens = set()
    for f in order:
        rows = np.argsort(-codes[:, f])[:8]
        if codes[rows[0], f] <= 0:
            continue
        top_tok = tok.decode([int(tok_ids[rows[0]])]).strip().lower()
        # skip near-duplicate features (same dominant token) to show variety
        if top_tok in seen_top_tokens:
            continue
        seen_top_tokens.add(top_tok)
        block = [f"feature #{int(f)}  (fires on {fire_rate[f] * 100:.2f}% of "
                 f"tokens, peak {max_act[f]:.1f})"]
        for r in rows:
            block.append(f"    {codes[r, f]:5.1f}  ...{context(int(r))}...")
        report.append("\n".join(block))
        chosen.append(int(f))
        if len(chosen) >= 12:
            break

    (OUT / "features.txt").write_text("\n\n".join(report) + "\n")
    print("\n" + "\n\n".join(report[:6]), flush=True)

    # sparsity histogram data (log10 fire rate of live features)
    np.save(CKPT / "fire_rate.npy", fire_rate)
    with open(OUT / "summary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"live_features,{len(live)}\n")
        f.write(f"dead_features,{dead}\n")
        f.write(f"n_features,{N_FEATURES}\n")
    return fire_rate, live


def run():
    acts = collect()
    train_sae(acts)
    interpret(acts)
    plot()


def plot():
    # (a) training curve: L0 sparsity and variance explained
    steps, l0, ve = [], [], []
    with open(OUT / "train.csv") as f:
        for r in csv.DictReader(f):
            steps.append(int(r["step"])); l0.append(float(r["l0"]))
            ve.append(float(r["var_explained"]))
    fig, ax = ps.new_axes()
    ax.plot(steps, l0, "-o", color=ps.SERIES[0], lw=2, ms=3,
            label="L0 (features / token)")
    ax.set_ylabel("L0  (active features per token)", color=ps.SERIES[0])
    ax2 = ax.twinx()
    ax2.plot(steps, ve, "-o", color=ps.SERIES[1], lw=2, ms=3)
    ax2.set_ylabel("variance explained", color=ps.SERIES[1])
    ax2.set_ylim(0, 1); ax2.grid(False)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    ps.finish(fig, ax, "SAE learns a sparse code: few features, faithful reconstruction",
              "training step", "", OUT / "sae_train.png")

    # (b) feature fire-rate histogram (how sparse features are)
    fr = np.load(CKPT / "fire_rate.npy")
    live = fr[fr > 1e-4]
    fig, ax = ps.new_axes()
    ax.hist(np.log10(live), bins=40, color=ps.SERIES[0], alpha=0.85)
    ax.set_title("Features are sparse — a typical one fires on a few percent of tokens",
                 color=ps.INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(r"$\log_{10}$ fraction of tokens a feature fires on",
                  color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("number of features", color=ps.INK_SECONDARY, fontsize=10)
    import matplotlib.pyplot as plt
    fig.tight_layout(); fig.savefig(OUT / "fire_rate_hist.png",
                                    facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fire_rate_hist.png'}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
