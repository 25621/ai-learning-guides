"""Linear probes for factual truth, layer by layer.

We freeze a small pretrained transformer (GPT-2, 12 layers) and, for a set of
true/false factual statements, read out the hidden activation at the *last*
token of each statement at every layer. Then we train one tiny linear
classifier (logistic regression, from scratch in numpy) per layer to predict
"is this statement true?" from that activation alone.

If a probe can recover truth from a layer's activations, the model *represents*
the fact internally at that depth — even though nobody trained it to. Plotting
probe accuracy vs. layer maps out where in the network the truth signal lives.

    python probe.py          # ~2 min on CPU (forwards + 13 tiny logregs)
    python probe.py --plot   # redraw from outputs/probe_acc.csv

CPU-only. The probe is trained; the language model is never updated.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-train-a-bpe-from-scratch"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------------- #
# A labelled set of factual statements. Each is unambiguously true or false, so
# the *label* is ground truth and the probe's only job is to read it back out of
# the activations. We use templates with known answers so the set is clean.
# --------------------------------------------------------------------------- #
CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Canada": "Ottawa",
    "Italy": "Rome", "Spain": "Madrid", "Russia": "Moscow", "China": "Beijing",
    "Germany": "Berlin", "Greece": "Athens", "Norway": "Oslo", "Cuba": "Havana",
    "Peru": "Lima", "Kenya": "Nairobi", "Iraq": "Baghdad", "Chile": "Santiago",
    "Poland": "Warsaw", "Sweden": "Stockholm", "Austria": "Vienna", "Ireland": "Dublin",
    "Portugal": "Lisbon", "Turkey": "Ankara", "India": "Delhi", "Brazil": "Brasilia",
    "Mexico": "Mexico City", "Thailand": "Bangkok", "Iran": "Tehran",
    "Argentina": "Buenos Aires", "Hungary": "Budapest", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Belgium": "Brussels", "Switzerland": "Bern",
    "Netherlands": "Amsterdam", "Ukraine": "Kyiv", "Vietnam": "Hanoi",
    "Indonesia": "Jakarta", "Nigeria": "Abuja", "Colombia": "Bogota",
    "Morocco": "Rabat",
}
# Country -> continent (world knowledge the model represents cleanly).
CONTINENTS = {
    "France": "Europe", "Japan": "Asia", "Egypt": "Africa", "Canada": "North America",
    "Brazil": "South America", "Australia": "Oceania", "China": "Asia",
    "Nigeria": "Africa", "Germany": "Europe", "Peru": "South America",
    "India": "Asia", "Kenya": "Africa", "Mexico": "North America", "Spain": "Europe",
    "Chile": "South America", "Thailand": "Asia", "Morocco": "Africa",
    "Sweden": "Europe", "Argentina": "South America", "Vietnam": "Asia",
}
ELEMENTS = {
    "Gold": "metal", "Oxygen": "gas", "Iron": "metal", "Helium": "gas",
    "Copper": "metal", "Nitrogen": "gas", "Silver": "metal", "Hydrogen": "gas",
    "Lead": "metal", "Neon": "gas",
}
ANIMALS = {
    "salmon": "fish", "eagle": "bird", "shark": "fish", "sparrow": "bird",
    "tuna": "fish", "penguin": "bird", "trout": "fish", "owl": "bird",
    "cobra": "reptile", "lizard": "reptile", "frog": "amphibian", "toad": "amphibian",
}


def build_statements(seed=0):
    rng = np.random.default_rng(seed)
    rows = []  # (text, label)

    # Capitals: true = right city, false = a different country's city.
    cities = list(CAPITALS.values())
    for country, city in CAPITALS.items():
        rows.append((f"The capital of {country} is {city}.", 1))
        wrong = rng.choice([c for c in cities if c != city])
        rows.append((f"The capital of {country} is {wrong}.", 0))

    # Continents.
    conts = sorted(set(CONTINENTS.values()))
    for country, cont in CONTINENTS.items():
        rows.append((f"{country} is located in {cont}.", 1))
        wrong = rng.choice([c for c in conts if c != cont])
        rows.append((f"{country} is located in {wrong}.", 0))

    # Element categories.
    cats = list(set(ELEMENTS.values()))
    for name, cat in ELEMENTS.items():
        rows.append((f"{name} is a {cat}.", 1))
        wrong = rng.choice([c for c in cats if c != cat])
        rows.append((f"{name} is a {wrong}.", 0))

    # Animal categories.
    acats = list(set(ANIMALS.values()))
    for name, cat in ANIMALS.items():
        rows.append((f"The {name} is a type of {cat}.", 1))
        wrong = rng.choice([c for c in acats if c != cat])
        rows.append((f"The {name} is a type of {wrong}.", 0))

    # Arithmetic (a clean, model-independent notion of true/false).
    for _ in range(60):
        a, b = int(rng.integers(2, 40)), int(rng.integers(2, 40))
        rows.append((f"{a} plus {b} equals {a + b}.", 1))
        off = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        rows.append((f"{a} plus {b} equals {a + b + off}.", 0))

    # Comparisons.
    for _ in range(80):
        a, b = sorted(rng.integers(1, 100, size=2).tolist())
        if a == b:
            b += 1
        rows.append((f"{b} is greater than {a}.", 1))
        rows.append((f"{a} is greater than {b}.", 0))

    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- #
# Activation extraction: one forward pass per statement, keep the last-token
# hidden state at each of the 13 layers (embeddings + 12 blocks).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_activations(rows):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, output_hidden_states=True)
    model.eval()

    texts = [t for t, _ in rows]
    labels = np.array([y for _, y in rows], dtype=np.float64)
    n_layers = model.config.num_hidden_layers + 1
    feats = [[] for _ in range(n_layers)]

    bs = 32
    for b in range(0, len(texts), bs):
        chunk = texts[b:b + bs]
        enc = tok(chunk, return_tensors="pt", padding=True)
        out = model(**enc)
        # index of the last real token in each row (right-padded by default)
        last = enc.attention_mask.sum(1) - 1
        idx = torch.arange(len(chunk))
        for L, hs in enumerate(out.hidden_states):     # (B, T, d)
            feats[L].append(hs[idx, last].to(torch.float64).numpy())

    X = [np.concatenate(f, 0) for f in feats]           # list of (N, d)
    return X, labels


# --------------------------------------------------------------------------- #
# Logistic regression from scratch (no sklearn here). Standardize features,
# full-batch gradient descent with L2 regularization.
# --------------------------------------------------------------------------- #
def train_logreg(Xtr, ytr, Xte, yte, epochs=300, lr=0.5, l2=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - ytr
        gw = Xtr.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    acc_tr = ((Xtr @ w + b > 0) == (ytr > 0.5)).mean()
    acc_te = ((Xte @ w + b > 0) == (yte > 0.5)).mean()
    return acc_tr, acc_te


def run():
    rows = build_statements()
    print(f"statements: {len(rows)} "
          f"({int(sum(y for _, y in rows))} true / "
          f"{len(rows) - int(sum(y for _, y in rows))} false)", flush=True)
    X, y = collect_activations(rows)

    # Fixed train/test split, shared across layers for a fair comparison.
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = perm[:cut], perm[cut:]

    # Control: shuffle the labels. A probe should now be unable to do better
    # than chance at every layer — proof that the real curve reflects a signal
    # in the activations, not an artifact of the probe or the split.
    y_shuf = y.copy()
    rng.shuffle(y_shuf)

    lines = ["layer,train_acc,test_acc,shuffled_test_acc"]
    accs, shuf = [], []
    for L in range(len(X)):
        atr, ate = train_logreg(X[L][tr], y[tr], X[L][te], y[te])
        _, ash = train_logreg(X[L][tr], y_shuf[tr], X[L][te], y_shuf[te])
        accs.append(ate); shuf.append(ash)
        lines.append(f"{L},{atr:.4f},{ate:.4f},{ash:.4f}")
        print(f"layer {L:2d}: train {atr:.3f}  test {ate:.3f}  "
              f"shuffled {ash:.3f}", flush=True)
    (OUT / "probe_acc.csv").write_text("\n".join(lines) + "\n")

    best = int(np.argmax(accs))
    print(f"\nbest layer {best}: test acc {accs[best]:.3f} "
          f"(shuffled control {shuf[best]:.3f})", flush=True)
    plot()


def plot():
    import csv
    layers, tr, te, sh = [], [], [], []
    with open(OUT / "probe_acc.csv") as f:
        for r in csv.DictReader(f):
            layers.append(int(r["layer"]))
            tr.append(float(r["train_acc"]))
            te.append(float(r["test_acc"]))
            sh.append(float(r["shuffled_test_acc"]))

    fig, ax = ps.new_axes()
    ax.axhline(0.5, color=ps.BASELINE, lw=1.2, ls="--")
    ax.text(0.2, 0.515, "chance", color=ps.INK_MUTED, fontsize=8)
    ax.plot(layers, tr, "-o", color=ps.SERIES[0], lw=2, ms=4, label="train")
    ax.plot(layers, te, "-o", color=ps.SERIES[2], lw=2, ms=4, label="held-out")
    ax.plot(layers, sh, "-o", color=ps.INK_MUTED, lw=1.4, ms=3,
            label="shuffled-label control")
    best = int(np.argmax(te))
    ax.scatter([best], [te[best]], s=140, facecolor="none",
               edgecolor=ps.INK, lw=1.6, zorder=5)
    ax.set_ylim(0.42, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ps.finish(fig, ax, "Where Qwen2.5-0.5B encodes 'is this statement true?'",
              "layer (0 = embeddings, 24 = final block)", "probe accuracy",
              OUT / "probe_acc.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
