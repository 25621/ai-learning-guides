"""Train a byte-level BPE tokenizer from scratch — the real, efficient algorithm.

BPE starts from raw bytes and repeatedly merges the most frequent adjacent pair
into a new symbol; the ordered list of merges IS the tokenizer. The naive version
(rescan the whole token stream every merge) is far too slow for a 5000-merge
vocabulary, so we use the standard trick every real trainer uses: pre-tokenize
into words, count word frequencies, and run BPE over the *unique* words weighted
by frequency, updating pair counts incrementally.

We train a 5000-token vocabulary on ~1 MB of text, serialize it, reload it, and
verify byte-exact round-trips — then chart how compression improves with vocab
size and what the learned tokens look like.

    python bpe.py --corpus data/corpus.txt      # ~1 min on CPU
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# GPT-2-style pre-tokenization (stdlib re): a word with an optional single
# leading space, or a run of whitespace. Keeping the leading space attached is
# what makes " Paris" and "Paris" different tokens downstream.
PInitial = re.compile(r" ?[^\s]+|\s+")


def pretokenize(text):
    """text -> {tuple(byte_ids): frequency}."""
    freq = defaultdict(int)
    for chunk in PInitial.findall(text):
        freq[tuple(chunk.encode("utf-8"))] += 1
    return freq


def _pairs(ids):
    return zip(ids, ids[1:])


class BPE:
    def __init__(self, merges=None):
        # merges: dict {(a,b): new_id} in learned order (dicts keep insertion order)
        self.merges = merges or {}

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size=5000, snapshot_every=250, sample=None):
        freq = pretokenize(text)
        words = {w: list(w) for w in freq}                 # word -> list of symbol ids
        pair_count = defaultdict(int)
        pair_words = defaultdict(set)                       # pair -> words containing it
        for w, ids in words.items():
            for p in _pairs(ids):
                pair_count[p] += freq[w]
                pair_words[p].add(w)

        self.merges = {}
        next_id = 256
        curve = []                                          # (vocab_size, tokens_per_byte)
        t0 = time.time()
        while next_id < vocab_size and pair_count:
            best = max(pair_count, key=pair_count.get)
            if pair_count[best] <= 0:
                break
            self.merges[best] = next_id
            for w in list(pair_words[best]):
                ids = words[w]
                f = freq[w]
                for p in _pairs(ids):                       # remove this word's old pairs
                    pair_count[p] -= f
                    if pair_count[p] <= 0:
                        pair_count.pop(p, None); pair_words.pop(p, None)
                    else:
                        pair_words[p].discard(w)
                new_ids = self._merge_seq(ids, best, next_id)
                words[w] = new_ids
                for p in _pairs(new_ids):                    # add its new pairs
                    pair_count[p] += f
                    pair_words[p].add(w)
            next_id += 1
            if sample is not None and (next_id % snapshot_every == 0 or next_id == vocab_size):
                curve.append((next_id, self.tokens_per_byte(sample)))
        self.train_seconds = time.time() - t0
        return curve

    @staticmethod
    def _merge_seq(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    # --------------------------------------------------------------- encode
    def encode_chunk(self, ids):
        # apply merges in the order they were learned (lowest new_id first)
        for pair, new_id in self.merges.items():
            if len(ids) < 2:
                break
            ids = self._merge_seq(ids, pair, new_id)
        return ids

    def encode(self, text):
        out = []
        for chunk in PInitial.findall(text):
            out.extend(self.encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        # rebuild the byte string for each id by expanding merges
        inv = {v: k for k, v in self.merges.items()}
        def expand(i):
            if i < 256:
                return bytes([i])
            a, b = inv[i]
            return expand(a) + expand(b)
        return b"".join(expand(i) for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i):
        inv = {v: k for k, v in self.merges.items()}
        def expand(j):
            if j < 256:
                return bytes([j])
            a, b = inv[j]
            return expand(a) + expand(b)
        return expand(i)

    # --------------------------------------------------------------- metrics
    def tokens_per_byte(self, text):
        nbytes = len(text.encode("utf-8"))
        return len(self.encode(text)) / nbytes

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ----------------------------------------------------------- serialize
    def save(self, path):
        data = {"merges": [[a, b, nid] for (a, b), nid in self.merges.items()]}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls({(a, b): nid for a, b, nid in data["merges"]})


def plot_curve(curve, path):
    import sys
    sys.path.insert(0, str(HERE))
    import plot_style as ps
    xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
    fig, ax = ps.new_axes(7.0, 4.2)
    ax.plot(xs, ys, "-o", color=ps.SERIES[0], markersize=3)
    ax.axhline(1.0, color=ps.SERIES[2], linestyle="--", linewidth=1,
               label="1 token / byte (no compression)")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Compression improves as the vocabulary grows",
              "vocabulary size", "tokens per byte (lower = better)", path)


def plot_token_lengths(bpe, path):
    import sys
    sys.path.insert(0, str(HERE))
    import plot_style as ps
    from collections import Counter
    lengths = Counter(len(bpe.token_bytes(nid)) for nid in bpe.merges.values())
    xs = sorted(lengths)
    fig, ax = ps.new_axes(7.0, 4.0)
    ax.bar(xs, [lengths[x] for x in xs], color=ps.SERIES[1], width=0.8)
    ax.set_xticks(xs)
    ps.finish(fig, ax, "How many bytes each learned token covers",
              "token length (bytes)", "number of tokens", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data/corpus.txt"))
    ap.add_argument("--vocab-size", type=int, default=5000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    text = Path(args.corpus).read_text(encoding="utf-8")
    nbytes = len(text.encode("utf-8"))
    sample = text[:20000]                                   # for the compression curve
    print(f"corpus: {nbytes:,} bytes")

    bpe = BPE()
    curve = bpe.train(text, vocab_size=args.vocab_size, sample=sample)
    print(f"trained vocab {bpe.vocab_size} in {bpe.train_seconds:.1f}s")

    # round-trip verification on held-out-ish text
    probe = text[50000:60000]
    ids = bpe.encode(probe)
    assert bpe.decode(ids) == probe, "round-trip mismatch!"
    tpb = len(ids) / len(probe.encode("utf-8"))
    print(f"round-trip: OK | compression on probe: {tpb:.3f} tokens/byte "
          f"({1/tpb:.2f} bytes/token)")

    # serialize + reload, confirm identical encoding
    bpe.save(OUT / "tokenizer.json")
    reloaded = BPE.load(OUT / "tokenizer.json")
    assert reloaded.encode(probe) == ids, "reload mismatch!"
    print(f"saved + reloaded tokenizer.json ({len(bpe.merges)} merges); identical encoding")

    # top merges (most-frequent-first = lowest new ids)
    top = list(bpe.merges.items())[:25]
    lines = ["rank\tnew_id\ttoken (repr)"]
    for r, (pair, nid) in enumerate(top, 1):
        lines.append(f"{r}\t{nid}\t{bpe.token_bytes(nid).decode('utf-8', 'replace')!r}")
    (OUT / "top_merges.txt").write_text("\n".join(lines) + "\n")
    print("\nfirst learned tokens (most frequent pairs):")
    print("  " + ", ".join(repr(bpe.token_bytes(nid).decode('utf-8', 'replace')) for _, nid in top[:12]))

    plot_curve(curve, OUT / "compression_curve.png")
    plot_token_lengths(bpe, OUT / "token_lengths.png")
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"corpus_bytes,{nbytes}\nvocab_size,{bpe.vocab_size}\n"
        f"probe_tokens_per_byte,{tpb:.4f}\nprobe_bytes_per_token,{1/tpb:.4f}\n"
        f"train_seconds,{bpe.train_seconds:.1f}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
