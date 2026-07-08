"""Chat-template debugger: render a chat template by hand, byte-compare it against
the model's own `apply_chat_template`, and catch a one-character whitespace bug.

A fine-tuned chat model learned to respond to ONE exact byte pattern of special
tokens and whitespace. If your serving code renders that pattern even one newline
differently from training, the model silently degrades — no error, just worse
answers. This project reproduces the classic mistake (a missing newline after
`<|im_end|>`), shows the exact byte and token where it bites, then fixes it to a
byte-perfect match.

    python chat_debug.py      # ~30s on CPU (tokenizer only, no model weights)

Uses Qwen2.5's ChatML template as the reference.
"""

from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello there."},
    {"role": "user", "content": "What is 2+2?"},
]


def render_buggy(messages):
    """A plausible hand-rolled ChatML renderer with ONE bug: it forgets the
    newline after each <|im_end|>, so turns run together."""
    s = ""
    for m in messages:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"   # <-- missing \n
    s += "<|im_start|>assistant\n"
    return s


def render_fixed(messages):
    """The correct ChatML renderer: <|im_start|>{role}\n{content}<|im_end|>\n."""
    s = ""
    for m in messages:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    s += "<|im_start|>assistant\n"
    return s


def vis(s):
    return s.replace(" ", "·").replace("\n", "↵")


def first_diff(a, b):
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def plot_diff(ref, buggy, idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo = max(0, idx - 28); hi = idx + 28
    ref_win, bug_win = vis(ref[lo:hi]), vis(buggy[lo:hi])
    # index of the differing char within the window (visualization is 1:1 since
    # each char maps to one glyph)
    off = idx - lo
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.axis("off")
    ax.text(0.01, 0.78, "reference (apply_chat_template):", fontsize=9, color="#52514e")
    ax.text(0.01, 0.60, ref_win, fontsize=12, family="monospace", color="#0b0b0b")
    ax.text(0.01, 0.30, "hand-rendered (buggy):", fontsize=9, color="#52514e")
    ax.text(0.01, 0.12, bug_win, fontsize=12, family="monospace", color="#0b0b0b")
    # highlight the differing column with a red marker under both rows
    charw = 0.0092
    xx = 0.01 + off * charw
    ax.annotate("▲ first mismatch here", (xx, 0.02), fontsize=9, color="#e34948")
    ax.axvline(xx + 0.004, ymin=0.05, ymax=0.72, color="#e34948", linewidth=1.2, alpha=0.7)
    ax.set_title("One missing ↵ (newline) between training and inference",
                 fontsize=12, color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    reference = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    buggy = render_buggy(MESSAGES)
    fixed = render_fixed(MESSAGES)
    # Tokenize both strings the same way (special-token literals are recognized);
    # this is exactly what a serving stack feeds the model.
    ref_ids = tok.encode(reference, add_special_tokens=False)

    (OUT / "reference_template.txt").write_text(reference)

    # 1) byte comparison of the buggy render
    idx = first_diff(reference, buggy)
    ok_buggy = reference == buggy
    print(f"reference length: {len(reference)} chars | buggy length: {len(buggy)} chars")
    print(f"buggy matches reference? {ok_buggy}")
    print(f"first byte mismatch at index {idx}:")
    print(f"  reference[{idx-6}:{idx+10}] = {vis(reference[idx-6:idx+10])!r}")
    print(f"  buggy    [{idx-6}:{idx+10}] = {vis(buggy[idx-6:idx+10])!r}")

    # 2) the token-level consequence
    bug_ids = tok.encode(buggy, add_special_tokens=False)
    print(f"\ntoken counts: reference {len(ref_ids)} vs buggy {len(bug_ids)} "
          f"(off by {len(ref_ids) - len(bug_ids)})")
    tdiff = next((i for i, (a, b) in enumerate(zip(ref_ids, bug_ids)) if a != b), None)
    if tdiff is not None:
        print(f"first differing token at position {tdiff}: "
              f"reference id {ref_ids[tdiff]} ({tok.decode([ref_ids[tdiff]])!r}) "
              f"vs buggy id {bug_ids[tdiff]} ({tok.decode([bug_ids[tdiff]])!r})")

    # 3) the fix: byte-perfect match
    ok_fixed = reference == fixed
    print(f"\nfixed hand-render matches reference byte-for-byte? {ok_fixed}")
    assert ok_fixed, "the 'fixed' renderer should be byte-identical to the reference"

    plot_diff(reference, buggy, idx, OUT / "diff.png")
    (OUT / "diff.txt").write_text(
        f"reference vs buggy hand-render\n"
        f"first mismatch at char {idx}\n"
        f"  reference: ...{vis(reference[idx-10:idx+14])}...\n"
        f"  buggy    : ...{vis(buggy[idx-10:idx+14])}...\n"
        f"token counts: reference {len(ref_ids)} vs buggy {len(bug_ids)}\n"
        f"fixed render byte-identical to reference: {ok_fixed}\n")
    (OUT / "results.csv").write_text(
        "metric,value\n"
        f"reference_chars,{len(reference)}\nbuggy_matches,{ok_buggy}\n"
        f"first_mismatch_char,{idx}\nreference_tokens,{len(ref_ids)}\n"
        f"buggy_tokens,{len(bug_ids)}\nfixed_matches,{ok_fixed}\n")
    print(f"wrote figures + {OUT/'results.csv'}")


if __name__ == "__main__":
    main()
