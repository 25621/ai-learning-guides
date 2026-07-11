"""The shared Phase-8 (Evaluation) stack.

One place for the machinery every evaluation project in this phase reuses:

  * `Model`          — a thin wrapper over a Hugging Face causal LM that gives
                       you the two primitives an eval needs on a CPU budget:
                         - `mc_score`  : score multiple-choice questions by the
                                         log-prob the model puts on each answer
                                         *letter* (one forward pass, batched).
                         - `generate`  : batched greedy / sampled generation for
                                         open-ended answers.
                       Plus `judge_choice`, a loglik verdict used by the
                       LLM-as-judge, custom-eval, and arena projects.
  * `load_parquet`   — pull any Hugging Face dataset split through the public
                       parquet API (no `datasets` library needed) and cache it.
  * MMLU helpers     — loader, the 57-subject → 4-category map, and the cloze
                       prompt format that reproduces published numbers.

Imported by projects 52-57 via `sys.path`.  CPU-only; every model here is small
enough (124M-500M params) to run a real evaluation in minutes.
"""

import io
import json
import os
import urllib.request

import numpy as np
import torch

torch.set_num_threads(12)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Datasets and model caches live next to this file, gitignored.  Every project
# in the phase shares the one download.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset access — the Hugging Face parquet API, no `datasets` dependency.
# --------------------------------------------------------------------------- #
def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "phase8-eval"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def load_parquet(dataset, config, split, columns=None):
    """Return a HF dataset split as a dict of column-lists, cached to DATA_DIR.

    Uses the datasets-server parquet endpoint, which serves every public
    dataset as parquet URLs — so we never need the (uninstalled) `datasets`
    library.  `pyarrow` does the decode.
    """
    import pyarrow.parquet as pq

    cache = os.path.join(DATA_DIR, f"{dataset}__{config}__{split}.json".replace("/", "_"))
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    api = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    urls = json.loads(_get(api))
    if isinstance(urls, dict):  # some datasets nest one more level
        urls = urls.get(split, urls)
    rows = {}
    for u in urls:
        tbl = pq.read_table(io.BytesIO(_get(u)))
        d = tbl.to_pydict()
        for k, v in d.items():
            rows.setdefault(k, []).extend(v)
    if columns:
        rows = {k: rows[k] for k in columns}
    with open(cache, "w") as f:
        json.dump(rows, f)
    return rows


# --------------------------------------------------------------------------- #
# The model wrapper — the eval-relevant primitives, batched for the CPU.
# --------------------------------------------------------------------------- #
LETTERS = ["A", "B", "C", "D", "E", "F"]


MAX_LEN = 1024  # a few MMLU subjects (professional_law) have very long stems


class Model:
    def __init__(self, name, dtype=torch.float32):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Left-padding is required so the *last* column is always a real token —
        # that is where we read the next-token distribution for scoring/gen.
        self.tok.padding_side = "left"
        # Over-long prompts are truncated from the *left* so the tail (the answer
        # choices and the "Answer:" cue we score at) always survives.
        self.tok.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.model.eval()
        # Cache the token id of each answer letter (with a leading space, the
        # form the model actually emits after "Answer:").
        self._letter_ids = [
            self.tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS
        ]

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    # -- multiple choice: read P(letter) at the answer position ------------- #
    def _last_logits(self, prompts, batch_size=16):
        """Logits over the vocabulary at the final (real) position of each prompt.

        Two things make this affordable on a CPU:
          * Prompts are length-sorted before batching, so a batch of short
            prompts isn't padded out to the length of the longest in the set.
          * `logits_to_keep=1` asks the model to run its `lm_head` on the final
            position only. Without it the model materializes a (B, T, V) logit
            tensor — with a 152k vocab and a 1k-token prompt that is tens of GB
            per batch, which is exactly how this OOMs.
        """
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = self.tok(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            with torch.no_grad():
                # left-pad => the last column is a real token for every row
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
            for j, i in enumerate(idx):
                out[i] = logits[j]
        return torch.stack(out)

    def mc_score(self, prompts, n_choices=4, batch_size=16):
        """Predicted answer index for each cloze prompt (ends in "Answer:").

        Returns (preds, probs) where probs is the softmax over the letter
        tokens — a calibrated confidence you can threshold or inspect.
        """
        logits = self._last_logits(prompts, batch_size)
        letter_logits = logits[:, self._letter_ids[:n_choices]]  # (B, n_choices)
        probs = torch.softmax(letter_logits, dim=-1)
        return probs.argmax(-1).tolist(), probs.numpy()

    # -- open-ended generation --------------------------------------------- #
    def generate(self, prompts, max_new_tokens=48, temperature=0.0, batch_size=8,
                 seed=0):
        outs = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        do_sample = temperature > 0
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = self.tok(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            if do_sample:
                torch.manual_seed(seed + b)
            with torch.no_grad():
                g = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    pad_token_id=self.tok.pad_token_id,
                )
            gen = g[:, enc.input_ids.shape[1] :]
            for j, i in enumerate(idx):
                outs[i] = self.tok.decode(gen[j], skip_special_tokens=True).strip()
        return outs

    def has_chat_template(self):
        return getattr(self.tok, "chat_template", None) is not None

    def chat(self, user_msgs, system=None):
        """Render single-turn user messages with the chat template.

        Base models (gpt2, bloom, ...) ship no chat template; for them we fall
        back to a plain prompt so an arena can mix instruct and base models
        fairly — each gets the format it was trained to expect."""
        prompts = []
        for u in user_msgs:
            if self.has_chat_template():
                msgs = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": u}
                ]
                prompts.append(self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            else:
                pre = f"{system}\n\n" if system else ""
                prompts.append(f"{pre}{u}\n")
        return prompts

    def answer(self, questions, system=None, max_new_tokens=48, temperature=0.0,
               batch_size=8, seed=0):
        """Chat-template a batch of questions and return short answers."""
        return self.generate(
            self.chat(questions, system=system),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            seed=seed,
        )

    # -- loglik judge: which option does the model prefer at a decision point #
    def judge_choice(self, prompts, options, batch_size=16):
        """For prompts ending right before a one-token verdict, return the index
        of the highest-logprob option token in `options` (e.g. ["A","B"]).

        This is LLM-as-judge without paying for generation: we read the model's
        next-token distribution restricted to the allowed verdict tokens.
        """
        logits = self._last_logits(prompts, batch_size)
        opt_ids = [self.tok(o, add_special_tokens=False).input_ids[-1] for o in options]
        sub = logits[:, opt_ids]
        return sub.argmax(-1).tolist(), torch.softmax(sub, -1).numpy()

    def score_tokens(self, prompts, tokens, batch_size=16):
        """Softmax probabilities over an arbitrary set of next-token `tokens`
        (e.g. the digits "1".."5" for a rubric score)."""
        logits = self._last_logits(prompts, batch_size)
        tok_ids = [self.tok(t, add_special_tokens=False).input_ids[-1] for t in tokens]
        return torch.softmax(logits[:, tok_ids], -1).numpy()


# --------------------------------------------------------------------------- #
# MMLU — loader, category map, and the prompt format that reproduces numbers.
# --------------------------------------------------------------------------- #
# The 57 subjects, grouped into the 4 macro-categories the MMLU paper reports.
MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology",
    ],
}
SUBJECT_TO_CATEGORY = {
    s: cat for cat, subs in MMLU_CATEGORIES.items() for s in subs
}
ALL_SUBJECTS = [s for subs in MMLU_CATEGORIES.values() for s in subs]


def load_mmlu(subjects=None, per_subject=None, seed=0):
    """Return a list of MMLU items: dict(question, choices, answer, subject).

    `per_subject` caps items per subject (a stratified sample) so the whole
    run fits a CPU time budget while staying balanced across categories.
    """
    subjects = subjects or ALL_SUBJECTS
    rng = np.random.default_rng(seed)
    items = []
    for subj in subjects:
        d = load_parquet("cais/mmlu", subj, "test")
        n = len(d["question"])
        idx = np.arange(n)
        if per_subject and n > per_subject:
            idx = rng.choice(n, per_subject, replace=False)
        for i in idx:
            items.append(
                dict(
                    question=d["question"][i],
                    choices=list(d["choices"][i]),
                    answer=int(d["answer"][i]),
                    subject=subj,
                )
            )
    return items


def mmlu_cloze(question, choices, style="plain"):
    """The prompt formats used by projects 51 (plain) and 52 (the sweep).

    All end just before the answer letter so `mc_score` reads P(letter)."""
    lettered = "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(choices))
    if style == "plain":
        return f"Question: {question}\n{lettered}Answer:"
    if style == "parenthesized":
        opts = "".join(f"({LETTERS[i]}) {c}\n" for i, c in enumerate(choices))
        return f"Question: {question}\n{opts}The answer is ("
    if style == "no_labels":
        return f"{question}\n{lettered}Answer:"
    if style == "verbose":
        return (
            "You are an expert taking a multiple-choice exam. Read the question "
            "and choose the single best answer.\n\n"
            f"Question: {question}\nOptions:\n{lettered}"
            "Respond with just the letter of the correct option.\nAnswer:"
        )
    if style == "qa_pairs":
        return f"Q: {question}\n{lettered}A:"
    if style == "chat":
        # The Qwen2.5 chat template, applied by hand. Instruct models are tuned
        # to answer inside these role markers; scoring the *cloze* format above
        # (which they never saw in post-training) gives a different number — that
        # gap is one of the wordings project 52 measures.
        body = f"Question: {question}\n{lettered}Reply with only the letter."
        return (f"<|im_start|>user\n{body}<|im_end|>\n"
                f"<|im_start|>assistant\nThe answer is")
    raise ValueError(style)


# --------------------------------------------------------------------------- #
# Small statistics helpers reused across projects.
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion — honest error bars on
    accuracy at the sample sizes a CPU eval can afford."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(a, b):
    """Agreement between two raters' label lists, corrected for chance."""
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    po = np.mean(a == b)
    ca = np.zeros(len(labels))
    cb = np.zeros(len(labels))
    for x in a:
        ca[idx[x]] += 1
    for x in b:
        cb[idx[x]] += 1
    pe = np.sum((ca / n) * (cb / n))
    return (po - pe) / (1 - pe + 1e-12)
