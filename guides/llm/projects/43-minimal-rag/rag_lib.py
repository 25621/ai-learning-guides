"""Shared Phase-7 RAG stack — corpus, embedder, retriever, reader, metrics.

One place for everything the retrieval projects (43-46) share:

- **Corpus**: SQuAD v1.1 dev — 2,067 real Wikipedia paragraphs across 48
  articles, each with crowd-written questions whose answers are literal spans
  in one gold paragraph. That gives retrieval a ground truth (did we fetch the
  gold paragraph?) and generation a ground truth (does the produced answer
  match the human answers?) without any LLM-as-judge.
- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` run through plain
  `transformers` (mean-pool + L2-normalize) — the standard tiny bi-encoder.
- **Reader**: `distilbert-base-cased-distilled-squad`, an extractive QA model
  that plays the "generation" role on CPU: given the retrieved context it
  produces an answer span. If retrieval misses, the reader cannot recover —
  which is exactly the property a RAG evaluation needs.
- **BM25 / RRF / nDCG**: implemented from scratch (no external IR libs).

Imported by projects 44, 45, and 46 via sys.path.
"""

import json
import os
import re
import string
import urllib.request
from collections import Counter

import numpy as np
import torch

torch.set_num_threads(12)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SQUAD_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
    "master/dataset/dev-v1.1.json"
)

EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"
READER_ID = "distilbert-base-cased-distilled-squad"
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------- corpus

def load_squad():
    """Download (once) and parse SQuAD v1.1 dev into article dicts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "squad_dev_v1.1.json")
    if not os.path.exists(path):
        print("downloading SQuAD v1.1 dev ...")
        urllib.request.urlretrieve(SQUAD_URL, path)
    with open(path) as f:
        return json.load(f)["data"]


def build_corpus(n_paragraphs=1000, n_questions=300, seed=0):
    """Sample a paragraph corpus + evaluation questions grounded in it.

    Returns (paragraphs, questions):
      paragraphs — list of {"title", "text"}
      questions  — list of {"q", "answers", "gold": paragraph index}
    """
    rng = np.random.default_rng(seed)
    articles = load_squad()
    all_paras = []
    for art in articles:
        for para in art["paragraphs"]:
            all_paras.append((art["title"], para))
    order = rng.permutation(len(all_paras))[:n_paragraphs]
    paragraphs, questions = [], []
    for gold_idx, pi in enumerate(order):
        title, para = all_paras[pi]
        paragraphs.append({"title": title, "text": para["context"]})
        for qa in para["qas"]:
            questions.append({
                "q": qa["question"].strip(),
                "answers": [a["text"] for a in qa["answers"]],
                "gold": gold_idx,
            })
    pick = rng.permutation(len(questions))[:n_questions]
    questions = [questions[i] for i in pick]
    print(f"corpus: {len(paragraphs)} paragraphs, {len(questions)} questions")
    return paragraphs, questions


# ---------------------------------------------------------------- embedder

class Embedder:
    """MiniLM bi-encoder: texts -> L2-normalized 384-d vectors."""

    def __init__(self, max_len=256):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER_ID)
        self.model = AutoModel.from_pretrained(EMBEDDER_ID).eval()
        self.max_len = max_len  # MiniLM's absolute cap is 512 wordpieces

    @torch.no_grad()
    def encode(self, texts, batch_size=64, verbose=False):
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(torch.nn.functional.normalize(pooled, dim=-1))
            if verbose and (i // batch_size) % 5 == 0:
                print(f"  embedded {i + len(batch['input_ids'])}/{len(texts)}",
                      flush=True)
        return torch.cat(vecs).numpy()


def dense_search(query_vecs, doc_vecs, k):
    """Cosine top-k (vectors are unit-norm, so dot product = cosine)."""
    sims = query_vecs @ doc_vecs.T
    top = np.argsort(-sims, axis=1)[:, :k]
    return top, np.take_along_axis(sims, top, axis=1)


# ---------------------------------------------------------------- BM25

_token_re = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text):
    return _token_re.findall(text.lower())


class BM25:
    """Okapi BM25 from scratch (k1=1.5, b=0.75) over a list of strings."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [bm25_tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens])
        self.avg_len = self.doc_len.mean()
        self.doc_tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: np.log(1 + (n - d + 0.5) / (d + 0.5))
                    for t, d in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.doc_tf))
        for t in bm25_tokenize(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (
                        1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                    s[i] += idf * f * (self.k1 + 1) / denom
        return s

    def search(self, query, k):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return top, s[top]


def rrf_fuse(rank_lists, k=60, top_k=10):
    """Reciprocal rank fusion: score(d) = sum over lists of 1/(k + rank_d)."""
    scores = Counter()
    for ranks in rank_lists:
        for r, doc in enumerate(ranks):
            scores[doc] += 1.0 / (k + r + 1)
    fused = [d for d, _ in scores.most_common(top_k)]
    return fused


# ---------------------------------------------------------------- reader

class Reader:
    """Extractive QA reader: (question, contexts) -> best answer span.

    Long contexts are windowed with a stride so nothing is silently cut off;
    the best-scoring span across all windows and contexts wins.
    """

    def __init__(self, max_len=384, stride=128):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(READER_ID)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            READER_ID).eval()
        self.max_len, self.stride = max_len, stride

    @torch.no_grad()
    def answer(self, question, contexts, batch_size=16):
        enc = self.tok(
            [question] * len(contexts), contexts, truncation="only_second",
            max_length=self.max_len, stride=self.stride, padding=True,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            return_tensors="pt")
        best, best_score = "", -1e9
        n = enc["input_ids"].shape[0]
        for i in range(0, n, batch_size):
            out = self.model(
                input_ids=enc["input_ids"][i:i + batch_size],
                attention_mask=enc["attention_mask"][i:i + batch_size])
            for j in range(out.start_logits.shape[0]):
                w = i + j
                seq_ids = enc.sequence_ids(w)
                ctx_mask = torch.tensor(
                    [s == 1 for s in seq_ids], dtype=torch.bool)
                start = out.start_logits[j].masked_fill(~ctx_mask, -1e9)
                end = out.end_logits[j].masked_fill(~ctx_mask, -1e9)
                s_idx = int(start.argmax())
                e_cand = end[s_idx:s_idx + 30]
                e_idx = s_idx + int(e_cand.argmax())
                score = float(start[s_idx] + end[e_idx])
                if score > best_score:
                    ctx = contexts[int(enc["overflow_to_sample_mapping"][w])]
                    off = enc["offset_mapping"][w]
                    best = ctx[int(off[s_idx][0]):int(off[e_idx][1])]
                    best_score = score
        return best, best_score


# ---------------------------------------------------------------- metrics

def normalize_answer(s):
    """SQuAD-official normalization: lowercase, drop punctuation/articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred, golds):
    return max(float(normalize_answer(pred) == normalize_answer(g))
               for g in golds)


def f1_score(pred, golds):
    def f1(p, g):
        pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        prec, rec = overlap / len(pt), overlap / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max(f1(pred, g) for g in golds)


def ndcg_at_k(ranked, gold, k=10):
    """Binary relevance, one gold doc: DCG = 1/log2(1+rank), ideal = 1."""
    ranked = list(ranked[:k])
    if gold not in ranked:
        return 0.0
    return 1.0 / np.log2(2 + ranked.index(gold))


def recall_at_k(ranked, gold, k):
    return float(gold in list(ranked[:k]))
