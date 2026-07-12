"""A JSON-schema FSM and the token mask it induces — what Outlines does, in miniature.

Outlines and sglang are not installed here, so we build the mechanism instead of
calling it. The mechanism is simpler than it sounds:

  1. Turn the schema into a finite-state machine over *characters*. In state s,
     only some characters can come next and still leave a valid document
     reachable.
  2. The model does not emit characters, it emits *tokens*. So for each state,
     walk every token in the vocabulary through the FSM: a token is allowed iff
     all of its characters are legal, and it lands the FSM in a known next state.
  3. At decode time, add -inf to the logits of every token that is not allowed.
     Sampling can now only produce schema-valid text.

Step 2 is the expensive one (vocabulary x states), which is why real libraries
*compile* the index once and cache it. We do the same, and measure both costs
separately: the one-off compile, and the per-token mask lookup.
"""

import torch

# The schema, written as an alternating template. Fixed key order is what makes
# this a simple FSM; a general JSON-schema compiler adds branching for optional
# keys and arrays, but every one of them reduces to this same idea.
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                   "city": {"type": "string"}},
    "required": ["name", "age", "city"],
}

STRING_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,'-"
)
DIGITS = set("0123456789")


class Lit:
    """A literal stretch of the document — punctuation and the key names."""

    def __init__(self, text):
        self.text = text


class Free:
    """A value the model gets to choose, within a character class and a length range."""

    def __init__(self, charset, lo, hi):
        self.charset, self.lo, self.hi = charset, lo, hi


TEMPLATE = [
    Lit('{"name": "'), Free(STRING_CHARS, 1, 32),
    Lit('", "age": '), Free(DIGITS, 1, 3),
    Lit(', "city": "'), Free(STRING_CHARS, 1, 32),
    Lit('"}'),
]
ACCEPT = (len(TEMPLATE), 0)


def transitions(state):
    """char -> next state, for every character legal in `state`."""
    seg, off = state
    if seg >= len(TEMPLATE):
        return {}                                   # document complete: only EOS
    s = TEMPLATE[seg]

    if isinstance(s, Lit):
        ch = s.text[off]
        nxt = (seg, off + 1) if off + 1 < len(s.text) else (seg + 1, 0)
        return {ch: nxt}

    out = {}
    if off < s.hi:                                  # keep extending the value
        for ch in s.charset:
            out[ch] = (seg, off + 1)
    if off >= s.lo:                                 # or finish it and move on
        out.update(transitions((seg + 1, 0)))       # (charsets never overlap the
    return out                                      #  literal that follows them)


def step_string(state, text):
    """Run `text` through the FSM. Returns the new state, or None if it goes off-schema."""
    for ch in text:
        t = transitions(state)
        if ch not in t:
            return None
        state = t[ch]
    return state


class TokenMasker:
    """The compiled index: for each FSM state, which token ids are legal."""

    def __init__(self, tok, eos_id, logit_width=None):
        self.tok = tok
        self.eos_id = eos_id
        self.n_tokens = len(tok)
        # The mask must be as wide as the model's *logits*, which is not the same as
        # the tokenizer's vocabulary: Qwen2.5 pads its embedding matrix to 151,936
        # rows while the tokenizer defines 151,665 tokens. Those extra rows are real
        # logits the model can argmax into, and they decode to nothing — so they must
        # be masked out too, not left off the end of the mask.
        self.vocab_size = logit_width or self.n_tokens
        # Decode every token once. A token's *string* is all the FSM cares about.
        self.strings = tok.batch_decode([[i] for i in range(self.n_tokens)])
        specials = set(tok.all_special_ids)
        self.usable = [i for i in range(self.n_tokens)
                       if i not in specials and self.strings[i]]
        self.cache = {}                             # state -> (mask, {token_id: next_state})
        self.compiled_states = 0

    def index(self, state):
        """Mask + transition table for one state, compiled on first use and cached."""
        if state in self.cache:
            return self.cache[state]

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        nxt = {}
        if state == ACCEPT:
            mask[self.eos_id] = True                # the only legal move is to stop
        else:
            for i in self.usable:
                s2 = step_string(state, self.strings[i])
                if s2 is not None:
                    mask[i] = True
                    nxt[i] = s2
        self.cache[state] = (mask, nxt)
        self.compiled_states += 1
        return self.cache[state]

    def precompile(self):
        """Walk every state the schema can reach and build its mask up front.

        This is what Outlines and sglang do when you hand them a schema: pay the
        vocabulary x states cost once, at startup, so that decoding itself is only
        a lookup. Charging that one-off compile to the first request (as a lazy
        index does) makes constrained decoding look far more expensive than it is.
        """
        seen, stack = set(), [(0, 0)]
        while stack:
            st = stack.pop()
            if st in seen:
                continue
            seen.add(st)
            _, nxt = self.index(st)
            stack.extend(s for s in set(nxt.values()) if s not in seen)
        self.index(ACCEPT)
        return len(seen)

    def apply(self, logits, state):
        """The one line that makes invalid output impossible."""
        mask, _ = self.index(state)
        return logits.masked_fill(~mask, -float("inf"))

    def advance(self, state, token_id):
        if token_id == self.eos_id:
            return ACCEPT
        return self.index(state)[1][token_id]
