# Stop-String Matcher

---

> The user wants to stop on `"</answer>"`, but the model thinks in token IDs that don't line up with letters.

---

## Key Insight

This project implements a streaming [stop-string](/shared/glossary/#stop-string) matcher that watches the *decoded text* — not the raw token stream — for a user-supplied substring, then proves it works against pathological inputs where the stop string straddles two [BPE](/shared/glossary/#bpe) tokens. The [tokenizer](/shared/glossary/#tokenizer) chops bytes into pieces that do not respect letter or word boundaries, so the matcher must keep a small rolling buffer of decoded output and only emit characters once they are safely outside any potential match.

## A Concrete Example

Say the stop string is `</answer>` and the model has just produced text ending in `…here is the answer</an`. Those last four characters, `</an`, *might* be the start of `</answer>` — or they might be literal text that just happens to look similar. If the matcher streams them to the user right away and the very next token completes `…swer>`, the user has already seen part of the stop string that was supposed to stay hidden (an **overshoot**).

So the matcher holds `</an` back in a small rolling buffer instead of emitting it:

- If the next token decodes to `swer>`, it now has the full `</answer>`, recognizes the match, and stops — emitting nothing past it.
- If the next token instead decodes to `gle bracket`, the buffer can safely flush `</an` to the user, because `</angle bracket` can no longer become `</answer>`.

Characters only leave the buffer once they are *outside any possible match*. The reason a [token](/shared/glossary/#tokenizer) boundary doesn't help here is that the model emits token IDs, not letters: a single token might decode to `</an`, or `swer>`, or even `er</answ` — the byte pieces don't line up with the stop string, so the matcher has to reason about the decoded text, one buffered character at a time.

## Why This Matters

Stop strings are the most common way users control generation length in chat and tool-use APIs, and a buggy matcher either overshoots (emits a few characters past the stop) or undershoots (stops too early on a partial match). Both fail in user-visible ways, and the bug only shows up on certain prompts — which is why every production serving stack handles this in a careful, well-tested helper.
