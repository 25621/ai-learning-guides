# Stop-String Matcher

---

> The user wants to stop on `"</answer>"`, but the model thinks in token IDs that don't line up with letters.

---

## Key Insight

This project implements a streaming [stop-string](/shared/glossary/#stop-string) matcher that watches the *decoded text* — not the raw token stream — for a user-supplied substring, then proves it works against pathological inputs where the stop string straddles two [BPE](/shared/glossary/#bpe) tokens. The [tokenizer](/shared/glossary/#tokenizer) chops bytes into pieces that do not respect letter or word boundaries, so the matcher must keep a small rolling buffer of decoded output and only emit characters once they are safely outside any potential match.

## Why This Matters

Stop strings are the most common way users control generation length in chat and tool-use APIs, and a buggy matcher either overshoots (emits a few characters past the stop) or undershoots (stops too early on a partial match). Both fail in user-visible ways, and the bug only shows up on certain prompts — which is why every production serving stack handles this in a careful, well-tested helper.
