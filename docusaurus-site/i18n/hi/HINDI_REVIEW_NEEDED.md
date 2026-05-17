# Hindi Translation Review Needed

The following docs were brought to structural parity with the English source
in a single sweep. The new/rewritten sections are best-effort translations
by a non-native Hindi speaker (Claude) and need a native-speaker review pass
for terminology, idiom, and tone before being treated as finished.

## Files rewritten

All of these had abridged or restructured Hindi versions that did not
mirror the English structure. Each was rewritten section-by-section
to match the English doc.

- `docusaurus-plugin-content-docs/current/policy_gradients/ppo_scratch_explained.md`
  (was 7 sections, now 11 — added: Problem-with-A2C, Core-Idea, Why-Catastrophic, GAE, Multiple-Epochs, Full-Loss, Results, Key-Equations)
- `docusaurus-plugin-content-docs/current/policy_gradients/reinforce_cartpole_explained.md`
  (was 9 sections in a different organization, now 9 mirroring en)
- `docusaurus-plugin-content-docs/current/policy_gradients/ppo_continuous_explained.md`
  (was 6 sections about Pendulum, now 11 mirroring en with BipedalWalker)
- `docusaurus-plugin-content-docs/current/policy_gradients/ppo_hyperparams_explained.md`
  (was 5 sections, now 8)
- `docusaurus-plugin-content-docs/current/policy_gradients/reinforce_baseline_explained.md`
  (was 8 sections, now 10)
- `docusaurus-plugin-content-docs/current/function_approximation/dqn_target_network_explained.md`
  (was 7 sections, now 8 — added Deadly-Triad)
- `docusaurus-plugin-content-docs/current/function_approximation/linear_q_cartpole_explained.md`
  (was 6 sections, now 8)

## What to check

- **Terminology consistency** — RL terms (पॉलिसी, एडवांटेज, रिटर्न, बेसलाइन, etc.)
  should match the conventions used elsewhere in the hi translation.
- **Tone** — the existing hi style uses Sie-form-equivalent polite second person
  with parenthetical English terms. The rewrites attempt to follow this; spot-check.
- **Technical accuracy** — formulas, code blocks, and numeric values were
  preserved from the English source unchanged.
- **Idiom** — non-native phrasings, awkward word order, or English-influenced
  syntax should be smoothed out.

## Anchor infrastructure (do not change)

The following English anchor pins are load-bearing for cross-locale links and
must be preserved:

- `multi_armed_bandit_explained.md` → `{#the-epsilon-greedy-strategy}`
- `ppo_scratch_explained.md` → `{#the-clipping-trick}` and `{#gae-smarter-advantage-estimates}`
- `reinforce_cartpole_explained.md` → `{#the-old-way-vs-the-new-way}`

Heading text can be retranslated for fluency; the `{#english-slug}` suffix must stay.
