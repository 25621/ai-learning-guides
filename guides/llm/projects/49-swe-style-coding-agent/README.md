# [SWE](/shared/glossary/#swe)-Style Coding Agent

---

> Give a model a shell and a test suite, and watch it debug like an engineer.

---

## Key Insight

This project wires an LLM to a shell and a file editor so it can read a codebase, make edits, and run tests in a loop, then points it at a few easy issues from a bug benchmark like [SWE-bench](/shared/glossary/#swe-bench).

## Why This Matters

Fixing a real bug end-to-end is the canonical test of an [agent](/shared/glossary/#agent): it must explore, act, check its own work, and recover from errors — the same loop behind coding assistants like Claude Code.
