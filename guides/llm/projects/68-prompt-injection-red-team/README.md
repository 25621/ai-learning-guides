# Prompt-Injection Red Team

---

> The model cannot tell instructions from data — so anything it reads can become an instruction.

---

## Key Insight

This project builds a small tool-using [agent](/shared/glossary/#agent) that reads retrieved documents, then attempts 20 [prompt injection](/shared/glossary/#prompt-injection) attacks — adversarial text hidden inside the documents that tries to override the system instructions — and records which defenses help and which fail.

## Why This Matters

Any LLM that reads untrusted text (web pages, emails, tool outputs, images) is a potential victim of prompt injection, and no single fix exists; running an injection red team against your own setup is the practical way to find which mitigations actually narrow the attack surface for your application.
