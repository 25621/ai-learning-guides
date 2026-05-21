# Memory-Mapped Tokens

---

> Don't load the file into memory — let the operating system pretend it already is.

---

## Key Insight

[Memory mapping](/shared/glossary/#memory-mapping) (via `numpy.memmap`) makes a file on disk look like an array in memory: you can read any slice of it without loading the whole file into RAM. After [tokenizing](/shared/glossary/#tokenizer) a huge text corpus into one flat `.bin` file, training reads small chunks on demand.

## Why This Matters

Language-model datasets are often far larger than RAM. Memory mapping lets you train on a corpus of any size while using almost no memory, because the operating system pulls in only the pieces you actually touch.
