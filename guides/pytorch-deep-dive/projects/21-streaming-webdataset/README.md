# Streaming WebDataset

---

> You don't need the whole dataset on disk — you just need the next batch, right now.

---

## Key Insight

[WebDataset](/shared/glossary/#webdataset) reads training data straight from `.tar` archives as a stream, one sample after another, instead of unpacking millions of files first. The archives are split into [shards](/shared/glossary/#sharding) so many workers — and many machines — can each read a different piece in parallel.

## Why This Matters

At large scale, storing and opening millions of tiny files is slow and sometimes impossible. Streaming sharded archives lets you train on datasets far bigger than your local disk, reading them directly from cloud storage.
