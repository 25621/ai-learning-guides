# Benchmark Contamination Check

## Key Insight

When a [benchmark](/shared/glossary/#benchmark)'s test questions have already appeared in a model's [pretraining](/shared/glossary/#pretraining) data, a high score measures memorization, not skill — this is [contamination](/shared/glossary/#contamination) (also called leakage), the AI equivalent of a student who studied from a stolen copy of the exam. This project searches a slice of a pretraining corpus for verbatim or near-verbatim copies of a benchmark's questions, typically by checking [n-gram](/shared/glossary/#n-gram) overlap — sliding a window of, say, 13 consecutive words across both and flagging exact matches — to estimate how much of the model's apparent "skill" is really recall. The lesson is a habit, not a tool: any suspiciously strong result deserves a contamination check before you believe it, because as benchmarks age they steadily seep into the web crawls that train the next generation of models.
