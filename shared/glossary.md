# Glossary

Terms from all guides in this repository, sorted alphabetically. Each guide's own concepts are included here; see individual guides for deeper context.

---

### (2+1)D {#21d}
Factorized spatiotemporal architecture: separate spatial and temporal layers

### 3D VAE {#3d-vae}
Variational autoencoder that compresses video in time as well as space

### ABA {#aba}
Articulated-Body Algorithm — `O(n)` forward dynamics for rigid-body chains

### Ablation {#ablation}
A controlled experiment that changes exactly one factor (a data step, a layer, a hyperparameter) while holding everything else fixed, to measure that factor's true effect.

### Activation checkpointing {#activation-checkpointing}
A memory-saving trick that throws away the intermediate [activations](/shared/glossary/#activations) from the forward pass and recomputes them during the [backward pass](/shared/glossary/#backward-pass) — trading a little extra compute for a lot less memory. Also called [gradient checkpointing](/shared/glossary/#gradient-checkpointing).

### Activations {#activations}
The intermediate tensor outputs produced by the layers of a neural network during the forward pass.

### AdaLN {#adaln}
Adaptive layer normalization; the conditioning mechanism in DiT

### AdaLN-Zero {#adaln-zero}
DiT's conditioning mechanism: layer norm modulated by shift/scale/gate, all initialized to zero

### Adam {#adam}
Adaptive Moment Estimation — gradient-descent optimizer that maintains per-parameter running averages of the first (mean) and second (uncentered variance) moments of the gradients to compute individual adaptive learning rates.

### AdamW {#adamw}
Adam optimizer with decoupled weight decay: the regularization term shrinks the parameter directly rather than being folded into the gradient update

### ADD (Adversarial Diffusion Distillation) {#add-adversarial-diffusion-distillation}
SDXL Turbo's recipe: distill a multi-step diffusion model into a 1–4-step student using a discriminator loss

### Admission control {#admission-control}
Refusing requests early when capacity is saturated, to protect SLOs for accepted requests

### Advantage {#advantage}
`A(s, a) = Q(s, a) − V(s)` — how much better than the baseline this action is

### Agent {#agent}
An [LLM](/shared/glossary/#llm) placed in a loop so it can plan, choose a tool, act, observe the result, and repeat until a task is finished — turning a one-shot answerer into something that carries out multi-step work, like a worker who keeps taking the next action until the whole job is done.

### AI (arithmetic intensity) {#ai-arithmetic-intensity}
FLOPs per byte of memory accessed; determines roofline position

### Alignment (multimodal) {#alignment-multimodal}
Making embeddings from different modalities comparable in a shared space

### Alignment stack {#alignment-stack}
The layered sequence of post-training steps that turns a raw [base model](/shared/glossary/#base-model) into a helpful, safe assistant — typically [SFT](/shared/glossary/#sft), then a [reward model](/shared/glossary/#reward-model), then [RLHF](/shared/glossary/#rlhf) (or [DPO](/shared/glossary/#dpo)). Like the stations on an assembly line, each layer builds on the one below it: the model first learns to follow instructions, then learns what people prefer, then is tuned to actually prefer it. "Alignment" here means getting the model's behavior to match human intent.

### AllReduce {#allreduce}
A collective op that sums tensors across all ranks and gives every rank the result

### AMP {#amp}
Automatic Mixed Precision — running operations in 16-bit floats ([float16](/shared/glossary/#float16) or [bfloat16](/shared/glossary/#bfloat16)) where it is safe, to save memory and speed up training while keeping a [float32](/shared/glossary/#float32) copy of the weights.

### Anomaly detection {#anomaly-detection}
A debugging mode (`torch.autograd.set_detect_anomaly(True)`) that makes [autograd](/shared/glossary/#autograd) check each operation and raise an error at the exact line that first produces a [NaN](/shared/glossary/#nan) or infinite [gradient](/shared/glossary/#gradients).

### AOTInductor {#aotinductor}
Ahead-of-Time Inductor — a deployment path built on [`torch.export`](/shared/glossary/#torchexport) that compiles a captured model graph into a standalone shared library (`.so`) ahead of time, enabling C++-only inference without a Python runtime.

### AnyRes {#anyres}
Dynamic-resolution input handling (tile images at native aspect ratio)

### Application {#application}
The specific real-world job a model is being built to do — for example, "answer customer-support questions about our refund policy," "summarize internal engineering tickets," or "write product descriptions in our brand voice." A model that scores high on a generic public [benchmark](/shared/glossary/#benchmark) can still flop on *your* application if the two don't match, the way a chef who aces a fine-dining contest may still be the wrong hire for your taco truck. That mismatch is why teams build a small targeted eval shaped like their application instead of trusting a famous leaderboard number.

### AprilTag {#apriltag}
Square fiducial marker with a known code; widely used for pose ground truth

### Arena {#arena}
A way to rank chat models by having them go head-to-head: two models answer the same prompt, a human or [LLM judge](/shared/glossary/#llm-as-judge) picks the winner, and many such duels are turned into [Elo](/shared/glossary/#elo) ratings — the scoring system used for chess players. The public [LMSys Chatbot Arena](https://lmarena.ai/) is the best-known example.

### argmax {#argmax}
The "which one is biggest?" operation: given a list of scores it returns the *position* of the largest one, not the value itself. If the [logits](/shared/glossary/#logits) are `[1.2, 4.8, 0.3]`, `argmax` is `1` — the index of `4.8` — which the model reads as "pick token #1." Like scanning a class's test scores and naming the top student rather than reading out their mark. [Greedy decoding](/shared/glossary/#greedy-decoding) is just `argmax` applied to the logits at every step, so it always makes the same choice and never gambles.

### ATen {#aten}
The C++ tensor library underneath PyTorch's Python frontend

### Attention {#attention}
The operation `softmax(QKᵀ/√d) V` — [content-addressable token mixing](/shared/glossary/#content-addressable-token-mixing); the core of every [transformer](/shared/glossary/#transformer)

### autograd {#autograd}
The [reverse-mode](/shared/glossary/#reverse-mode) automatic differentiation engine

### AWQ {#awq}
Activation-aware Weight Quantization — preserve weights important to large activations

### Backend {#backend}
A device- or library-specific implementation that actually executes an operation's [kernel](/shared/glossary/#kernel) — for example the CPU, [CUDA](/shared/glossary/#cuda), or [MPS](/shared/glossary/#mps) backend. The [dispatcher](/shared/glossary/#dispatcher) routes each call to the correct backend based on the tensor's device and [dtype](/shared/glossary/#dtype).

### Backward pass {#backward-pass}
The process of traversing the computation graph in reverse to compute gradients using the chain rule.

### Base model {#base-model}
A model fresh out of [pretraining](/shared/glossary/#pretraining) that only continues text and has not yet been taught to follow instructions — a brilliant autocomplete, not yet an assistant.

### Batch {#batch}
A small group of examples (sentences, images, prompts) that the model processes together in a single forward pass instead of one at a time. Like a chef who slices a whole basket of onions at once rather than picking up the knife for each onion separately — the GPU pays a fixed startup cost per pass, so doing 32 examples in one shot is far faster than 32 single passes. In training, the batch size sets how many examples contribute to each gradient update; in [quantization](/shared/glossary/#quantization) methods like [GPTQ](/shared/glossary/#gptq), a small *calibration batch* of representative inputs is run through the model to estimate which weights matter most. See also [Batching](/shared/glossary/#batching), which is the same idea applied to grouping inference *requests* on a serving stack.

### Batching {#batching}
Grouping multiple inference requests together into a single forward pass so the GPU processes them in parallel, increasing [throughput](/shared/glossary/#throughput) at a small cost to [latency](/shared/glossary/#latency). Production servers like [Triton Inference Server](/shared/glossary/#triton-inference-server) perform batching automatically.

### BC {#bc}
Behavior Cloning — supervised imitation of demonstrator actions

### Behavior policy {#behavior-policy}
The policy that generated the data, in off-policy or offline RL

### Bellman equation {#bellman-equation}
The recursive consistency condition `V(s) = E[r + γV(s')]`

### Benchmark {#benchmark}
A fixed, shared test set used to measure and compare models on a task — like a standardized exam everyone sits so scores line up side by side. [MMLU](/shared/glossary/#mmlu) tests knowledge and [GSM8K](/shared/glossary/#gsm8k) tests math; a benchmark is only meaningful while models have not already seen its answers (see [contamination](/shared/glossary/#contamination)).

### Best-of-N {#best-of-n}
An inference trick that samples `N` candidate answers to the same prompt and keeps the single one a scorer — usually a [reward model](/shared/glossary/#reward-model) or [verifier](/shared/glossary/#verifier) — rates highest, like writing several drafts of an email and sending only the best.

### bfloat16 {#bfloat16}
16-bit float with fp32's exponent range — the modern default for training (also written bf16, BF16)

### Bias correction {#bias-correction}
An adjustment applied in the Adam family of optimizers to counteract the zero-initialization of moment estimates; without it, early steps would be artificially small

### Biases {#biases}
The additive [parameter](/shared/glossary/#parameters) vectors in a linear layer (the `b` in `y = xW + b`). Each output neuron has one bias value, which shifts the result independently of the input.

### Blackwell {#blackwell}
NVIDIA's 2024 GPU architecture (B100, B200, B200 Ultra) and the successor to [Hopper](/shared/glossary/#hopper). Like swapping a sports car engine for a more powerful one of the same shape, it keeps the same overall design as Hopper but doubles down on low-precision math — better [FP8](/shared/glossary/#fp8) throughput and brand-new FP4 [Tensor Cores](/shared/glossary/#tensor-core) — which is what makes it the preferred chip for the largest 2025-era training and serving runs.

### BM25 (Best Matching 25) {#bm25}
A classic keyword-search ranking — short for **Best Matching 25** — that scores a document by how often the query's words appear in it, weighting rare words more heavily. Think of it like a librarian scanning pages for your exact search words and ranking pages where those words appear most often (especially unusual words) higher on the list. It is the *sparse* (exact-word) counterpart to dense [embedding](/shared/glossary/#embedding) search.

### Bootstrapping {#bootstrapping}
Using a current estimate (e.g., `V(s')`) in the target instead of a full return

### Bottleneck {#bottleneck}
The single slowest stage in a pipeline, which caps the overall speed; in training this is often the data loader rather than the model.

### bpd (bits per dimension) {#bpd-bits-per-dimension}
Standard likelihood metric for image models; `-log₂ p(x) / D`

### BPE {#bpe}
Byte-Pair Encoding — subword [tokenization](/shared/glossary/#tokenizer) by greedy frequent-pair merges. It starts from raw bytes and repeatedly glues together the neighboring pair that appears most often, building up reusable chunks. For example, on lots of English text BPE notices `t` and `h` sit side by side constantly and merges them into `th`; a later round merges `th` + `e` into `the`. So a common word like `the` ends up as a single token, while a rarer word like `tokenizer` is left as familiar pieces such as `token` + `izer`. "Greedy" means each round simply takes the single most-frequent merge available, never looking ahead to see whether a different choice would pay off later.

### C++ extension {#c-extension}
A custom operation written in C++ (optionally with CUDA), compiled and loaded so it can be called from Python like a built-in PyTorch op.

### C-space {#c-space}
Configuration space — the abstract space of joint configurations

### c10 {#c10}
PyTorch's core C++ library (the "core ten[sor]" library)

### Calibration {#calibration}
Running a few representative batches of data through a model to measure the typical range of its [activations](/shared/glossary/#activations), so that static [quantization](/shared/glossary/#quantization) can pick fixed [int8](/shared/glossary/#int8) scales.

### Catastrophic forgetting {#catastrophic-forgetting}
When training a model on new data erases skills it had already learned, because the new [gradients](/shared/glossary/#gradients) overwrite the old [weights](/shared/glossary/#weights).

### Causal mask {#causal-mask}
A mask applied to attention scores that hides future positions, so each token can attend only to itself and the tokens before it

### CBF {#cbf}
Control Barrier Function — runtime safety filter via a constraint on `ḣ`

### CDNA / RDNA {#cdna--rdna}
AMD's datacenter / consumer GPU architectures

### CFG (classifier-free guidance) {#cfg-classifier-free-guidance}
Inference trick: combine conditional and unconditional model outputs to amplify conditioning

### CFG fusion {#cfg-fusion}
A diffusion-serving optimization for [classifier-free guidance](/shared/glossary/#cfg-classifier-free-guidance), which normally needs *two* model passes per denoising step — one conditioned on the prompt, one unconditioned. CFG fusion runs both in a single batched forward pass (stacking them as a batch of two) instead of two separate calls, so the GPU is launched once per step rather than twice. Like cooking two portions in one pan instead of washing up between them — same result, far less overhead.

### Chain rule {#chain-rule}
A calculus principle used to compute the derivative of a composite function by multiplying the derivatives of its parts.

### Chat template {#chat-template}
The structured format (system/user/assistant) the model is fine-tuned on

### Checkpoint {#checkpoint}
A saved snapshot of a model's [weights](/shared/glossary/#weights) (and [optimizer state](/shared/glossary/#optimizer-state)) at a point in training, so a run can be resumed or rolled back to it after a failure.

### Chinchilla {#chinchilla}
The scaling law showing compute-optimal training uses ~20 tokens per parameter

### Chunked prefill {#chunked-prefill}
Splitting long prompts across multiple iterations to interleave with decode steps

### Chunking {#chunking}
Splitting documents into smaller passages (often a few hundred tokens each) before indexing them for retrieval, so a search returns a focused snippet instead of a whole book.

### CLIP {#clip}
Contrastive Language-Image Pretraining — paired text-image dual encoder

### Closed-form {#closed-form}
A solution you can write down and compute directly with a fixed formula, instead of reaching it through many rounds of trial-and-error. Solving `2x = 10` by writing `x = 5` is closed-form; nudging `x` up and down until both sides match is not. In [DPO](/shared/glossary/#dpo) a closed-form objective lets the model learn straight from preference pairs with one training [loss](/shared/glossary/#loss-function), skipping the slow [reward model](/shared/glossary/#reward-model)-plus-[PPO](/shared/glossary/#ppo) loop of classic [RLHF](/shared/glossary/#rlhf).

### CNN {#cnn}
Convolutional Neural Network — a neural network built mainly from convolution layers; the standard architecture for image tasks.

### Collate function {#collate-function}
The function a [DataLoader](/shared/glossary/#dataloader) uses to combine a list of individual samples into one batched tensor; a custom one can pad variable-length data.

### Collective operation {#collective-operation}
A communication step that all processes ([ranks](/shared/glossary/#rank)) in a distributed job perform together — such as [AllReduce](/shared/glossary/#allreduce); if one rank skips it, the others wait forever.

### Collision mesh {#collision-mesh}
Simplified geometry used for collision tests, distinct from visual mesh

### Column-wise partitioning {#column-wise-partitioning}
Splitting a weight matrix along its column (output) dimension so that each GPU holds a vertical slice and computes part of the output independently — the standard first step in [Megatron](/shared/glossary/#megatron)-style [tensor parallelism](/shared/glossary/#tensor-parallelism-tp).

### Consistency model {#consistency-model}
A diffusion-derived model that samples in 1–4 steps via consistency distillation

### Constitutional AI {#constitutional-ai}
An alignment recipe (introduced by Anthropic) where some or all human preference labels are replaced by an AI judge that grades responses against a written "constitution" — a short list of principles like "be helpful," "refuse to assist with harm," "don't pretend to be human." Like running a debate club with a published rulebook instead of asking the audience to vote: cheaper, more consistent, and easier to update than collecting fresh human labels for every new behavior. The technique is the foundation of [RLAIF](/shared/glossary/#rlaif).

### Constrained generation {#constrained-generation}
A decoding-time technique that masks out any next-token choices that would break a target structure — a regex, a JSON schema, a grammar — so the model is only *allowed* to pick valid continuations. Like a Mad Libs game whose blanks accept only nouns or only numbers: the writer can be creative inside each blank but cannot break the form. Libraries such as Outlines and sglang are common implementations, and the technique is what makes reliable [function calling](/shared/glossary/#function-calling) and tool-using agents possible.

### Contamination {#contamination}
When items from an evaluation [benchmark](/shared/glossary/#benchmark) accidentally end up in a model's training data, so its score reflects memorization rather than skill — like a student who studied from a leaked copy of the exam. Also called train-test contamination, it is a leading reason a high benchmark number can mislead.

### Content-addressable token mixing {#content-addressable-token-mixing}
The routing and retrieval of information between tokens based on their query-key similarity (as in [attention](/shared/glossary/#attention)) rather than their positions

### Context window {#context-window}
The maximum number of tokens the model can attend over in one forward pass

### Continued pretraining {#continued-pretraining}
Taking an already-pretrained model and training it further on a new corpus to add domain knowledge, rather than starting from random weights.

### Continuous batching {#continuous-batching}
A serving trick where the GPU adds new requests into the running [batch](/shared/glossary/#batching) — and drops finished ones — at every decode step, instead of waiting for the whole batch to finish together. Like a hotel shuttle that can pick up and drop off passengers anywhere along its loop rather than only at the start and end: far fewer empty seats overall, so [throughput](/shared/glossary/#throughput) goes up dramatically. It is the single largest speedup in modern LLM serving and is the default in [vLLM](/shared/glossary/#vllm) and [TGI](/shared/glossary/#tgi).

### ControlNet {#controlnet}
Architecture that adds an auxiliary conditioning branch (depth, pose, edges, …) to a frozen diffusion model

### copy {#copy}
A tensor that owns its own storage, independent of any source tensor; created by `.clone()`, or automatically by operations like `.contiguous()` and `reshape` when a view is not possible

### Cosine decay {#cosine-decay}
A [learning-rate](/shared/glossary/#learning-rate) schedule that, after [warmup](/shared/glossary/#warmup), lowers the rate along the smooth downward half of a cosine curve until it reaches near zero by the end of training. The step size starts large and eases off gently — like braking smoothly as you coast up to a stop sign instead of slamming the pedal at the last moment — which helps the model settle into a good solution. It is the long-standing default schedule, before newer recipes like [WSD](/shared/glossary/#wsd).

### CoT {#cot}
Chain of Thought — prompting or training a model to write out its reasoning step by step before giving a final answer, the way a student shows their work on a math problem instead of blurting out just the result.

### CQL {#cql}
Conservative Q-Learning — offline RL with a pessimistic Q penalty

### Cross-attention {#cross-attention}
Attention where queries come from one modality and keys/values from another

### Cross-encoder {#cross-encoder}
A model that reads a query and one candidate document *together* in a single pass and outputs one relevance score — far more accurate than comparing their separate [embeddings](/shared/glossary/#embedding), but too slow to run over a whole corpus, so it is used to [rerank](/shared/glossary/#reranker) a short candidate list.

### Cross-entropy {#cross-entropy}
A [loss function](/shared/glossary/#loss-function) that scores how surprised a model is by the correct answer: it stays small when the model gave the true next word a high probability and grows large when it was confidently wrong. Like grading a weather forecaster on confidence and not just on being right — announcing "90% chance of sun" and then getting rain costs far more points than a hedged "50%." Training an [LLM](/shared/glossary/#llm) means adjusting the [weights](/shared/glossary/#weights) to push this surprise as low as it will go.

### cuBLAS {#cublas}
NVIDIA's optimized library of dense linear-algebra [kernels](/shared/glossary/#kernel); PyTorch calls it for matrix multiplication on [CUDA](/shared/glossary/#cuda).

### CUDA {#cuda}
NVIDIA's GPU compute backend; tensors on the `cuda` device run their kernels here

### Custom op {#custom-op}
A user-defined operation registered with PyTorch (e.g. via `torch.library.custom_op`) so it behaves like a built-in — including working with [`torch.compile`](/shared/glossary/#torchcompile).

### CUTLASS {#cutlass}
NVIDIA's open template library for matmul kernels

### Data parallelism {#data-parallelism}
The default way to train across many GPUs: put a full copy of the model on each GPU, feed each one a different slice of the batch, then average their [gradients](/shared/glossary/#gradients) so all copies stay identical — like several graders each marking part of an exam pile and then pooling the scores. (See [DDP](/shared/glossary/#ddp).)

### DataLoader {#dataloader}
PyTorch's iterator that pulls samples from a Dataset, groups them into batches, and can load them in parallel using [worker processes](/shared/glossary/#worker-processes).

### DDIM {#ddim}
Deterministic, accelerated sampler for diffusion models

### DDP {#ddp}
Distributed Data Parallel — replicate model, split batch, all-reduce gradients

### DDPG {#ddpg}
Deep Deterministic Policy Gradient — the first deep-RL continuous-control algorithm

### DDPM {#ddpm}
Denoising Diffusion Probabilistic Models — the foundational 2020 paper and training recipe

### Deadly triad {#deadly-triad}
Function approximation + bootstrapping + off-policy data → instability

### Decode {#decode}
The token-by-token half of LLM inference: after [prefill](/shared/glossary/#prefill) digests the prompt, the model generates one new token per forward pass, each step reading the whole [KV cache](/shared/glossary/#kv-cache) before producing the next [logits](/shared/glossary/#logits). Like writing a sentence one word at a time while glancing back over every word already written — fast per step, but the constant re-reading of the page is what bounds speed. Decode is [memory-bandwidth-bound](/shared/glossary/#roofline) on a GPU, the opposite of [prefill](/shared/glossary/#prefill), and is what most serving optimizations target.

### Decoupled {#decoupled}
A training technique where two effects that are mathematically equivalent in standard SGD are separated into independent operations. In AdamW, weight decay is decoupled from the gradient update so that the regularization strength is not scaled by the adaptive learning rate.

### Deduplication {#deduplication}
Removing repeated or near-repeated documents from a training corpus; one of the highest-return cleaning steps in [pretraining](/shared/glossary/#pretraining).

### DeepSpeed {#deepspeed}
Microsoft's open-source library for training very large models efficiently. It is best known for [ZeRO](/shared/glossary/#zero), which shards a model's [parameters](/shared/glossary/#parameters), [gradients](/shared/glossary/#gradients), and [optimizer state](/shared/glossary/#optimizer-state) across GPUs so no single GPU has to hold the whole model — the same idea as PyTorch's [FSDP](/shared/glossary/#fsdp). Think of it as a moving company that splits one giant load across several trucks instead of trying to cram everything into one.

### Detached tensor {#detached-tensor}
A tensor that has been removed from the [dynamic computation graph](/shared/glossary/#dynamic-computation-graph) via the `.detach()` method, meaning operations performed on it will not be tracked for [autograd](/shared/glossary/#autograd).

### Derivative {#derivative}
The instantaneous rate of change of a function with respect to its input. In deep learning, derivatives are computed via the chain rule during backpropagation to produce gradients used to update model parameters.

### Deterministic algorithms {#deterministic-algorithms}
Operations that produce bit-identical outputs for identical inputs every time; enabled in PyTorch via `torch.use_deterministic_algorithms(True)` at the cost of some performance

### Detokenization {#detokenization}
Turning a sequence of token IDs back into a UTF-8 string — the reverse of what the [tokenizer](/shared/glossary/#tokenizer) did on the way in. The tricky part for streaming servers is that a single visible character (like an emoji or a Chinese character) is often spread across several [BPE](/shared/glossary/#bpe) pieces, so emitting each token's text the moment it arrives can produce broken bytes; a correct streaming detokenizer buffers the partial bytes until they form a complete character.

### DH parameters {#dh-parameters}
Denavit-Hartenberg parameters — textbook arm-geometry description

### Diffusion model {#diffusion-model}
A generative model that learns to *un-noise* an image (or video, or audio) — training starts from clean data, gradually adds Gaussian noise until it looks like static, and teaches the network to reverse one small step of that corruption. At inference time you start from pure static and call the network many times (often 4–50), each call removing a bit of noise until a coherent picture emerges. Like sculpting in reverse: the marble starts as a featureless block of noise and the model chips away until the shape appears. Stable Diffusion is the best-known example.

### Disaggregated serving {#disaggregated-serving}
Running prefill and decode on separate GPU pools with KV cache transfer between them

### Dispatcher {#dispatcher}
The PyTorch component that routes `torch.foo(...)` calls to the right backend/dtype kernel

### Distillation {#distillation}
Training a smaller "student" model to copy the output of a larger, more capable "teacher" so the student inherits most of the teacher's behavior at a fraction of the cost. Like a junior cook shadowing a head chef and learning each recipe by mimicking the dish — they may never match the master, but they can plate most of the menu for far less money. Distillation works for skills the teacher already has but cannot conjure new abilities the teacher lacks.

### DiT {#dit}
Diffusion Transformer — Peebles & Xie's transformer-based diffusion backbone

### Double backward {#double-backward}
Computing the gradient of a gradient by tracking the backward pass operations in a new computation graph.

### Downstream {#downstream}
The later, real-world tasks a model is eventually judged on — such as question answering or coding — as opposed to the [pretraining](/shared/glossary/#pretraining) objective it was trained on. "Downstream scores" measure how much a change (cleaner data, a better [learning rate](/shared/glossary/#learning-rate)) actually pays off on those end tasks, the way a river's health downstream reflects what happened upstream at the source.

### DPO {#dpo}
Direct Preference Optimization — [closed-form](/shared/glossary/#closed-form) RLHF without a reward model or [PPO](/shared/glossary/#ppo)

### DQN {#dqn}
Deep Q-Network — Q-learning with neural-net function approximation + experience replay + target network

### DreamBooth {#dreambooth}
Fine-tuning recipe for subject personalization; updates the whole model on a few subject images

### dtype {#dtype}
A tensor's element data type — e.g. `float32`, `float16`, `bfloat16`, `int8`, `bool`

### Dynamic computation graph {#dynamic-computation-graph}
A graph of operations built on-the-fly as code executes, representing the forward pass used for autograd.

### Dynamic quantization {#dynamic-quantization}
A [quantization](/shared/glossary/#quantization) method that stores [weights](/shared/glossary/#weights) as [int8](/shared/glossary/#int8) ahead of time but computes each layer's [activation](/shared/glossary/#activations) scale at runtime, just before the layer runs.

### Eager mode {#eager-mode}
PyTorch's default execution, where each operation runs immediately as its Python line is reached — flexible and easy to debug, but without the cross-operation optimizations a compiler can apply.

### EAGLE / Medusa {#eagle--medusa}
Self-speculation: extra heads on the target model propose tokens, no separate draft model

### EDM {#edm}
Karras et al. 2022 — a reformulation of diffusion in σ-space with clean preconditioning

### EKF {#ekf}
Extended Kalman Filter — Kalman filter linearized about the current estimate

### ELBO {#elbo}
Evidence Lower Bound — the variational objective trained by VAEs

### Elementwise operation {#elementwise-operation}
An operation applied independently to each element of a tensor (e.g. add, multiply, ReLU), where output position `i` depends only on input position `i`.

### Elo {#elo}
A rating system borrowed from chess that turns a series of head-to-head wins and losses into a single number per player: beat a strong opponent and your rating jumps, lose to a weak one and it drops. LLM [arenas](/shared/glossary/#arena) use it to rank chat models from pairwise comparisons instead of from a fixed-answer [benchmark](/shared/glossary/#benchmark).

### EMA weights {#ema-weights}
Exponential moving average of model weights; samples better than the live weights

### Embedding {#embedding}
A dense vector that represents a token (or other item) so the model can compute over it; each token ID maps to one row of the [embedding matrix](/shared/glossary/#embedding-matrix)

### Embedding matrix {#embedding-matrix}
The lookup table `E ∈ ℝ^{V×d}` that turns each token ID into a dense vector by selecting its row; growing the [vocabulary](/shared/glossary/#vocabulary) means adding rows

### ExecuTorch {#executorch}
PyTorch's lightweight runtime for running models on mobile and edge devices, built on the graph captured by [`torch.export`](/shared/glossary/#torchexport).

### Expert {#expert}
In a [Mixture-of-Experts (MoE)](/shared/glossary/#moe), one of several parallel [MLP](/shared/glossary/#mlp) sub-networks; a router sends each token to only the top few experts instead of all of them. Like a hospital triage desk that routes each patient to the right specialist rather than making everyone see every doctor — lots of expertise on hand, but only a little used per case.

### Expert parallelism (EP) {#expert-parallelism-ep}
For MoE models, distributing experts across GPUs with all-to-all token routing

### Exponent {#exponent}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that records its *scale* — how many places to shift the decimal point. In scientific notation like `3.5 × 10¹²`, the `12` is the exponent (using base 10 instead of base 2). More exponent bits give a wider range of representable magnitudes, from astronomically large to vanishingly small; fewer exponent bits mean values overflow or [underflow](/shared/glossary/#underflow) more easily. This is why [FP8](/shared/glossary/#fp8) has two flavors: **E5M2** (5 exponent bits) for gradients that can swing wildly in size, and **E4M3** (4 exponent bits) for activations that stay in a tighter range. See also [mantissa](/shared/glossary/#mantissa).

### FFN {#ffn}
Feed-Forward Network — a position-wise neural network block (often referred to as an [MLP](/shared/glossary/#mlp)) in a [transformer](/shared/glossary/#transformer) that processes each token independently.

### F/T sensor {#ft-sensor}
Force/Torque sensor — six-axis force and moment at a wrist or fingertip

### FID {#fid}
Fréchet Inception Distance — the standard sample-quality metric for image generation

### FineWeb-Edu {#fineweb-edu}
A large, openly released [pretraining](/shared/glossary/#pretraining) dataset built by running a [quality filter](/shared/glossary/#quality-filter) over crawled web pages and keeping only the educational-looking ones — like skimming a huge pile of internet text and saving just the pages that read like a textbook. Models trained on it often beat models trained on far more unfiltered text, making it a go-to example that data quality can matter more than raw quantity.

### FK / IK {#fk--ik}
Forward / Inverse Kinematics — compute end-effector pose from joints or vice versa

### FlashAttention {#flashattention}
IO-aware attention kernel that avoids materializing the T×T score matrix in HBM

### float16 {#float16}
16-bit floating-point format (`fp16`); saves memory and can be fast on GPUs, but has a limited range (max ~65,504) that can cause [underflow](/shared/glossary/#underflow) when accumulating very small values

### float32 {#float32}
32-bit floating-point format (`fp32`); the standard default precision for PyTorch tensors — wide enough range and enough precision for most training and inference tasks

### FLOPs {#flops}
Floating-Point Operations — a measure of computational complexity representing the number of individual arithmetic operations (additions or multiplications) performed.

### Flow matching {#flow-matching}
Training a velocity field that transports noise to data via an ODE; modern alternative to DDPM

### Forensics {#forensics}
Working backward from a training failure to the operation that first caused it, instead of chasing the visible symptom. In PyTorch this means turning on [autograd](/shared/glossary/#autograd) [anomaly detection](/shared/glossary/#anomaly-detection) to halt at the first [NaN](/shared/glossary/#nan) or bad [gradient](/shared/glossary/#gradients).

### Forward hook {#forward-hook}
A callback registered on an `nn.Module` that PyTorch calls automatically after the module's forward pass, receiving the input and output tensors; used for capturing activations and debugging

### FP8 {#fp8}
8-bit floating point — half the bits of [bfloat16](/shared/glossary/#bfloat16). Comes in two flavors: **E4M3** (4 [exponent](/shared/glossary/#exponent) bits + 3 [mantissa](/shared/glossary/#mantissa) bits) keeps a bit more precision and is used for [weights](/shared/glossary/#weights) and the forward [activations](/shared/glossary/#activations); **E5M2** (5 [exponent](/shared/glossary/#exponent) + 2 [mantissa](/shared/glossary/#mantissa)) trades precision for a wider range and is used for gradients, which can be very large or very small. Supported by [Hopper](/shared/glossary/#hopper) and later NVIDIA GPUs, it is rapidly becoming the modern default serving precision.

### Frontier run {#frontier-run}
A training run for one of the largest, most capable models at the leading edge of what is currently possible — the kind that ties up thousands of GPUs for weeks and costs millions of dollars. Because the stakes are so high, a [loss spike](/shared/glossary/#loss-spike) that cannot be recovered cleanly can throw away days of that compute, which is why teams rehearse [checkpoint](/shared/glossary/#checkpoint) recovery on small models first.

### F.scaled_dot_product_attention {#fscaled_dot_product_attention}
PyTorch's built-in fused [attention](/shared/glossary/#attention) function (in `torch.nn.functional`) that computes `softmax(QKᵀ/√d)·V` in a single call, dispatching to an optimized [backend](/shared/glossary/#backend) such as [FlashAttention](/shared/glossary/#flashattention).

### FSDP {#fsdp}
Fully Sharded Data Parallel — shard params, grads, and optimizer state across [ranks](/shared/glossary/#rank)

### FSQ {#fsq}
Finite Scalar Quantization — codebook-free discrete tokenization

### Function calling {#function-calling}
The mechanism by which a model uses a tool: it emits a structured request (such as JSON naming a function and its arguments), an external program runs that request, and the result is handed back to the model. Also called tool use.

### Fusion (early/middle/late) {#fusion-earlymiddlelate}
When in the network different modalities are combined

### FVD {#fvd}
Fréchet Video Distance — the standard (and flawed) automatic eval metric for video generation

### GAE {#gae}
Generalized Advantage Estimation — TD(λ) for advantages

### GANs (Generative Adversarial Networks) {#gans}
A class of generative models in which a generator network and a discriminator network are trained adversarially. The generator learns to produce realistic samples to fool the discriminator, which learns to distinguish real from generated data.

### Gated {#gated}
An operation where one path of a neural network modulates the information flow in another path via element-wise multiplication (e.g. in a [SwiGLU](/shared/glossary/#swiglu) block).

### GCG {#gcg}
Short for **Greedy Coordinate Gradient** — a gradient-based attack that finds an adversarial suffix (a short string of seemingly random tokens) which, when appended to a harmful question, causes an aligned [LLM](/shared/glossary/#llm) to comply anyway. It works by swapping one suffix token at a time for whatever the gradient says raises the probability of an unsafe answer most. Like picking a combination lock by feeling each dial until the click; once one model is unlocked the same suffix often opens other models, which is why GCG is the standard benchmark attack in [jailbreak](/shared/glossary/#jailbreak) research.

### GELU {#gelu}
Gaussian Error Linear Unit — a smooth activation function widely used in transformer [MLPs](/shared/glossary/#mlp).

### GLU {#glu}
Gated Linear Unit — a layer whose output is the element-wise product of two linear projections, one of them passed through a [gating](/shared/glossary/#gated) non-linearity; [SwiGLU](/shared/glossary/#swiglu) is the variant that uses [Swish](/shared/glossary/#swish) as that non-linearity.

### GPTQ {#gptq}
Short for **Generative Pre-trained Transformer Quantization** — a [post-training quantization (PTQ)](/shared/glossary/#ptq--qat) method that compresses each layer's [weights](/shared/glossary/#weights) row by row, using second-order ([Hessian](/shared/glossary/#hessian)) information to choose the int8 / int4 values that minimize the reconstruction error one layer at a time. Despite the name, GPTQ is not GPT-specific; it works on any [transformer](/shared/glossary/#transformer).

### GQA {#gqa}
Grouped-Query Attention — sharing K/V heads across query heads; primary KV-cache saver at serving time

### Gradient accumulation {#gradient-accumulation}
Summing the [gradients](/shared/glossary/#gradients) from several small batches before calling the [optimizer](/shared/glossary/#optimizer), so the update matches a larger effective batch size without its memory cost.

### Gradients {#gradients}
The vector of [partial derivatives](/shared/glossary/#partial-derivative) of a function with respect to its inputs. In neural networks, gradients represent the direction and magnitude of the change required to minimize the [loss function](/shared/glossary/#loss-function).

### Gradient checkpointing {#gradient-checkpointing}
A memory-saving technique that discards intermediate activations during the forward pass and recomputes them during the backward pass.

### GradScaler {#gradscaler}
A helper used with [float16](/shared/glossary/#float16) mixed-precision training that multiplies the loss before the backward pass, preventing small gradients from rounding to zero ([underflow](/shared/glossary/#underflow)).

### Graph break {#graph-break}
A point where [`torch.compile`](/shared/glossary/#torchcompile) cannot trace the code (e.g. a `print` or a data-dependent branch), forcing it to split the model and fall back to [eager mode](/shared/glossary/#eager-mode) — a common cause of lost speedup.

### Greedy decoding {#greedy-decoding}
The simplest [sampling](/shared/glossary/#sampling) rule: at every step, pick the single most likely next token (the [`argmax`](/shared/glossary/#argmax) of the [logits](/shared/glossary/#logits)) and never roll the dice. Like always ordering the most popular dish on the menu — boring but predictable. Useful when reproducibility matters, though on a GPU even greedy decoding is not bit-for-bit deterministic across batch sizes because floating-point sums reorder.

### Grounding {#grounding}
Producing spatial outputs (boxes, points) referring to image regions

### GRPO {#grpo}
Group Relative Policy Optimization — [value-function](/shared/glossary/#value-function)-free PPO variant; DeepSeek lineage

### GSM8K {#gsm8k}
A benchmark of about 8,000 grade-school math word problems, widely used to test step-by-step reasoning because each problem has a single checkable numeric answer.

### GTSAM {#gtsam}
Factor-graph SLAM library; the standard back-end for many modern systems

### Half-rotation {#half-rotation}
An efficient way to apply [RoPE](/shared/glossary/#rope): rather than rotating each adjacent pair of vector components on its own, you split the vector into two halves and combine them in one shot (the `rotate_half` trick, `[x₁, x₂] → [−x₂, x₁]`). It turns many tiny 2-D rotations into a couple of whole-vector operations, so it runs fast on a GPU while giving the same result.

### Hallucination {#hallucination}
When an [LLM](/shared/glossary/#llm) states something false with the same confident tone it uses for true things — invented citations, made-up people, fabricated facts. Like a student who didn't read the book but answers the essay question anyway in confident prose; the grammar is fine, the facts are not. Hallucination is built in to the [next-token prediction](/shared/glossary/#next-token-prediction) objective, which rewards fluent continuation rather than truth, and is mitigated (not solved) by [RAG](/shared/glossary/#rag), [verifiers](/shared/glossary/#verifier), and abstention training.

### HBM {#hbm}
High-Bandwidth Memory — stacked DRAM on a modern GPU; usually the bandwidth bottleneck

### Headroom {#headroom}
The safety margin you have left before something breaks. In low-precision training it is the spare range of values a number format can still represent before it overflows or rounds down to zero and triggers [numerical issues](/shared/glossary/#numerical-issues) — like the gap between your head and the ceiling: the less you have, the easier it is to bump into trouble. [FP8](/shared/glossary/#fp8) packs numbers into far fewer bits than [bfloat16](/shared/glossary/#bfloat16), so it has much less headroom and is more likely to destabilize a run.

### Heads (attention) {#heads}
The independent, parallel [attention](/shared/glossary/#attention) sub-computations in multi-head attention. Each head operates on its own learned projections of queries, keys, and values, allowing the model to attend to different representation subspaces simultaneously.

### Hessian {#hessian}
The matrix of all second partial derivatives of a function — it captures the *curvature* of a loss landscape, not just its slope. Where the [gradient](/shared/glossary/#gradients) tells you "which way is downhill," the Hessian tells you "and how sharply does it bend." Like the difference between knowing a road slopes down and knowing whether it banks into a tight curve or stretches out almost flat. For real LLMs the full Hessian is too big to store (rows × columns each equal to the parameter count), so methods like [GPTQ](/shared/glossary/#gptq) use cheap approximations of it — typically built from a small [batch](/shared/glossary/#batch) of [calibration](/shared/glossary/#calibration) activations — to decide which weights matter most when [quantizing](/shared/glossary/#quantization).

### Holonomic {#holonomic}
A vehicle whose instantaneous motion can be any direction (mecanum, omni)

### Hopper {#hopper}
NVIDIA's 2022 GPU architecture (H100, H200) and the workhorse of LLM training and serving in 2023–2024. It was the first generation to ship dedicated [FP8](/shared/glossary/#fp8) [Tensor Cores](/shared/glossary/#tensor-core), which is what made FP8 inference a practical option. Named after Grace Hopper, the computer scientist who invented the compiler.

### Hybrid retrieval {#hybrid-retrieval}
Retrieving with both dense [embedding](/shared/glossary/#embedding) search (matches meaning) and sparse keyword search ([BM25](/shared/glossary/#bm25)) (matches exact words) and merging the two result lists, so each method covers the other's weaknesses.

### I2V {#i2v}
Image-to-Video

### Identity function {#identity-function}
A function that returns its input unchanged: `f(x) = x`. In the context of straight-through estimators, gradients are passed through a non-differentiable operation as if it were the identity function.

### Impedance control {#impedance-control}
Command a virtual spring-damper between end-effector and reference

### IMU {#imu}
Inertial Measurement Unit — gyroscope + accelerometer (often + magnetometer)

### Indexing {#indexing}
Mapping a multidimensional index `[i, j, …]` to a flat storage position via `offset + Σ iₖ·strideₖ`

### Inference-time compute {#inference-time-compute}
The work a model does while answering a question (not while training) — for reasoning models, mostly the tokens it spends "thinking" before it replies. Giving a fixed model more inference-time compute, like giving a student more time on an exam, can raise its accuracy without changing the model at all.

### InfiniBand (IB) {#infiniband-ib}
High-speed network with RDMA; standard for AI clusters

### InfoNCE {#infonce}
The contrastive loss used by CLIP; softmax over a similarity matrix

### int8 {#int8}
8-bit integer format; storing [weights](/shared/glossary/#weights) or [activations](/shared/glossary/#activations) as int8 uses a quarter of the memory of [float32](/shared/glossary/#float32) and can run faster, at some cost in precision.

### IQL {#iql}
Implicit Q-Learning — offline RL that never queries `Q` at OOD actions

### Isaac Lab {#isaac-lab}
NVIDIA GPU-parallel robotics simulation platform

### iso {#iso}
A prefix meaning "equal" or "the same" (from the Greek *isos*). In a phrase like *iso-[FLOP](/shared/glossary/#flops)* it marks a group of training runs that all spent the same compute budget, so they can be compared fairly — like rating cars by how far each travels on the *same* tank of fuel rather than on top speed. Plotting the [loss](/shared/glossary/#loss-function) of several iso-FLOP runs is how a [Chinchilla](/shared/glossary/#chinchilla)-style [scaling-law](/shared/glossary/#scaling-laws) curve is drawn.

### ITL / TPOT {#itl--tpot}
Inter-token latency / time per output token — steady-state per-token decode time

### Jacobian {#jacobian}
Linear map from joint velocities to end-effector spatial velocity

### Jailbreak {#jailbreak}
A prompt — sometimes plain English, sometimes a gradient-found suffix like in [GCG](/shared/glossary/#gcg), sometimes a long role-play setup or a translation into a low-resource language — that gets a safety-trained model to do what its alignment training was supposed to refuse. Like picking a hotel-room door lock instead of asking for the key. Modern defenses assume any single safety layer can be jailbroken and use *defense in depth* — input filtering, output filtering, monitoring, refusal classifiers — instead of trusting the model alone.

### Kernel {#kernel}
A single function that runs on the GPU (or CPU) to carry out one operation, such as a matrix multiply or an element-wise add.

### Kernel fusion {#kernel-fusion}
Combining several small operations into one [kernel](/shared/glossary/#kernel) so the hardware reads and writes memory fewer times and pays fewer launch costs.

### KF {#kf}
Kalman Filter — optimal linear-Gaussian Bayes filter

### KL divergence {#kl-divergence}
Short for **Kullback-Leibler divergence** — a number that measures how far one probability distribution has drifted from another, growing larger the more the two disagree. In [RLHF](/shared/glossary/#rlhf) it acts as a leash on the [policy](/shared/glossary/#policy) being trained: the further its word probabilities wander from the frozen [reference model](/shared/glossary/#reference-model), the bigger the penalty it pays. Like a tether that lets a climber explore but stops them straying somewhere dangerous, it lets the model chase reward without forgetting how to talk sensibly.

### KV cache {#kv-cache}
A [scratchpad](/shared/glossary/#scratchpad) that stores the [attention](/shared/glossary/#attention) keys and values already computed for every earlier token in the sequence, so generating the next token only has to compute keys and values for that *one* new token instead of redoing all the previous ones. Like writing out a long multiplication table once and then looking up products instead of recalculating them — it turns each decode step from "redo the whole prompt" into "do one more token," which is what makes long-context serving fast enough to be usable.

### L2 regularization {#l2-regularization}
A regularization technique that adds a penalty proportional to the squared magnitude of model weights to the loss function, encouraging smaller weights and reducing overfitting. In standard adaptive optimizers such as [Adam](/shared/glossary/#adam), this penalty is folded into the gradient and scaled by the adaptive learning rate, which is why [AdamW](/shared/glossary/#adamw) uses [decoupled](/shared/glossary/#decoupled) weight decay instead.

### Latency {#latency}
The time it takes to complete a single request, from input to output; distinct from [throughput](/shared/glossary/#throughput), which counts how many requests finish per second.

### Latent video {#latent-video}
Compressed (T', H', W', C) tensor produced by a 3D VAE

### LCM {#lcm}
Latent Consistency Model — consistency-distilled few-step latent diffusion

### LDM {#ldm}
Latent Diffusion Model — diffusion in the latent space of a VAE (i.e., Stable Diffusion)

### Learning rate {#learning-rate}
The step size an [optimizer](/shared/glossary/#optimizer) takes when nudging the [weights](/shared/glossary/#weights) along the [gradient](/shared/glossary/#gradients). Too large and training overshoots and diverges; too small and it crawls — like choosing how big a step to take walking downhill in fog. It is usually ramped up during [warmup](/shared/glossary/#warmup) and then decayed over the run.

### LiDAR {#lidar}
Light Detection And Ranging — laser range scanner

### Linear probe {#linear-probe}
A small linear classifier trained on the frozen hidden [activations](/shared/glossary/#activations) of a layer of a neural network to test whether that layer has *already* encoded some property — for example, "is this sentence true?", "what is the capital of this country?", or "which language is this?" Like sticking a voltmeter into one wire of a circuit to see what signal is flowing past that point; you don't change the circuit, you just read what's already there. The standard first tool in [mechanistic interpretability](/shared/glossary/#mechanistic-interpretability).

### LLM {#llm}
Large Language Model — a [transformer](/shared/glossary/#transformer) trained on large amounts of text to predict and generate language.

### LLM-as-judge {#llm-as-judge}
Using a strong [LLM](/shared/glossary/#llm) to grade or compare other models' answers in place of a human rater — fast, cheap, and surprisingly well-calibrated, though it tends to favor longer answers and ones written in its own style. To catch [position bias](/shared/glossary/#position-bias) you usually ask twice with the two answers swapped and trust only an agreeing verdict — like a blind wine tasting where the same two bottles are poured first as "Glass A, Glass B" and then again as "Glass B, Glass A"; you only believe the judge picked the better wine if they pick the same bottle both times, because that rules out them simply liking whichever glass sat on the left.

### Logits {#logits}
The raw, unnormalized scores a model produces at its output, one per [vocabulary](/shared/glossary/#vocabulary) entry, before they are turned into probabilities by [softmax](/shared/glossary/#softmax). Like the points each contestant has scored at the end of a game — bigger means "more likely the next token" — but to read them as percentages you have to normalize. [Sampling](/shared/glossary/#sampling) rules ([temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), [top-p](/shared/glossary/#top-p)) all reshape the logits before the random draw, and [`argmax`](/shared/glossary/#argmax) of the logits is what [greedy decoding](/shared/glossary/#greedy-decoding) picks.

### LoRA {#lora}
[Low-Rank](/shared/glossary/#low-rank) Adaptation — fine-tune by adding small low-rank matrices, freeze the base

### Loss function {#loss-function}
A mathematical function that measures the difference between a model's prediction and the actual target. The goal of training is to minimize this value using [gradients](/shared/glossary/#gradients).

### Loss masking {#loss-masking}
Telling the trainer to compute the [loss](/shared/glossary/#loss-function) only on the tokens you want the model to learn to produce — in [SFT](/shared/glossary/#sft), the assistant's reply — and to ignore the rest, like grading only a student's answers and not the printed questions.

### Loss spike {#loss-spike}
A sudden jump in the training [loss](/shared/glossary/#loss-function), usually from an outlier batch or optimizer instability; small spikes are normal, but a diverging one can ruin a run.

### Loss value {#loss-value}
The single scalar number produced by evaluating the [loss function](/shared/glossary/#loss-function) on a model's predictions. [autograd](/shared/glossary/#autograd)'s [backward pass](/shared/glossary/#backward-pass) computes [gradients](/shared/glossary/#gradients) of this one scalar with respect to every [parameter](/shared/glossary/#parameters), which is what makes [reverse-mode](/shared/glossary/#reverse-mode) differentiation efficient.

### Lorax / S-LoRA {#lorax--s-lora}
Multi-LoRA serving engines; one base model + many adapters in HBM

### Low-rank {#low-rank}
A way of approximating a big matrix as the product of two much skinnier ones, capturing most of its information with far fewer numbers. A full 1000×1000 weight matrix holds a million entries, but if its real content is "low rank" you can rebuild it well from, say, two 1000×8 matrices — a few thousand numbers instead of a million. Like summarizing a thick report with a handful of bullet points that still carry the gist. This is the trick behind [LoRA](/shared/glossary/#lora): freeze the giant base [weights](/shared/glossary/#weights) and learn only a small low-rank update on top.

### LQR {#lqr}
Linear-Quadratic Regulator — optimal linear feedback for quadratic cost

### MagViT-v2 {#magvit-v2}
The strongest open recipe for discrete video tokenization

### Manipulability {#manipulability}
Scalar measure of how "easy" motion is from a given configuration (e.g. `sqrt(det(JJᵀ))`)

### Mantissa {#mantissa}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that holds the *precision digits* — the significant figures sitting in front of the scale factor. In `3.5 × 10¹²`, the `3.5` is the mantissa (also called the *significand*). More mantissa bits give finer resolution between nearby values; fewer mantissa bits leave larger gaps between representable numbers. [FP8](/shared/glossary/#fp8)'s `E4M3` format means 4 [exponent](/shared/glossary/#exponent) bits + 3 mantissa bits, so it can only distinguish about 8 distinct values between each consecutive power of two — coarse, but small enough to fit twice as many numbers in the same memory as [bfloat16](/shared/glossary/#bfloat16).

### matmul {#matmul}
Matrix multiplication — the dominant compute operation in neural networks; written `A @ B` in PyTorch.

### MDP {#mdp}
Markov Decision Process — the tuple `(S, A, P, R, γ)`

### Mechanistic interpretability {#mechanistic-interpretability}
The line of research that tries to reverse-engineer *what individual pieces of a neural network actually do* — which neurons or [attention heads](/shared/glossary/#heads) detect what, where a fact is stored, why a particular output came out. Like opening up a watch to see which gears turn the hands, instead of only timing how fast the watch runs. Main tools: [linear probes](/shared/glossary/#linear-probe), [sparse autoencoders](/shared/glossary/#sae), activation patching, and circuit analysis.

### Meta-learning {#meta-learning}
"Learning to learn" — training a model to adapt quickly to new tasks with few examples. Many meta-learning algorithms, such as MAML, rely on higher-order gradients to optimize across tasks.

### Memorization {#memorization}
When an [LLM](/shared/glossary/#llm) reproduces a chunk of its training data verbatim instead of generalizing from it — give it the right opening prompt and out comes the original passage word for word. Like a student who recites a textbook sentence rather than explaining the idea; useful for trivia, dangerous for copyright, PII, and security. [Deduplication](/shared/glossary/#deduplication) at training time and prompt filtering at serving time are the main mitigations.

### Memory leak {#memory-leak}
An unintended increase in memory usage over time, often caused in PyTorch by holding onto references to the [loss function](/shared/glossary/#loss-function) or other parts of the [dynamic computation graph](/shared/glossary/#dynamic-computation-graph) across training iterations.

### Memory mapping {#memory-mapping}
Accessing a file on disk as if it were an in-memory array, reading slices on demand without loading the whole file into RAM (e.g. `numpy.memmap`).

### Memory snapshot {#memory-snapshot}
A recording of how much GPU memory is allocated at one moment; comparing snapshots taken across training steps reveals a steadily growing [memory leak](/shared/glossary/#memory-leak).

### Megatron {#megatron}
NVIDIA's approach to [tensor parallelism](/shared/glossary/#tensor-parallelism-tp) that splits [attention](/shared/glossary/#attention) and [MLP](/shared/glossary/#mlp) layers [column-wise](/shared/glossary/#column-wise-partitioning) and row-wise across GPUs with carefully placed [AllReduce](/shared/glossary/#allreduce) collectives, allowing efficient intra-layer parallelism.

### MFU {#mfu}
Model FLOPs Utilization — the fraction of a GPU's peak arithmetic speed a training run actually uses (e.g. 70% MFU). Like a delivery truck's fill rate, it shows how much of the hardware you are paying for is doing useful work instead of waiting on memory or the network.

### Micrograd {#micrograd}
A tiny, educational autograd engine implemented in basic Python by Andrej Karpathy to illustrate how reverse-mode differentiation works.

### MinHash {#minhash}
A hashing technique for estimating how similar two documents are, used to find and remove near-duplicate text at corpus scale (see [deduplication](/shared/glossary/#deduplication)).

### MLP {#mlp}
Multi-Layer Perceptron — a [feedforward neural network](/shared/glossary/#ffn) of one or more fully-connected (linear) layers separated by non-linear [activations](/shared/glossary/#activations). In [transformer](/shared/glossary/#transformer) architectures, each block contains an [attention](/shared/glossary/#attention) sublayer followed by an MLP sublayer (often using [SwiGLU](/shared/glossary/#swiglu) activation).

### MMDiT {#mmdit}
Multi-Modal Diffusion Transformer — joint text+image attention layers, used in SD3 and Flux

### MMLU {#mmlu}
Massive Multitask Language Understanding — a 57-subject multiple-choice [benchmark](/shared/glossary/#benchmark) (history, law, medicine, math, and more) that became the standard quick test of how much general knowledge a model has, like a giant trivia exam spanning many school subjects at once.

### Modality gap {#modality-gap}
Empirical finding that different-modality embeddings stay in separable regions

### Mode collapse {#mode-collapse}
GAN failure mode: generator produces few distinct outputs

### MoE {#moe}
Mixture-of-Experts — sparse routing across N expert [MLPs](/shared/glossary/#mlp); high total params, fixed compute per token

### Momentum {#momentum}
A technique that accumulates a moving average of past gradients to dampen oscillations and accelerate gradient descent in consistent directions

### Monosemantic {#monosemantic}
A feature inside a neural network that fires for exactly *one* concept — for example, a direction in [activation](/shared/glossary/#activations) space that lights up only for "Golden Gate Bridge," or only for "negation in a clause." The opposite is *polysemantic*: one neuron that activates for several unrelated concepts at once. Like a single word that means just one thing versus a homonym that means several. Recovering monosemantic features is the main goal of [SAE](/shared/glossary/#sae)-based interpretability.

### MoveIt {#moveit}
ROS 2 manipulation-planning framework

### MPC {#mpc}
Model Predictive Control — re-solved finite-horizon optimization at each step

### MPS {#mps}
Metal Performance Shaders — the GPU backend for Apple Silicon

### MQA {#mqa}
Multi-Query Attention — all query heads share a single key/value head; the most aggressive [KV-cache](/shared/glossary/#kv-cache) saver, at some quality cost

### MT-Bench {#mt-bench}
A benchmark that scores a chat model's answers to a set of multi-turn questions, often using a strong [LLM](/shared/glossary/#llm) as the judge; a quick proxy for how helpful an assistant feels.

### MuJoCo {#mujoco}
Open-source physics engine; the de facto manipulation/locomotion simulator

### Multi-head attention {#multi-head-attention}
Running several [attention](/shared/glossary/#attention) operations ([heads](/shared/glossary/#heads)) in parallel, each with its own learned projections of queries, keys, and values, then concatenating their results. Like having several readers skim the same sentence for different things — one tracks the grammar, another tracks who-did-what — and then pooling what each one noticed.

### Multi-LoRA {#multi-lora}
Serving many fine-tuned adapters on a single shared base model

### NaN {#nan}
"Not a Number" — a floating-point value representing an undefined or unrepresentable result (e.g., `0/0` or `inf - inf`). In PyTorch, NaNs often appear when [gradients](/shared/glossary/#gradients) explode or when taking the logarithm of zero/negative numbers.

### Native multimodal {#native-multimodal}
A model trained from scratch on all modalities with a unified vocabulary

### Numerical issues {#numerical-issues}
Problems arising from the finite precision of floating-point numbers, such as [underflow](/shared/glossary/#underflow), overflow, or loss of precision, which can lead to unstable training or [NaN](/shared/glossary/#nan) values.

### Nav2 {#nav2}
ROS 2 navigation stack

### NCCL {#nccl}
NVIDIA Collective Communications Library — does AllReduce etc. on NVIDIA GPUs

### nDCG {#ndcg}
Normalized Discounted Cumulative Gain — a ranking-quality score from 0 to 1 that rewards putting the most relevant results near the top of the list; the standard way to check whether a [reranker](/shared/glossary/#reranker) actually improved the ordering.

### Needle-in-a-haystack {#needle-in-a-haystack}
A long-context test that hides one fact (the "needle") inside a long stretch of irrelevant text (the "haystack") and checks whether the model can find it

### Next-token prediction {#next-token-prediction}
The training objective of an [LLM](/shared/glossary/#llm): given the tokens so far, predict the next one, scored with [cross-entropy](/shared/glossary/#cross-entropy) [loss](/shared/glossary/#loss-function).

### nn.Module {#nnmodule}
PyTorch's base class for all neural network components; acts as a registry that automatically tracks sub-modules, parameters, and buffers assigned in `__init__`

### Node (distributed) {#node-distributed}
One physical machine (server) in a distributed job, usually holding several GPUs; multi-node training spreads work across several of them over a network.

### non_blocking {#non_blocking}
The `non_blocking=True` flag on `.to()` / `.cuda()` that lets a host→device copy run asynchronously from pinned memory

### Normalization {#normalization}
Rescaling a layer's outputs so they keep a consistent size — typically zero mean and unit variance (LayerNorm) or unit root-mean-square ([RMSNorm](/shared/glossary/#rmsnorm)). Like adjusting every photo to the same brightness before comparing them, it stops numbers from ballooning or vanishing as they flow through a deep network, which is what keeps training stable.

### NVLink {#nvlink}
NVIDIA's GPU-GPU interconnect; much faster than PCIe

### NVSwitch {#nvswitch}
NVLink switch chip; full-bandwidth all-to-all within a node

### Off-policy {#off-policy}
The data comes from a different policy than the one being optimized

### Offset {#offset}
The starting index into the underlying storage where a tensor's data begins (`.storage_offset()`)

### OMPL {#ompl}
Open Motion Planning Library — sampling-based planners

### Online softmax {#online-softmax}
An incremental method for computing [softmax](/shared/glossary/#softmax) that maintains running maximum and sum statistics, enabling single-pass computation over tiled inputs without materializing the full exponent sum beforehand.

### ONNX {#onnx}
Open Neural Network Exchange — a framework-neutral file format that stores a model as a graph of operations, so it can run outside the framework that trained it.

### ONNX Runtime {#onnx-runtime}
A fast, cross-platform engine that runs models saved in the [ONNX](/shared/glossary/#onnx) format, without needing the original framework like PyTorch.

### On-policy {#on-policy}
The data comes from the same policy being optimized (PPO, REINFORCE)

### Open-ended {#open-ended}
A task where many different answers can all be reasonable and there is no single right one to check against — writing a poem, summarizing an article, replying helpfully in a chat. The opposite of a closed-ended task like a multiple-choice question (one correct letter) or arithmetic (one correct number). Like grading a creative-writing assignment versus grading a true/false quiz: with the quiz you just count matches, but with the essay you need a human reader — or an [LLM-as-judge](/shared/glossary/#llm-as-judge) — to weigh quality, which is why evaluating open-ended work is the hard part of LLM evals.

### Open model {#open-model}
A model whose [weights](/shared/glossary/#weights) you can download and run yourself — Meta's Llama, Mistral, Qwen, DeepSeek — as opposed to a *closed* model like GPT-4 or Claude where the weights stay on the provider's servers and you can only call them through an API. Like the difference between buying a recipe book (you have the actual instructions, can modify them, can bake offline) and ordering at a restaurant (you only see the finished dish). Open models are essential for any white-box research that needs the model's internals: methods like [GCG](/shared/glossary/#gcg) optimize against the model's own [gradients](/shared/glossary/#gradients), and interpretability tools like [SAEs](/shared/glossary/#sae) read its hidden [activations](/shared/glossary/#activations) — neither is possible through a closed API.

### Optimizer {#optimizer}
An algorithm that updates model parameters using computed gradients; in PyTorch, a subclass of `torch.optim.Optimizer` that holds parameter groups and per-parameter state

### Optimizer state {#optimizer-state}
The extra per-parameter values an [optimizer](/shared/glossary/#optimizer) stores between steps — for example, [Adam](/shared/glossary/#adam) keeps two (the first- and second-moment estimates) — which adds to training memory.

### Outcome reward model {#outcome-reward-model}
A scorer that judges only a solution's final answer as right or wrong, ignoring the steps in between — simpler than a [process reward model](/shared/glossary/#process-reward-model), which grades each step, but blind to *where* a wrong answer first went off track.

### Outlines {#outlines}
An open-source Python library for [constrained generation](/shared/glossary/#constrained-generation): you hand it a regular expression, a JSON schema, or a [Pydantic](/shared/glossary/#pydantic) model and it patches the [LLM](/shared/glossary/#llm)'s decoder to mask out any next-token choices that would break the structure. Like putting guardrails on a road so the car physically cannot drive off the edge no matter how the driver steers, it makes the model's output structurally valid by construction rather than by hope.

### Padding {#padding}
Filling shorter sequences with a placeholder value so that every sample in a batch has the same length.

### Parameters {#parameters}
The learnable [tensors](/shared/glossary/#tensor) inside a model (such as [weights](/shared/glossary/#weights) and [biases](/shared/glossary/#biases)) that are updated by the [optimizer](/shared/glossary/#optimizer) during training. In PyTorch, they are instances of `nn.Parameter` and are automatically registered when assigned to an [`nn.Module`](/shared/glossary/#nnmodule).

### Partial derivative {#partial-derivative}
How much a function changes when you nudge just one of its inputs and hold all the others still — the [derivative](/shared/glossary/#derivative) taken one input at a time. If a recipe's tastiness depends on both salt and sugar, the partial derivative with respect to salt tells you the effect of adding a pinch more salt while keeping the sugar fixed. A [gradient](/shared/glossary/#gradients) is simply the full list of these one-at-a-time slopes, one per [parameter](/shared/glossary/#parameters).

### PagedAttention {#pagedattention}
A way of storing the [KV cache](/shared/glossary/#kv-cache) for many concurrent requests by splitting each request's cache into small fixed-size "pages" that the engine can scatter freely around GPU memory and look up through a per-request page table — the same idea operating systems use for virtual memory. It removes the wasted space and fragmentation you get when each request needs its own contiguous chunk, which is why [vLLM](/shared/glossary/#vllm) made it the default scheme.

### Patchification {#patchification}
Splitting a (latent) tensor into a sequence of patch tokens for a transformer

### PCIe {#pcie}
The standard CPU-GPU connection (and slower GPU-GPU when no NVLink)

### Perceptual loss (LPIPS) {#perceptual-loss-lpips}
Loss computed in the feature space of a pretrained classifier; sharper than pixel MSE

### permute {#permute}
Reorders all of a tensor's dimensions by rewriting strides — never copies

### PID {#pid}
Proportional-Integral-Derivative — the workhorse linear controller

### Pinned memory {#pinned-memory}
Page-locked CPU memory that enables faster, asynchronous transfers to the GPU; enabled with `pin_memory=True` on a [DataLoader](/shared/glossary/#dataloader).

### Pinocchio {#pinocchio}
Fast rigid-body dynamics library (CRBA, RNEA, ABA)

### Plücker coordinates {#plücker-coordinates}
6D representation of a camera ray; standard for camera-conditioning

### Policy {#policy}
In reinforcement learning, the model being trained to choose what to do next — for an [LLM](/shared/glossary/#llm), the network that picks the next token. "Improving the policy" just means making those choices earn more reward.

### Position bias {#position-bias}
A judge's tendency to pick an answer based on *where* it sits rather than *what* it says — for example, an [LLM-as-judge](/shared/glossary/#llm-as-judge) that quietly prefers whichever response appears first (or last) when shown two side-by-side. Like a job interviewer who can't help favoring the candidate they meet right after lunch, regardless of qualifications. The standard fix is to ask the judge twice with the two answers swapped and accept the verdict only if both runs name the same winner.

### Position interpolation {#position-interpolation}
Extending a model's context length by linearly rescaling [RoPE](/shared/glossary/#rope) position indices so longer sequences fall within the trained range

### Posterior collapse {#posterior-collapse}
VAE failure mode: encoder collapses to the prior; latent carries no information

### PPO {#ppo}
Proximal Policy Optimization — the [workhorse](/shared/glossary/#workhorse) [on-policy](/shared/glossary/#on-policy) RL algorithm, used in classic RLHF

### Prefill {#prefill}
Processing the prompt in one parallel forward pass before decoding begins

### Prefix cache {#prefix-cache}
Sharing KV cache across requests that begin with the same tokens (e.g., system prompts)

### Pretraining {#pretraining}
Self-supervised training on a large unlabeled corpus to predict the next token

### PRM {#prm}
Probabilistic Roadmap — multi-query sampling-based planner

### PRM800K {#prm800k}
A public dataset of about 800,000 human labels that mark each step of a math solution as right or wrong, released by OpenAI to train [process reward models](/shared/glossary/#process-reward-model). Rather than only checking whether the final answer was correct, human graders read each worked solution line by line — like a math teacher putting a check or an X next to every step of a student's proof, not just the boxed answer at the bottom. Because the feedback is step-level, a model trained on it learns to spot exactly where the reasoning went off the rails instead of whether the ending happened to be lucky. It is the standard training set for the step-by-step scorers used in [Best-of-N](/shared/glossary/#best-of-n) re-ranking.

### Probability flow ODE {#probability-flow-ode}
The deterministic ODE equivalent of the reverse-time diffusion SDE

### Process reward model {#process-reward-model}
A scorer that grades each individual step of a model's reasoning rather than just the final answer — like a teacher marking every line of a proof, not only the last one — so a mistake can be caught at the exact step it happens. Contrast with an [outcome reward model](/shared/glossary/#outcome-reward-model).

### Profiler {#profiler}
A tool (`torch.profiler`) that records how long each operation in a training step takes, used to locate performance [bottlenecks](/shared/glossary/#bottleneck).

### Projector {#projector}
The (usually small) network that maps one modality's features into another's space

### Prompt injection {#prompt-injection}
An attack in which adversarial text smuggled into something the model reads — a retrieved document, a tool's output, an email, even text inside an image — overrides the original system instructions. Like a customer slipping a fake "manager-approved" note into a server's order pile: the server can't easily tell the planted note from a real one. The hardest unsolved security problem in deployed [LLMs](/shared/glossary/#llm), because the model has no built-in way to separate "instructions" from "data" in its input.

### PTQ / QAT {#ptq--qat}
Post-Training Quantization / Quantization-Aware Training

### Pydantic {#pydantic}
A popular Python library for declaring the *shape* of your data as a class — you write a class with typed fields (e.g. `name: str`, `age: int`) and Pydantic validates that any data you load actually matches, raising a clear error if a value is the wrong type or a required field is missing. Like a customs form for data: anything that does not match the listed fields gets stopped at the border. In LLM work it is the standard way to describe the JSON object you want the model to produce, which tools like [Outlines](/shared/glossary/#outlines) or OpenAI's structured-output mode can then enforce during decoding.

### Q-Former {#q-former}
BLIP-2's learnable-query cross-attention module for distilling images into LLM tokens

### QLoRA {#qlora}
[LoRA](/shared/glossary/#lora) with the frozen base model stored in 4-bit [quantized](/shared/glossary/#quantization) form, cutting memory so much you can fine-tune a large model on a single consumer GPU.

### Quality filter {#quality-filter}
A classifier that scores each training document and keeps only the high-quality ones (e.g. educational web text), discarding low-value text before [pretraining](/shared/glossary/#pretraining).

### Quantization {#quantization}
Reducing weight / activation precision (FP16, BF16, FP8, INT8, INT4) to save memory and bandwidth

### RadixAttention {#radixattention}
sglang's KV cache organized as a radix tree keyed on prompt prefixes for automatic sharing

### RAG {#rag}
Retrieval-Augmented Generation — fetch documents, prepend to prompt, then generate

### rank {#rank}
The unique integer ID of a process in a distributed job. `RANK` is the global ID across all machines; `LOCAL_RANK` is the ID within one machine; `WORLD_SIZE` is the total number of processes.

### ReAct {#react}
A simple [agent](/shared/glossary/#agent) pattern that interleaves **Rea**soning and **Act**ing: the model writes a thought, takes an action with a tool, reads the observation, then repeats — the loop most basic agents are built on.

### Reciprocal rank fusion {#reciprocal-rank-fusion}
A simple, robust way to merge several ranked lists into one: each item scores the sum of `1 / (rank + constant)` across the lists, so items ranked highly by more than one retriever rise to the top. Common for combining dense and sparse search in [hybrid retrieval](/shared/glossary/#hybrid-retrieval).

### Rectified flow {#rectified-flow}
A flow-matching parameterization with straight-line trajectories; popular in 2024+ models

### Reference model {#reference-model}
A frozen copy of the starting model that [RLHF](/shared/glossary/#rlhf) and [DPO](/shared/glossary/#dpo) measure against (through a [KL](/shared/glossary/#kl-divergence) term) so the model being trained does not drift too far from sensible behavior — a "before" photo to compare every change against.

### Reparameterization trick {#reparameterization-trick}
`z = μ + σ · ε` — lets gradients flow through a random sample

### Reranker {#reranker}
A second-stage model that re-scores the top candidates from a fast first-stage retriever and reorders them by true relevance — usually a [cross-encoder](/shared/glossary/#cross-encoder). The "retrieve then rerank" two-stage pattern is standard in search and [RAG](/shared/glossary/#rag).

### reshape {#reshape}
Returns a tensor with a new shape, copying only when a no-copy view isn't possible

### Residual connection {#residual-connection}
A shortcut that adds a block's input to its output (`x + f(x)`), letting [gradients](/shared/glossary/#gradients) flow directly and making deep networks trainable

### Residual stream {#residual-stream}
In a [transformer](/shared/glossary/#transformer), the running activation vector that flows through every layer via [residual connections](/shared/glossary/#residual-connection) — each [attention](/shared/glossary/#attention) block and [MLP](/shared/glossary/#mlp) block reads from this stream and adds its update back to it, without erasing what came before. Like a shared bulletin board that every department reads and pins notes to as it passes through the office: by the end of the building, the board carries the combined contribution of every team. Because every layer reads and writes the same vector space, the residual stream is the most natural place to look for interpretable features, which is why [sparse autoencoders (SAEs)](/shared/glossary/#sae) are usually trained on residual-stream [activations](/shared/glossary/#activations).

### reverse-mode {#reverse-mode}
The order [autograd](/shared/glossary/#autograd) walks the computation graph when differentiating: the forward pass first, then a single [backward pass](/shared/glossary/#backward-pass) that propagates [gradients](/shared/glossary/#gradients) from the scalar output back to every input. It is the efficient choice when a model has many parameters but only one [loss value](/shared/glossary/#loss-value).

### Reward hacking {#reward-hacking}
A policy that maximizes the reward signal without doing what was intended

### Reward model {#reward-model}
A model trained on human preference comparisons to score how good a response is; it stands in for a human rater so [RLHF](/shared/glossary/#rlhf) can score millions of answers automatically.

### RLAIF {#rlaif}
Reinforcement Learning from **AI** Feedback — the same recipe as [RLHF](/shared/glossary/#rlhf) but the preference labels (or grades) are produced by another, stronger LLM following a written rubric instead of by paid human raters. Like swapping a panel of human judges for a single expert judge who works for free, never sleeps, and applies the same rules every time. Cheaper and faster than human labeling, often nearly as good on well-defined tasks, and the basis of [Constitutional AI](/shared/glossary/#constitutional-ai).

### RLHF {#rlhf}
Reinforcement Learning from Human Feedback — preference learning, classically via [PPO](/shared/glossary/#ppo) + [KL](/shared/glossary/#kl-divergence)

### RLVR {#rlvr}
RL with Verifiable Rewards — RL when the reward is a deterministic checker

### RMSNorm {#rmsnorm}
Root-Mean-Square LayerNorm without mean-centering; the modern default

### RNEA {#rnea}
Recursive Newton-Euler — `O(n)` inverse-dynamics algorithm

### Rollout {#rollout}
One sample of the model actually generating a full response to a prompt, used in RL to see what behavior to reward; producing many rollouts is the expensive part of [PPO](/shared/glossary/#ppo) and [GRPO](/shared/glossary/#grpo).

### Rollout distribution {#rollout-distribution}
The spread of responses a model is currently generating when it produces [rollouts](/shared/glossary/#rollout) during RL training — what it tends to say and how varied those answers are. This distribution shifts as training proceeds, which is the whole point; but if it drifts toward weird, repetitive, or gamed outputs, that is a warning sign of [reward hacking](/shared/glossary/#reward-hacking). Watching how it moves is like checking what a student actually writes on practice tests, not just their final score.

### Roofline {#roofline}
Performance model bounding throughput as min(peak FLOPs, memory bandwidth × arithmetic intensity)

### RoPE {#rope}
Rotary Position [Embedding](/shared/glossary/#embedding) — encodes position by rotating Q, K vectors

### ROS / ROS 2 {#ros--ros-2}
Robot Operating System — robotics middleware (ROS 2 is the modern version)

### RRT {#rrt}
Rapidly-exploring Random Tree — single-query sampling-based planner

### SAC {#sac}
Soft Actor-Critic — maximum-entropy continuous-control algorithm; the modern default

### SAE {#sae}
Sparse Autoencoder — interpretability tool decomposing activations into monosemantic features

### Sampler {#sampler}
The component that decides the order in which a [DataLoader](/shared/glossary/#dataloader) visits dataset examples (e.g. random, sequential, or class-weighted).

### Sandbox {#sandbox}
An isolated, throwaway environment — like a fenced-off playground — where an [agent](/shared/glossary/#agent) or program can run commands, create files, and make mistakes without affecting your real computer. If the agent breaks something inside the sandbox, you just throw the sandbox away; nothing outside it is touched. Containers (like Docker) and virtual machines are common ways to build one.

### Sampling {#sampling}
Drawing the next token from the model's predicted probability distribution instead of always taking the most likely one; [temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), and [top-p](/shared/glossary/#top-p) control how random the choice is.

### Scaling laws {#scaling-laws}
The empirical finding that a model's [loss](/shared/glossary/#loss-function) drops in a smooth, predictable curve as you add [parameters](/shared/glossary/#parameters), training data, and compute — like a growth chart that lets you forecast a bigger model's quality from smaller ones before you ever build it.

### Score {#score}
`∇_x log p(x)` — diffusion training implicitly learns this

### Scratchpad {#scratchpad}
A temporary, fast-access workspace where intermediate results are stashed so they don't have to be recomputed later. Like a math student's scratch paper next to an exam: jot the partial sums, look them up later, move on much faster than redoing each calculation. In serving, the [KV cache](/shared/glossary/#kv-cache) is the model's scratchpad — every key and value it has already computed sits there ready to be reused on the next decode step.

### SDF {#sdf}
Signed Distance Field — scalar field giving distance to nearest obstacle (negative inside)

### SE(3) / SO(3) {#se3--so3}
Special Euclidean / Orthogonal group — rigid-body motions / rotations in 3D

### Seed {#seed}
A fixed starting number for a random-number generator; setting the same seed makes random operations (shuffling, initialization, dropout) produce the identical sequence every run.

### Sentence embedding {#sentence-embedding}
A single dense vector that captures the meaning of an entire sentence (or short passage), so two sentences about the same topic end up close together in vector space even if they use completely different words. Think of it as a GPS coordinate for meaning — two sentences that "mean the same thing" land near the same point on the map. Sentence [embeddings](/shared/glossary/#embedding) are the backbone of semantic search in [RAG](/shared/glossary/#rag): you embed the user's question and every stored passage, then find the passages whose coordinates are closest.

### Self-consistency {#self-consistency}
Sampling many independent [chain-of-thought](/shared/glossary/#cot) solutions to the same problem and taking a majority vote on the final answer — like asking several people to solve a puzzle on their own and trusting the answer most of them land on.

### SFT {#sft}
Supervised Fine-Tuning — train on demonstration data with [cross-entropy](/shared/glossary/#cross-entropy)

### SGD {#sgd}
Stochastic Gradient Descent — updates parameters by subtracting a scaled gradient computed on a mini-batch; the simplest optimizer and the basis for more advanced methods

### sglang {#sglang}
An open-source LLM serving runtime that pairs fast inference (via [RadixAttention](/shared/glossary/#radixattention) prefix sharing) with first-class [constrained generation](/shared/glossary/#constrained-generation) — built-in regex / JSON / grammar constraints applied at decode time. Plays a similar role to [vLLM](/shared/glossary/#vllm) but is the popular pick when reliable structured output (function calls, tool use, schema-conformant JSON) matters most.

### Shape {#shape}
The size of a tensor along each dimension; the tuple returned by `.shape`

### Sharding {#sharding}
Splitting a dataset (or model) into many smaller pieces so they can be stored, loaded, or processed in parallel.

### SigLIP {#siglip}
Sigmoid-loss CLIP variant; scales better and works at smaller batch sizes

### SIMT {#simt}
Single Instruction Multiple Threads; NVIDIA's execution model

### SLAM {#slam}
Simultaneous Localization and Mapping

### SLO {#slo}
Service Level Objective — a quantified commitment (e.g., P95 TTFT < 500 ms)

### SM {#sm}
Streaming Multiprocessor; the GPU's "core"

### softmax {#softmax}
The function that turns a vector of scores into a probability distribution (each value in 0–1, summing to 1); the core of [attention](/shared/glossary/#attention) and classification heads.

### Special tokens {#special-tokens}
Reserved [vocabulary](/shared/glossary/#vocabulary) entries that mark structure rather than text — e.g. `<bos>`, `<eos>`, `<pad>`, and chat-boundary tokens like `<|im_start|>`

### Speculative decoding {#speculative-decoding}
Use a draft model to propose tokens; verify with the target in one parallel pass; accepted tokens are appended

### State dict {#state-dict}
A Python `OrderedDict` that maps every parameter and buffer name to its tensor value; the standard format for saving, loading, and transplanting PyTorch model weights

### Static quantization (PTQ) {#static-quantization-ptq}
A [quantization](/shared/glossary/#quantization) method that converts both [weights](/shared/glossary/#weights) and [activations](/shared/glossary/#activations) to [int8](/shared/glossary/#int8) before serving, using a [calibration](/shared/glossary/#calibration) pass to fix the activation scales in advance.

### Stop-string {#stop-string}
A user-supplied substring that tells the server "as soon as the generated text contains this, stop." Matched on the *decoded* text, not the raw token IDs, because the same letters can land in different [BPE](/shared/glossary/#bpe) tokens depending on what came before — so the matcher has to keep a small rolling window of recent output and check for the string at every step.

### Storage {#storage}
The 1-D buffer that a tensor is a view into

### Straight-through estimator {#straight-through-estimator}
A technique used to bypass non-differentiable operations by passing gradients unchanged through the operation during the backward pass.

### Streaming {#streaming}
Sending the model's reply to the client one piece at a time as it is generated, instead of waiting for the whole answer and then returning it in a single response. Over HTTP this is usually done with Server-Sent Events (SSE) or chunked transfer encoding; the connection stays open and the server flushes each new token as soon as it is sampled. Like a waiter who brings each course out as it leaves the kitchen rather than holding the whole meal until dessert is ready — the user sees [TTFT](/shared/glossary/#ttft) drop dramatically even though total generation time is the same.

### Stride {#stride}
The number of storage elements to step over for each dimension of a tensor

### SWE (Software Engineering) {#swe}
Short for **Software Engineering** — the discipline of building, testing, and maintaining software systems. In the AI/LLM context, "SWE" usually appears in compound terms like [SWE-bench](/shared/glossary/#swe-bench) or "SWE-style agent," meaning an [agent](/shared/glossary/#agent) that does the kind of work a human software engineer does: reading code, diagnosing bugs, writing fixes, and running tests.

### SWE-bench {#swe-bench}
Short for **Software Engineering Benchmark** — a benchmark of real GitHub issues paired with the code changes that fixed them; an [agent](/shared/glossary/#agent) is judged by whether its edits make the project's test suite pass, which makes it the standard test of coding agents.

### Sweep {#sweep}
Training the same model many times while changing one setting across a range of values, then comparing results to pick the best — for example trying ten different [learning rates](/shared/glossary/#learning-rate) and keeping the winner. Like tasting a sauce as you add salt in small steps to find the amount you like, rather than guessing the whole spoonful at once. A sweep is how you turn a hyperparameter hunch into a measured choice.

### SwiGLU {#swiglu}
[Gated](/shared/glossary/#gated) [MLP](/shared/glossary/#mlp) activation `(xW) · σ(xV)` — the modern default [FFN](/shared/glossary/#ffn); a [GLU](/shared/glossary/#glu) variant that uses [Swish](/shared/glossary/#swish) as its gating non-linearity

### Swish {#swish}
Activation function `x · σ(x)` (also called SiLU) — a smooth, non-monotonic alternative to ReLU; the gating non-linearity used inside [SwiGLU](/shared/glossary/#swiglu).

### System prompt {#system-prompt}
A message placed at the very start of a chat conversation that tells the model how to behave — its role, tone, rules, and the tools it can call — before the user's first turn ever arrives. Like a stage director's note to an actor before the curtain rises: *"You're a polite customer-support agent who answers only refund questions."* System prompts are usually long and shared across many requests, which is why caching their KV state (see [prefix cache](/shared/glossary/#prefix-cache)) saves so much repeated work.

### Systolic array {#systolic-array}
Data-flow matmul fabric used in TPUs

### T2V {#t2v}
Text-to-Video

### Tail latency {#tail-latency}
The [latency](/shared/glossary/#latency) of the slowest requests (for example the p95 or p99 percentiles) rather than the median (p50); it is what users notice most.

### TCP {#tcp}
Tool Center Point — the configurable point on a tool whose pose tracking controls

### TD error {#td-error}
`δ_t = r_t + γV(s_{t+1}) − V(s_t)` — the signal that drives every TD update

### TD3 {#td3}
Twin Delayed DDPG — DDPG plus three stability fixes

### Temperature {#temperature}
A [sampling](/shared/glossary/#sampling) knob that scales the model's scores before [softmax](/shared/glossary/#softmax): low temperature (e.g. 0.2) sharpens the distribution so the model plays it safe and repeats the likeliest words, while high temperature (e.g. 1.5) flattens it so rarer, more surprising words can win. Think of it as a creativity dial — turn it down for factual answers, up for brainstorming.

### Temporal inflation {#temporal-inflation}
Adding time-axis layers to a pretrained 2D model

### Tensor {#tensor}
A multidimensional array — a (storage, shape, stride, offset, dtype, device, requires_grad) tuple that views a 1-D storage buffer

### Tensor Core {#tensor-core}
Specialized matmul unit in NVIDIA GPUs since [Volta](/shared/glossary/#volta)

### Tensor parallelism (TP) {#tensor-parallelism-tp}
Sharding each layer's weights across GPUs with all-reduce at attention/[MLP](/shared/glossary/#mlp) boundaries

### TFLOPs {#tflops}
Tera (10¹²) floating-point operations per second

### TGI {#tgi}
Short for **Text Generation Inference** — Hugging Face's open-source LLM serving engine, similar in role to [vLLM](/shared/glossary/#vllm). It implements [continuous batching](/shared/glossary/#continuous-batching), [PagedAttention](/shared/glossary/#pagedattention), and quantized inference behind a simple HTTP API, and is one of the two engines most commonly used to put an LLM in front of real users.

### Throughput {#throughput}
How much work is completed per unit of time — for training, the number of examples processed per second.

### Tiling {#tiling}
Splitting a large computation into small blocks ("tiles") that fit in fast on-chip memory, so a [kernel](/shared/glossary/#kernel) reads slow memory fewer times.

### Token (visual/audio) {#token-visualaudio}
Discrete code from a VQ-VAE or neural codec; lets transformers treat the modality like language

### Tokenizer {#tokenizer}
The mapping from string to integer IDs; trained, frozen, part of the model contract

### Tokens per byte {#tokens-per-byte}
A measure of tokenizer efficiency: how many tokens it emits per byte of input text; higher means the same text costs more tokens

### Top-k {#top-k}
A [sampling](/shared/glossary/#sampling) rule that keeps only the `k` most likely next tokens and draws from those, throwing away the long tail. With `k=1` it always takes the single best word (greedy decoding); with `k=50` it chooses among the top 50 — like ordering only from a menu's 50 most popular dishes instead of the whole cookbook.

### Top-p {#top-p}
Also called nucleus sampling: instead of a fixed count like [top-k](/shared/glossary/#top-k), it keeps the smallest set of top tokens whose probabilities add up to `p` (e.g. 0.9), then samples from them. The shortlist automatically grows when the model is unsure and shrinks when it is confident — always keeping just enough candidates to cover 90% of the model's belief.

### TOPP {#topp}
Time-Optimal Path Parameterization — time-parameterize a geometric path under bounds

### torch.compile {#torchcompile}
The PyTorch 2.x API that traces a model into a graph and generates optimized, [fused kernels](/shared/glossary/#kernel-fusion), speeding up [eager mode](/shared/glossary/#eager-mode) code with a single call.

### torch.export {#torchexport}
The modern PyTorch API that captures a model into a standalone graph; the foundation for deployment paths like [ExecuTorch](/shared/glossary/#executorch) and [AOTInductor](/shared/glossary/#aotinductor).

### torch.multinomial {#torchmultinomial}
The PyTorch function that draws a random sample from a probability distribution: hand it a list of probabilities and it rolls a weighted die, returning the index it lands on. A token with probability `0.6` comes up about 60% of the time. It is the "roll the dice" step at the end of [sampling](/shared/glossary/#sampling) — the opposite of [`argmax`](/shared/glossary/#argmax), which never gambles. On the GPU each call is its own [kernel](/shared/glossary/#kernel) launch, which is why folding it into the rest of the sampling math can speed up [decode](/shared/glossary/#decode).

### torchrun {#torchrun}
PyTorch's launcher command that starts one process per GPU and sets the `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables those processes need to find each other.

### TorchScript {#torchscript}
The legacy serialization/IR for PyTorch; superseded by `torch.export`

### Transformer {#transformer}
The decoder-only / encoder-only / encoder-decoder architecture built from [attention](/shared/glossary/#attention) + [MLP](/shared/glossary/#mlp) blocks

### TransformerEngine {#transformerengine}
NVIDIA's open-source library that automates safe [FP8](/shared/glossary/#fp8) training and inference on [Hopper](/shared/glossary/#hopper) and [Blackwell](/shared/glossary/#blackwell) GPUs — it picks per-tensor scales each step so the low-bit math stays numerically stable. Like a thermostat for low-precision arithmetic: as values drift toward overflow or underflow, it nudges the scale to keep them inside the safe range. Drop-in [transformer](/shared/glossary/#transformer) layers wrap your model and turn FP8 on without the user having to manage the scaling manually.

### transpose {#transpose}
Swaps two dimensions by rewriting strides — never copies; the result is usually non-contiguous

### Tree-of-Thoughts {#tree-of-thoughts}
A reasoning method that explores several partial solutions at once as branches of a tree, scores them, and expands only the promising ones — like working through a maze by trying multiple paths and backing out of dead ends instead of committing to the first turn.

### Triage {#triage}
Sorting cases by what each one needs, borrowed from emergency-room medicine where a nurse classifies arriving patients by severity before any doctor sees them. In LLM evaluation, *hallucination triage* means sorting model answers into useful buckets — *correctly answered*, *correctly abstained ("I don't know")*, *confidently wrong* ([hallucination](/shared/glossary/#hallucination)) — so each rate can be measured separately, instead of collapsing everything into one "accuracy" number that hides which failures are dangerous.

### Triton {#triton}
A Python-flavored language for writing GPU kernels, developed by OpenAI

### Triton Inference Server {#triton-inference-server}
NVIDIA's production server for hosting models behind an HTTP/gRPC API, with [batching](/shared/glossary/#batching) and multi-model support; unrelated to the [Triton](/shared/glossary/#triton) kernel language despite the shared name.

### TTFT {#ttft}
Time to *produce* the first token — the elapsed time from when a request arrives at the server until the model returns its first output token, dominated by [prefill](/shared/glossary/#prefill) plus any queue wait. Like a restaurant's "time until your drink arrives" — felt separately from the rest of the meal, and the first thing the user actually notices.

### U-Net {#u-net}
Encoder-decoder architecture with skip connections; the standard diffusion backbone before DiT

### Underflow {#underflow}
Condition where a floating-point value is too small to be represented and rounds to zero; common with `float16` when accumulating very small gradients

### URDF / MJCF / USD {#urdf--mjcf--usd}
Robot description formats (ROS, MuJoCo, NVIDIA respectively)

### User turn {#user-turn}
One message a user sends in a chat conversation, paired with the model's reply (the *assistant turn*). A back-and-forth between user and assistant is a sequence of alternating turns, all under the same opening [system prompt](/shared/glossary/#system-prompt). In typical traffic, the system prompt is long and fixed while each user turn is short and varies — which is exactly the pattern a [prefix cache](/shared/glossary/#prefix-cache) exploits.

### V2V {#v2v}
Video-to-Video

### Vanishing gradients {#vanishing-gradients}
A problem during training where [gradients](/shared/glossary/#gradients) become extremely small, effectively preventing the weights from changing their value and stalling the learning process.

### VAE {#vae}
Variational Autoencoder — encoder/decoder pair trained on the ELBO

### Validation loss {#validation-loss}
The [loss](/shared/glossary/#loss-function) measured on held-out data the model was not trained on; the honest signal of how well training is generalizing.

### Value function {#value-function}
Expected return; `V(s)` for state-value, `Q(s, a)` for action-value

### Value network {#value-network}
The helper network (the "critic") in some RL algorithms that estimates the [value function](/shared/glossary/#value-function) — its best guess of how much future reward a situation is worth — so the [policy](/shared/glossary/#policy) can tell whether an action turned out better or worse than expected. [PPO](/shared/glossary/#ppo) trains one alongside the policy, which roughly doubles the networks held in memory; [GRPO](/shared/glossary/#grpo) skips it entirely by comparing each sampled answer to the group's average instead, which is what makes it cheaper.

### VBench {#vbench}
Comprehensive open evaluation suite for video generation

### Verifier {#verifier}
A program that automatically checks whether an answer is correct — running unit tests, or comparing to a known math result — giving the exact, unhackable reward that [RLVR](/shared/glossary/#rlvr) trains on.

### view {#view}
A no-copy alias that shares storage with its source; requires a contiguous-compatible layout

### VIO {#vio}
Visual-Inertial Odometry — fuse camera and IMU for high-rate ego-motion

### VLA {#vla}
Vision-Language-Action model — transformer mapping image + instruction → action

### vLLM {#vllm}
The reference open-source inference engine with PagedAttention and continuous [batching](/shared/glossary/#batching)

### VLM {#vlm}
Vision-Language Model — image (+ text) in, text out

### Vocabulary {#vocabulary}
The fixed set of tokens a [tokenizer](/shared/glossary/#tokenizer) can produce, each with an integer ID; its size trades tokens-per-document against [embedding matrix](/shared/glossary/#embedding-matrix) size

### Volta {#volta}
NVIDIA's 2017 GPU architecture (V100) and the first generation to ship [Tensor Cores](/shared/glossary/#tensor-core), the dedicated matmul units that made deep-learning training dramatically faster. Subsequent generations — Turing, Ampere, [Hopper](/shared/glossary/#hopper), [Blackwell](/shared/glossary/#blackwell) — kept Tensor Cores and added support for ever-lower-precision formats. Named after the Italian physicist Alessandro Volta.

### VP / VE SDE {#vp--ve-sde}
Variance-Preserving / Variance-Exploding — the two SDE families for diffusion

### VQ-GAN {#vq-gan}
VQ-VAE trained with perceptual + adversarial losses; SD's VAE recipe descends from this

### VQ-VAE {#vq-vae}
Vector-quantized VAE — discrete latent codes from a learned codebook

### Warmup {#warmup}
The opening phase of training where the learning rate ramps up from near zero to its peak, stabilizing the first noisy updates

### Warp {#warp}
32 threads scheduled in lockstep on NVIDIA GPUs

### WBC {#wbc}
Whole-Body Control — fast QP solving for joint torques from task-space goals

### WebDataset {#webdataset}
A library that streams training data directly from sharded `.tar` archives, avoiding the need to unpack millions of individual files.

### Weight decay {#weight-decay}
A regularization technique that shrinks model parameters toward zero at each update step, discouraging large weights and improving generalization

### Weights {#weights}
The learned [parameter](/shared/glossary/#parameters) matrices inside a neural network layer (e.g. the `W` in `y = xW + b`). During training, weights are updated by the [optimizer](/shared/glossary/#optimizer) to minimize the [loss function](/shared/glossary/#loss-function).

### Worker processes {#worker-processes}
Background subprocesses that a [DataLoader](/shared/glossary/#dataloader) spawns to load and preprocess data in parallel with GPU computation.

### Workhorse {#workhorse}
The dependable, go-to method that does the bulk of the everyday work in a field — not the flashiest, but the one practitioners reach for by default because it reliably gets the job done. Just as a workhorse on a farm pulls the heavy loads day in and day out, [PPO](/shared/glossary/#ppo) earned the title in [RLHF](/shared/glossary/#rlhf) and the [PID](/shared/glossary/#pid) controller earned it in robotics.

### World Model {#world-model}
Action-conditioned generative model of the world; a video model with actions

### WSD {#wsd}
Warmup-Stable-Decay — a learning-rate schedule that warms up, holds the rate constant for most of training, then decays sharply at the end.

### XLA {#xla}
Accelerated Linear Algebra — a compiler backend (e.g. for TPUs) used via `torch_xla`

### YaRN {#yarn}
Yet another [RoPE](/shared/glossary/#rope) extensioN method — a context-extension scheme that rescales rotation frequencies unevenly across dimensions to reach long contexts with minimal fine-tuning

### ZeRO {#zero}
[DeepSpeed](/shared/glossary/#deepspeed)'s parameter/gradient/state sharding scheme — comparable to FSDP

### Zero-conv {#zero-conv}
A 1×1 convolution with zero-initialized weights and bias; used by ControlNet to add a branch without disturbing init

### ZMP {#zmp}
Zero-Moment Point — classical biped balance criterion

### σ-schedule (Karras) {#σ-schedule-karras}
The EDM reformulation: parameterize diffusion by noise standard deviation σ rather than discrete timestep

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
