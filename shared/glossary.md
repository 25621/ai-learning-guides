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

### Acceptance rate {#acceptance-rate}
In [speculative decoding](/shared/glossary/#speculative-decoding), the share of the [draft model](/shared/glossary/#draft-model)'s guessed tokens that the big [target model](/shared/glossary/#target-model) agrees with and keeps — `accepted ÷ proposed`. Like a junior writer drafting sentences that the editor either approves or crosses out: the higher the approval rate, the less the editor has to redo and the faster the work goes. Higher acceptance means bigger speedups.

### Activation checkpointing {#activation-checkpointing}
A memory-saving trick that throws away the intermediate [activations](/shared/glossary/#activations) from the forward pass and recomputes them during the [backward pass](/shared/glossary/#backward-pass) — trading a little extra compute for a lot less memory. Also called [gradient checkpointing](/shared/glossary/#gradient-checkpointing).

### Activations {#activations}
The intermediate outputs that flow *between* the layers of a network — the numbers each layer hands to the next during the forward pass. If [weights](/shared/glossary/#weights) are the fixed recipe a model learned, activations are the half-finished dish moving down the kitchen line, changing with every new input. Unlike weights, they are not saved after training; they are recomputed fresh each time the model runs on a new input.

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

### All-to-all token routing {#all-to-all-token-routing}
In a [Mixture-of-Experts (MoE)](/shared/glossary/#moe) model spread across many GPUs, tokens must be sent to the specific GPU that holds the expert they need. "All-to-all" is the massive communication step where every GPU simultaneously sends its tokens to every other GPU and receives tokens in return. Imagine a busy postal sorting center where workers at different tables all throw packages to each other's tables at the exact same time—it requires incredibly fast network connections to prevent a traffic jam.

### AllReduce {#allreduce}
A team operation in distributed computing: every worker ([rank](/shared/glossary/#rank)) starts with its own array of numbers, and AllReduce adds them all together and hands the *same* combined result back to everyone. (A [tensor](/shared/glossary/#tensor) here is just a grid of numbers, not a function; "summing tensors" means lining up two equal-shaped grids and adding matching cells — `[1,2,3] + [10,20,30] = [11,22,33]`.) Imagine four friends who each counted part of a crowd: they pool their counts, add them up, and all walk away knowing the same total. In [tensor-parallel](/shared/glossary/#tensor-parallelism-tp) inference each GPU computes part of a layer, and an AllReduce combines those partial results so every GPU ends up holding the full answer before the next layer runs.

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
The "which one is biggest?" operation: given a list of scores it returns the *position* of the largest one, not the value itself. The name is short for *argument of the maximum* — in math the "argument" is the input you hand a function, so `argmax` answers "which input gives the biggest output?" and returns that input's position. If the [logits](/shared/glossary/#logits) are `[1.2, 4.8, 0.3]`, `argmax` is `1` — the index of `4.8` — which the model reads as "pick token #1." Like scanning a class's test scores and naming the top student rather than reading out their mark. [Greedy decoding](/shared/glossary/#greedy-decoding) is just `argmax` applied to the logits at every step, so it always makes the same choice and never gambles.

### ATen {#aten}
The C++ tensor library underneath PyTorch's Python frontend

### Attention {#attention}
The operation `softmax(QKᵀ/√d) V` — [content-addressable token mixing](/shared/glossary/#content-addressable-token-mixing); the core of every [transformer](/shared/glossary/#transformer). The [softmax](/shared/glossary/#softmax) step turns the raw query–key match scores into weights between 0 and 1 that decide how much each earlier token contributes to the next one.

### Attention sink {#attention-sink}
The first few tokens of a sequence, which [attention](/shared/glossary/#attention) heads keep putting weight on no matter what those tokens actually say. They are called a *sink* in the plumbing sense — a drain where leftover water collects: on every step the [softmax](/shared/glossary/#softmax) has to spread a full 100% of attention across the tokens, so when a head has nothing important to look at, that spare attention drains into these first tokens. Because the model leans on them, [KV cache](/shared/glossary/#kv-cache) eviction schemes deliberately keep these tokens even when they look unimportant, which keeps quality stable in long-context serving.

### autograd {#autograd}
The [reverse-mode](/shared/glossary/#reverse-mode) automatic differentiation engine

### Autoregressive model {#autoregressive-model}
A model that generates a sequence one piece at a time, where each new piece is predicted from all the pieces produced so far — like writing a sentence word by word, with every word depending on the words already on the page. For images, an autoregressive model (such as PixelCNN) draws pixel by pixel in a fixed order. This makes the math clean and the samples sharp, but generation is slow because each step has to wait for the previous one, with no way to compute them all at once.

### AWQ {#awq}
Activation-aware Weight Quantization — preserve weights important to large activations

### Backend {#backend}
A device- or library-specific implementation that actually executes an operation's [kernel](/shared/glossary/#kernel) — for example the CPU, [CUDA](/shared/glossary/#cuda), or [MPS](/shared/glossary/#mps) backend. The [dispatcher](/shared/glossary/#dispatcher) routes each call to the correct backend based on the tensor's device and [dtype](/shared/glossary/#dtype).

### Backward pass {#backward-pass}
The process of going through the network *in reverse* — from the output back to the first layer — to compute [gradients](/shared/glossary/#gradients): how much each [weight](/shared/glossary/#weights) should change to lower the error. It is used **only during training**, right after each [forward pass](/shared/glossary/#forward-pass): the forward pass makes a prediction, the [loss](/shared/glossary/#loss-function) measures how wrong it was, and the backward pass traces that error back to assign blame to each weight (using the chain rule). Like a chef tasting a dish that came out too salty and working backwards through the recipe to figure out which step added too much. A model that is only *serving* answers (inference) never runs the backward pass — that is why serving is cheaper than training.

### Base model {#base-model}
A model fresh out of [pretraining](/shared/glossary/#pretraining) that only continues text and has not yet been taught to follow instructions — a brilliant autocomplete, not yet an assistant.

### Batch {#batch}
A small group of examples (sentences, images, prompts) that the model processes together in a single forward pass instead of one at a time. Like a chef who slices a whole basket of onions at once rather than picking up the knife for each onion separately — the GPU pays a fixed startup cost per pass, so doing 32 examples in one shot is far faster than 32 single passes. In training, the batch size sets how many examples contribute to each gradient update; in [quantization](/shared/glossary/#quantization) methods like [GPTQ](/shared/glossary/#gptq), a small *calibration batch* of representative inputs is run through the model to estimate which weights matter most. See also [Batching](/shared/glossary/#batching), which is the same idea applied to grouping inference *requests* on a serving stack.

### Batching {#batching}
Grouping several inference requests so the GPU runs them together in one forward pass instead of one at a time. Like an elevator that waits a moment to gather a few people and carry them up in a single trip rather than going up and down for each person separately — every rider's start is a touch slower, but far more people move per minute. That is the trade-off batching makes: higher [throughput](/shared/glossary/#throughput) (requests finished per second) at a small cost in [latency](/shared/glossary/#latency) (how long one request waits). Production servers like [Triton Inference Server](/shared/glossary/#triton-inference-server) do this grouping for you automatically; see [continuous batching](/shared/glossary/#continuous-batching) for the version that lets riders hop on and off mid-trip.

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
The smaller, additive group of learned [parameters](/shared/glossary/#parameters) in a layer — the `b` in `y = xW + b`. After the [weights](/shared/glossary/#weights) combine the inputs, each output neuron adds its own bias: a fixed offset that shifts the result up or down no matter what the input was. Like the `+ b` that lets a line `y = mx + b` sit above or below the origin, or a starting balance in a bank account before any transactions — it gives each neuron a baseline to lean toward.

### Blackwell {#blackwell}
NVIDIA's 2024 GPU architecture (B100, B200, B200 Ultra) and the successor to [Hopper](/shared/glossary/#hopper). Like swapping a sports car engine for a more powerful one of the same shape, it keeps the same overall design as Hopper but doubles down on low-precision math — better [FP8](/shared/glossary/#fp8) throughput and brand-new FP4 [Tensor Cores](/shared/glossary/#tensor-core) — which is what makes it the preferred chip for the largest 2025-era training and serving runs.

### BM25 (Best Matching 25) {#bm25}
A classic keyword-search ranking — short for **Best Matching 25** — that scores a document by how often the query's words appear in it, weighting rare words more heavily. Think of it like a librarian scanning pages for your exact search words and ranking pages where those words appear most often (especially unusual words) higher on the list. It is the *sparse* (exact-word) counterpart to dense [embedding](/shared/glossary/#embedding) search.

### Bootstrapping {#bootstrapping}
Using a current estimate (e.g., `V(s')`) in the target instead of a full return

### Bottleneck {#bottleneck}
The single slowest stage in a pipeline, which caps the overall speed; in training this is often the data loader rather than the model.

### bpd (bits per dimension) {#bpd-bits-per-dimension}
The standard likelihood metric for image models: the average number of bits needed to store each number (dimension) in an image, computed as `-log₂ p(x) / D`. Think of it as how "surprised" the model is per pixel-value — a better model predicts the data more confidently and so needs fewer bits, like a good compressor that zips a file smaller. A model that thinks all 256 pixel values are equally likely scores exactly 8 bits per dimension, so any real model must come in under 8 to show it learned something.

### BPE {#bpe}
Byte-Pair Encoding — subword [tokenization](/shared/glossary/#tokenizer) by greedy frequent-pair merges. It starts from raw bytes and repeatedly glues together the neighboring pair that appears most often, building up reusable chunks. For example, on lots of English text BPE notices `t` and `h` sit side by side constantly and merges them into `th`; a later round merges `th` + `e` into `the`. So a common word like `the` ends up as a single token, while a rarer word like `tokenizer` is left as familiar pieces such as `token` + `izer`. "Greedy" means each round simply takes the single most-frequent merge available, never looking ahead to see whether a different choice would pay off later.

### Broadcasting {#broadcasting}
A tensor operation trick where a smaller tensor is automatically stretched to match the shape of a larger one without actually copying data in memory. Like painting a stripe down a wall: you only load the stripe pattern once, but apply it everywhere as you roll.

### C++ extension {#c-extension}
A custom operation written in C++ (optionally with CUDA), compiled and loaded so it can be called from Python like a built-in PyTorch op.

### C-space {#c-space}
Configuration space — the abstract space of joint configurations

### c10 {#c10}
PyTorch's core C++ library (the "core ten[sor]" library)

### Calibration {#calibration}
Running a few representative batches of data through a model to learn how big its [activations](/shared/glossary/#activations) typically get — their usual smallest and largest values — before [quantizing](/shared/glossary/#quantization) it. Knowing that range lets static quantization choose one fixed [int8](/shared/glossary/#int8) "scale": the conversion factor that maps the real numbers onto the 256 slots an int8 can hold. It is like measuring the tallest guest you expect before setting a doorframe height — check the real range once, then size the fixed scale so almost nothing gets clipped.

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

### CIFAR-10 {#cifar-10}
A classic dataset of 60,000 tiny 32×32 color photos sorted into 10 everyday categories (airplane, cat, dog, ship, truck, and so on). Because the images are small and the whole set downloads in seconds, it is a go-to "hello world" for image models — big enough to be interesting, small enough to train on a laptop. The name stands for "Canadian Institute For Advanced Research, 10 classes." See also [MNIST](/shared/glossary/#mnist), its even simpler grayscale cousin.

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

### Context parallelism {#context-parallelism}
Splitting one very long prompt across several GPUs by sequence position, so each GPU holds and processes a different slice of the tokens. Like handing each of four friends one chapter of the same long book to read at the same time, instead of one person reading all four chapters alone. It is how engines serve 100k–1M-token contexts whose [KV cache](/shared/glossary/#kv-cache) would never fit on a single GPU.

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

### Cost per million tokens {#cost-per-million-tokens}
The standard price unit for running a model in production: how many dollars it costs to generate one million tokens of output. You get it by dividing the hardware's hourly cost by how many tokens it produces per hour — like working out a car's cost per mile from its fuel bill and the distance it covers. Almost every serving optimization, from [batching](/shared/glossary/#batching) to [quantization](/shared/glossary/#quantization), is ultimately a way to push this one number down.

### CoT {#cot}
Chain of Thought — prompting or training a model to write out its reasoning step by step before giving a final answer, the way a student shows their work on a math problem instead of blurting out just the result.

### Covariance {#covariance}
A measure of how two quantities move *together*: when one is above its average, does the other tend to be above too (positive covariance), below (negative), or neither (near zero)? Stacked up for many quantities at once it becomes a *covariance matrix*, which describes the overall shape and spread of a [cloud of points](/shared/glossary/#point-cloud) — how wide it is in each direction and how tilted. Picture a scatter of darts on a board: the covariance tells you whether the cloud is a tight circle, a wide oval, or a diagonal streak. [FID](/shared/glossary/#fid) compares the covariances of real and generated image features to check that the two clouds have the same *shape*, not just the same center.

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

### CUDA Graphs {#cuda-graphs}
A way to record a whole sequence of GPU [kernel](/shared/glossary/#kernel) launches once and then replay them all with a single command, instead of telling the GPU what to do step by step every time. Like pressing "play" on a saved macro instead of retyping the same keystrokes — it skips the per-launch bookkeeping. What has to stay fixed is the *list of steps*, not the data they run on: every [decode](/shared/glossary/#decode) step runs the exact same kernels in the exact same order, just on a different token, so it can be recorded once and replayed each step while the actual tokens keep changing. (It only stops helping if the steps themselves change — say a different model path on every call.) This saves a noticeable 5–20% on small models, where launching dozens of tiny kernels per token is itself a real cost.

### CUDA stream {#cuda-stream}
A queue of GPU work that runs in order, but *independently* of other streams — so the GPU can be doing one stream's job while the CPU prepares the next, or two streams can overlap. Like separate checkout lanes at a store: putting independent tasks in different lanes lets them progress at the same time instead of waiting in one long line, which is how a serving stack overlaps [detokenization](/shared/glossary/#detokenization) or KV transfer with the next forward pass.

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
The token-by-token half of LLM inference: after [prefill](/shared/glossary/#prefill) digests the prompt, the model generates one new token per [forward pass](/shared/glossary/#forward-pass), each step reading the whole [KV cache](/shared/glossary/#kv-cache) before producing the next [logits](/shared/glossary/#logits). Like writing a sentence one word at a time while glancing back over every word already written — fast per step, but the constant re-reading of the page is what bounds speed. Decode is [memory-bandwidth-bound](/shared/glossary/#roofline) on a GPU, the opposite of [prefill](/shared/glossary/#prefill), and is what most serving optimizations target.

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

### Distribution drift {#distribution-drift}
When the kind of data a model sees in production slowly changes away from the data it was tuned on — like a store whose regular customers gradually change their tastes, so last year's best-selling stock starts to sit on the shelf. For a [quantized](/shared/glossary/#quantization) model it matters because [calibration](/shared/glossary/#calibration) was fitted to the old traffic, so quality can quietly slip as the new traffic drifts further away.

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

### Draft model {#draft-model}
In [speculative decoding](/shared/glossary/#speculative-decoding), a small, fast model that *guesses* the next few tokens so the big [target model](/shared/glossary/#target-model) can check them all at once. Like a quick assistant who scribbles a rough draft for the expert to approve or correct — cheap to run, and most of its guesses turn out right, so the slow expert is consulted far less often.

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

### Edge inference {#edge-inference}
Running a model directly on the device in front of the user — a phone, laptop, car, or small embedded board — instead of sending the request to a data-center GPU. Like cooking at home rather than ordering delivery: it is private and works without a network, but you are limited to the small "kitchen" the device has, so models are kept small (1–8B), heavily [quantized](/shared/glossary/#quantization), and tuned to sip battery and fit in shared memory.

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

### Enormous on paper {#enormous-on-paper}
Describes a model like a [Mixture-of-Experts (MoE)](/shared/glossary/#moe) that has a massive total number of parameters (e.g., 100 billion), but only uses a small fraction of them (e.g., 10 billion) for any single word. Like a giant university with 5,000 courses listed in its catalog (enormous on paper)—no single student takes all 5,000 courses. Each student only takes a few classes at a time, so the cost per student remains low, even though the total catalog is huge.

### Error budget {#error-budget}
The small amount of failure an [SLO](/shared/glossary/#slo) allows. If your target is 99.9% success, the remaining 0.1% — about 43 minutes a month — is your error budget. Like a monthly data allowance on a phone plan: you can "spend" it on risky deploys and experiments, but once it runs out you stop taking risks until it resets. It turns reliability from a vague goal into a balance you can watch.

### ExecuTorch {#executorch}
PyTorch's lightweight runtime for running models on mobile and edge devices, built on the graph captured by [`torch.export`](/shared/glossary/#torchexport).

### Expert {#expert}
In a [Mixture-of-Experts (MoE)](/shared/glossary/#moe), one of several parallel [MLP](/shared/glossary/#mlp) sub-networks; a router sends each token to only the top few experts instead of all of them. Like a hospital triage desk that routes each patient to the right specialist rather than making everyone see every doctor — lots of expertise on hand, but only a little used per case.

### Expert parallelism (EP) {#expert-parallelism-ep}
For MoE models, distributing experts across GPUs with [all-to-all token routing](/shared/glossary/#all-to-all-token-routing)

### Exponent {#exponent}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that records its *scale* — how many places to shift the decimal point. In scientific notation like `3.5 × 10¹²`, the `12` is the exponent (using base 10 instead of base 2). More exponent bits give a wider range of representable magnitudes, from astronomically large to vanishingly small; fewer exponent bits mean values overflow or [underflow](/shared/glossary/#underflow) more easily. This is why [FP8](/shared/glossary/#fp8) has two flavors: **E5M2** (5 exponent bits) for gradients that can swing wildly in size, and **E4M3** (4 exponent bits) for activations that stay in a tighter range. See also [mantissa](/shared/glossary/#mantissa).

### FCFS {#fcfs}
First-Come, First-Served — the simplest scheduling rule: handle requests in the exact order they arrive, like a single queue at a bakery where nobody can skip ahead. It is fair and easy to build, but it has no sense of deadlines, so one slow request at the front can make everyone behind it late.

### FFN {#ffn}
Feed-Forward Network — the small [MLP](/shared/glossary/#mlp) inside each [transformer](/shared/glossary/#transformer) block. *Position-wise* means it is applied to each token (each position in the sequence) on its own, reusing the **same** weights at every position — like one cashier serving each customer in line one at a time at the same till, never letting them interact. That is the opposite of the [attention](/shared/glossary/#attention) sublayer, where tokens *do* look at each other; the FFN just lets each token "think" by itself.

### F/T sensor {#ft-sensor}
Force/Torque sensor — six-axis force and moment at a wrist or fingertip

### FID {#fid}
Fréchet Inception Distance — the standard sample-quality metric for image generation. It runs both real and generated images through a pretrained [Inception network](/shared/glossary/#inception-network) to turn each image into a feature vector, then measures how far apart the two [clouds](/shared/glossary/#point-cloud) of features sit by comparing their means and [covariances](/shared/glossary/#covariance) (their centers and spreads). A lower FID means the generated images look statistically more like the real ones — picture two overlapping clouds of dots: the more they overlap, the smaller the distance.

### FineWeb-Edu {#fineweb-edu}
A large, openly released [pretraining](/shared/glossary/#pretraining) dataset built by running a [quality filter](/shared/glossary/#quality-filter) over crawled web pages and keeping only the educational-looking ones — like skimming a huge pile of internet text and saving just the pages that read like a textbook. Models trained on it often beat models trained on far more unfiltered text, making it a go-to example that data quality can matter more than raw quantity.

### FK / IK {#fk--ik}
Forward / Inverse Kinematics — compute end-effector pose from joints or vice versa

### FlashAttention {#flashattention}
A much faster way to compute [attention](/shared/glossary/#attention) that never writes the giant token-by-token score table to slow [HBM](/shared/glossary/#hbm) memory. Plain attention builds the full `T × T` grid of how strongly every token attends to every other token, parks it in HBM, then reads it back — a flood of slow memory traffic. FlashAttention instead works on small tiles inside the chip's fast on-chip memory (SRAM) and keeps a running total, so the huge grid never has to be stored at all. Like adding up a long column of numbers in your head as you go instead of writing every subtotal on paper — same answer, far fewer trips to the slow notebook. Every modern inference engine relies on it.

### FlashDecoding {#flashdecoding}
A version of [FlashAttention](/shared/glossary/#flashattention) tuned for the [decode](/shared/glossary/#decode) step, where there is just one new query token but a long [KV cache](/shared/glossary/#kv-cache) to read. It splits that long read across many GPU workers so the [HBM](/shared/glossary/#hbm) bandwidth stays fully used instead of one worker plodding through the cache alone — the trick that lets engines like [vLLM](/shared/glossary/#vllm) hit near-peak bandwidth on decode-heavy traffic.

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

### FP4 {#fp4}
4-bit floating point — half the bits of [FP8](/shared/glossary/#fp8) again, so a weight takes a quarter of the space of [bfloat16](/shared/glossary/#bfloat16). With only 4 bits there are just 16 possible values, so it sits near the edge of usable precision and needs careful checking; newer [Blackwell](/shared/glossary/#blackwell) GPUs accelerate it in hardware, making it attractive for squeezing huge models onto fewer chips.

### Forward hook {#forward-hook}
A callback registered on an `nn.Module` that PyTorch calls automatically after the module's forward pass, receiving the input and output tensors; used for capturing activations and debugging

### Forward pass {#forward-pass}
One complete run of an input through the *whole* network — every layer in order, from the first to the last — to produce an output (for an [LLM](/shared/glossary/#llm), the [logits](/shared/glossary/#logits) for the next token). It means start-to-finish through *all* the layers, not a single layer. Like running a part down an entire assembly line once to get the finished product. The reverse direction, used in training to compute [gradients](/shared/glossary/#gradients), is the [backward pass](/shared/glossary/#backward-pass).

### FP8 {#fp8}
8-bit floating point — half the bits of [bfloat16](/shared/glossary/#bfloat16). Comes in two flavors: **E4M3** (4 [exponent](/shared/glossary/#exponent) bits + 3 [mantissa](/shared/glossary/#mantissa) bits) keeps a bit more precision and is used for [weights](/shared/glossary/#weights) and the forward [activations](/shared/glossary/#activations); **E5M2** (5 [exponent](/shared/glossary/#exponent) + 2 [mantissa](/shared/glossary/#mantissa)) trades precision for a wider range and is used for gradients, which can be very large or very small. Supported by [Hopper](/shared/glossary/#hopper) and later NVIDIA GPUs, it is rapidly becoming the modern default serving precision.

### Fragmentation {#fragmentation}
Memory wasted in gaps too small to reuse, left behind when each request is given its own contiguous chunk — like a parking lot full of single empty spaces where no bus can fit even though there is plenty of total room. Paged schemes such as [PagedAttention](/shared/glossary/#pagedattention) avoid it by handing out small fixed-size pages instead of one big block per request.

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

### GEMM {#gemm}
GEneral Matrix Multiply — the workhorse operation `C = A × B` on two matrices, and the single most common heavy computation inside a neural network. GPUs are built to do GEMMs fast; nearly every layer's forward pass is one. When one input is very "skinny" (a tiny batch, as in single-token [decode](/shared/glossary/#decode)) the GPU's [Tensor Cores](/shared/glossary/#tensor-core) sit half-idle, so that case needs a different kernel from a big, square prefill GEMM.

### GGUF {#gguf}
A single-file format for storing a [quantized](/shared/glossary/#quantization) model — weights plus all the metadata needed to run it — popularized by [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Like a self-contained zip that a laptop or phone can open and run without extra setup, it is the format of choice for [edge and on-device inference](/shared/glossary/#edge-inference).

### Glow {#glow}
A well-known [normalizing flow](/shared/glossary/#normalizing-flow) model (from OpenAI, 2018) that improved on [Real NVP](/shared/glossary/#real-nvp) by adding learnable 1×1 convolutions that shuffle and mix the channels between steps, letting it generate sharp, high-resolution faces. It showed that flows could produce convincing images and smoothly morph one face into another, though they were later overtaken by [diffusion models](/shared/glossary/#diffusion-model) on hard, real-world images.

### GLU {#glu}
Gated Linear Unit — a layer that computes *two* things from the input and multiplies them together element by element: one is the actual content, the other is a "gate" (a [non-linearity](/shared/glossary/#activations) whose output sits near 0–1) that decides how much of that content to let through. Like a row of dimmer switches, one per wire, that the network *learns* to turn up or down — rather than a plain on/off. Being able to suppress parts of its own signal makes a GLU more expressive than a single linear layer; [SwiGLU](/shared/glossary/#swiglu) is the popular variant that uses [Swish](/shared/glossary/#swish) for the gate.

### GPTQ {#gptq}
Short for **Generative Pre-trained Transformer Quantization** — a [post-training quantization (PTQ)](/shared/glossary/#ptq--qat) method that compresses each layer's [weights](/shared/glossary/#weights) row by row, using second-order ([Hessian](/shared/glossary/#hessian)) information to choose the int8 / int4 values that minimize the reconstruction error one layer at a time. Despite the name, GPTQ is not GPT-specific; it works on any [transformer](/shared/glossary/#transformer).

### GQA {#gqa}
Grouped-Query Attention — sharing K/V [heads](/shared/glossary/#heads) across query heads; primary KV-cache saver at serving time

### Gradient accumulation {#gradient-accumulation}
Summing the [gradients](/shared/glossary/#gradients) from several small batches before calling the [optimizer](/shared/glossary/#optimizer), so the update matches a larger effective batch size without its memory cost.

### Gradients {#gradients}

The vector of [partial derivatives](/shared/glossary/#partial-derivative) of a function with respect to each of its parameters. Think of it as a list of slopes telling you exactly the rate of change of the output with respect to each parameter.

**The Intuition**
This combination of direction and size is exactly what an [optimizer](/shared/glossary/#optimizer) follows downhill—much like feeling which way a hillside slopes and how steeply to find the lowest point.

**A Concrete Example**
Consider a simple model `y = w·x + b` with [weight](/shared/glossary/#weights) `w = 2`, [bias](/shared/glossary/#biases) `b = 1`, and input `x = 3`. 
* **Prediction:** `y = 2·3 + 1 = 7`
* **Target:** `10`
* **[Loss](/shared/glossary/#loss-function):** `L = (y - target)² = (7 - 10)² = 9`

**Calculating the Gradient**
Using the chain rule, we can determine the exact rate of change of the Loss (`L`) with respect to `w` and `b`. Since the derivative of `L` with respect to `y` is `2(y - target)`, and the partial derivative of `y` with respect to `w` is `x` (while with respect to `b` it is `1`), the gradients are calculated as follows:
* **For Weight (w):** `∂L/∂w = (∂L/∂y) · (∂y/∂w) = 2(y - target) · x = 2(-3) · 3 = -18`
* **For Bias (b):** `∂L/∂b = (∂L/∂y) · (∂y/∂b) = 2(y - target) · 1 = 2(-3) · 1 = -6`

**Interpreting the Results**
* **Direction (The Sign):** The negative signs indicate that you need to *increase* both `w` and `b` to reduce the error.
* **Magnitude (The Size):** "Bigger" here refers to the absolute value (`|-18| > |-6|`). Even though `-18` is a more negative number than `-6`, its magnitude is larger. This means the weight (`w`) has a much stronger influence on the loss than the bias (`b`).

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

### H2O {#h2o}
Short for *Heavy-Hitter Oracle*, a [KV cache](/shared/glossary/#kv-cache) eviction method that keeps only the handful of past tokens that have been getting most of the [attention](/shared/glossary/#attention) — the "heavy hitters" — and throws the rest away. Like skimming a long book and keeping only the few sentences you keep flipping back to: you save shelf space while barely losing the plot, which lets a model serve much longer sequences in the same memory. It always keeps the very first tokens too (the [attention sink](/shared/glossary/#attention-sink)), since those anchor the model no matter what they say.

### Half-rotation {#half-rotation}
An efficient way to apply [RoPE](/shared/glossary/#rope): rather than rotating each adjacent pair of vector components on its own, you split the vector into two halves and combine them in one shot (the `rotate_half` trick, `[x₁, x₂] → [−x₂, x₁]`). It turns many tiny 2-D rotations into a couple of whole-vector operations, so it runs fast on a GPU while giving the same result.

### Hallucination {#hallucination}
When an [LLM](/shared/glossary/#llm) states something false with the same confident tone it uses for true things — invented citations, made-up people, fabricated facts. Like a student who didn't read the book but answers the essay question anyway in confident prose; the grammar is fine, the facts are not. Hallucination is built in to the [next-token prediction](/shared/glossary/#next-token-prediction) objective, which rewards fluent continuation rather than truth, and is mitigated (not solved) by [RAG](/shared/glossary/#rag), [verifiers](/shared/glossary/#verifier), and abstention training.

### HBM {#hbm}
High-Bandwidth Memory — stacked DRAM on a modern GPU; usually the bandwidth bottleneck

### Headroom {#headroom}
The safety margin you have left before something breaks. In low-precision training it is the spare range of values a number format can still represent before it overflows or rounds down to zero and triggers [numerical issues](/shared/glossary/#numerical-issues) — like the gap between your head and the ceiling: the less you have, the easier it is to bump into trouble. [FP8](/shared/glossary/#fp8) packs numbers into far fewer bits than [bfloat16](/shared/glossary/#bfloat16), so it has much less headroom and is more likely to destabilize a run.

### Heads (attention) {#heads}
The independent, parallel [attention](/shared/glossary/#attention) sub-computations in multi-head attention. Each head operates on its own learned projections of queries, keys, and values, so different heads can latch onto different relationships — one might track which word is the grammatical subject while another tracks what rhymes — and the model attends to several representation subspaces at once. They are called *heads* by analogy to the read/write "heads" of a tape or disk drive: several separate readers scanning the same strip of data in parallel, each pulling out something different. "Multi-head" attention simply runs many such readers side by side and then joins their findings.

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

### Inception network {#inception-network}
A famous image-classification [convolutional neural network](/shared/glossary/#cnn) (the "Inception" / GoogLeNet family) trained on millions of labeled photos. Along the way it learns to boil any image down to a compact *feature vector* — a list of numbers that captures *what is in the picture* (fur, wheels, sky) rather than the raw pixels. Because those features are such good summaries of image content, quality metrics like [FID](/shared/glossary/#fid) reuse a frozen, pretrained Inception network as a fixed yardstick instead of training anything new — like always using the same trusted scale to weigh two bags so the comparison is fair. (It was nicknamed "Inception" after the movie, for its "network inside a network" design.)

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

### Load balancing {#load-balancing}
Spreading incoming requests across several copies of a service so no single one is overwhelmed while others sit idle — like a supermarket opening more checkout lanes and a greeter waving each new customer to the shortest one. The simplest rule is *round-robin* (hand requests out in turn, 1-2-3-1-2-3…); smarter rules send each request to the least-busy replica or to the one whose cache is already warm. The component that does this is a *load balancer*.

### Load shedding {#load-shedding}
Deliberately dropping or rejecting some requests when a server is overloaded, so the ones it does accept still meet their targets. Returning a fast "try again later" to low-priority traffic is far kinder than letting every request crawl — like a busy restaurant turning new walk-ins away so the diners already seated still get served on time. It usually works hand in hand with [admission control](/shared/glossary/#admission-control) and request priority.

### Logits {#logits}
The raw, unnormalized scores a model produces at its output, one per [vocabulary](/shared/glossary/#vocabulary) entry, before they are turned into probabilities by [softmax](/shared/glossary/#softmax). Like the points each contestant has scored at the end of a game — bigger means "more likely the next token" — but to read them as percentages you have to normalize. [Sampling](/shared/glossary/#sampling) rules ([temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), [top-p](/shared/glossary/#top-p)) all reshape the logits before the random draw, and [`argmax`](/shared/glossary/#argmax) of the logits is what [greedy decoding](/shared/glossary/#greedy-decoding) picks.

### LoRA {#lora}
Low-Rank Adaptation — a cheap way to fine-tune a huge model without rewriting it. Instead of changing the model's billions of frozen [weights](/shared/glossary/#weights), you leave them all untouched and bolt on a tiny pair of extra [low-rank](/shared/glossary/#low-rank) matrices that nudge the output. Like leaving a printed textbook exactly as it is and slipping in a few sticky notes that change how you read it: the notes are small to store, quick to write, and you can keep a different set of notes for each task and swap them in and out.

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

### Manifold {#manifold}
The thin, curved surface inside a much larger space where real data actually lives. A 32×32 color image is a point in a space of 3,072 numbers, but almost every random point in that space looks like static — only a vanishingly small, smoothly connected sliver of it looks like a real photo, and that sliver is the manifold. A useful analogy: a sheet of paper is a 2D surface, but if you crumple it and drop it into a room it traces out a thin curved shape floating in 3D space; the paper is the manifold and the room is the full space. Learning to generate images is largely learning the shape of this surface so you only ever land on it.

### Manipulability {#manipulability}
Scalar measure of how "easy" motion is from a given configuration (e.g. `sqrt(det(JJᵀ))`)

### Mantissa {#mantissa}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that holds the *precision digits* — the significant figures sitting in front of the scale factor. In `3.5 × 10¹²`, the `3.5` is the mantissa (also called the *significand*). More mantissa bits give finer resolution between nearby values; fewer mantissa bits leave larger gaps between representable numbers. [FP8](/shared/glossary/#fp8)'s `E4M3` format means 4 [exponent](/shared/glossary/#exponent) bits + 3 mantissa bits, so it can only distinguish about 8 distinct values between each consecutive power of two — coarse, but small enough to fit twice as many numbers in the same memory as [bfloat16](/shared/glossary/#bfloat16).

### Marlin {#marlin}
A specialized GPU [kernel](/shared/glossary/#kernel) for mixed-precision [matmul](/shared/glossary/#matmul) — 4-bit [weights](/shared/glossary/#weights) multiplied by 16-bit [activations](/shared/glossary/#activations) — built to stay fast even on the skinny, small-batch shapes of [decode](/shared/glossary/#decode). It unpacks the 4-bit weights on the fly while keeping the [Tensor Cores](/shared/glossary/#tensor-core) busy, so a [quantized](/shared/glossary/#quantization) model runs nearly as fast as the math allows. (Named after the fast-swimming marlin fish.)

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
Multi-Layer Perceptron — the simplest kind of neural network (also called a [feedforward network](/shared/glossary/#ffn)): a stack of *fully-connected layers* with a *non-linear [activation](/shared/glossary/#activations)* in between. A **fully-connected layer** means every input number connects to every output number, each connection carrying its own [weight](/shared/glossary/#weights) — like a voting panel where every voter influences every result. A **non-linear activation** (such as [ReLU](/shared/glossary/#relu) or [SwiGLU](/shared/glossary/#swiglu)) is a simple bend applied after each layer; you need one because stacking plain linear layers just collapses back into a single straight line, so the bend is what lets the network learn curved, complicated patterns. In a [transformer](/shared/glossary/#transformer), the model is a tall stack of identical **blocks**, and each block has two sublayers in order: an [attention](/shared/glossary/#attention) sublayer (tokens look at each other) then an MLP sublayer (each token is processed on its own). So going up the stack it really does look like *attention → MLP → attention → MLP → …* — attention passes notes around the room, the MLP is each person quietly thinking about what they just read.

Attention comes *before* the MLP because a token should gather context from the others first and only then think for itself — you read the room, then form your own thought. The two are built differently: **attention** has each token build a weighted blend of every token's values (that blend is how they "look at each other"), while the **MLP** runs a plain feed-forward on each token's vector alone, with the same weights at every position (that is "on its own"). The bend inside that MLP has grown more capable over time: plain [ReLU](/shared/glossary/#relu) just clips negatives to 0; a [GLU](/shared/glossary/#glu) instead multiplies the content by a learned 0-to-1 "gate" so the network can dial parts of it down; and [SwiGLU](/shared/glossary/#swiglu) is a GLU whose gate uses the smooth [Swish](/shared/glossary/#swish) curve — the modern default.

### MMDiT {#mmdit}
Multi-Modal Diffusion Transformer — joint text+image attention layers, used in SD3 and Flux

### MMLU {#mmlu}
Massive Multitask Language Understanding — a 57-subject multiple-choice [benchmark](/shared/glossary/#benchmark) (history, law, medicine, math, and more) that became the standard quick test of how much general knowledge a model has, like a giant trivia exam spanning many school subjects at once.

### MNIST {#mnist}
A classic dataset of 70,000 small 28×28 grayscale images of handwritten digits 0–9. It is the most common "hello world" for image models — tiny, clean, and quick to train on — so a brand-new idea is almost always tried on MNIST first, before anyone risks it on harder, fuller-color data like [CIFAR-10](/shared/glossary/#cifar-10).

### Modality gap {#modality-gap}
Empirical finding that different-modality embeddings stay in separable regions

### Mode collapse {#mode-collapse}
GAN failure mode: generator produces few distinct outputs

### MoE {#moe}
Mixture-of-Experts — instead of one big [MLP](/shared/glossary/#mlp) per layer, the model holds many parallel "[expert](/shared/glossary/#expert)" MLPs and a small router sends each token to only the top few. Like a big company where every question goes to just the two or three relevant specialists rather than the whole staff, the model can hold a huge number of total [parameters](/shared/glossary/#weights) while doing only a fixed, small amount of compute per token. The serving catch: which experts get used shifts with the workload, so keeping them evenly busy across GPUs ([expert parallelism](/shared/glossary/#expert-parallelism-ep)) is the hard part.

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
Serving many [LoRA](/shared/glossary/#lora) adapters from one shared copy of the base model at the same time. Keeping LoRA's sticky-note picture: one cookbook (the base model) plus a drawer full of sticky-note sets (the adapters), one set per customer. The kitchen keeps a *single* cookbook and just grabs the right set of notes for each order, instead of buying a whole new cookbook for every customer — so one GPU can serve hundreds of fine-tunes at once.

### Multi-tenant {#multi-tenant}
One shared system serving many independent users or customers ("tenants") at the same time, who must not see or slow down one another — like an apartment building where many families live under one roof but each behind their own locked door. A multi-tenant inference service mixes everyone's requests onto the same GPUs, which is why fair scheduling, per-user rate limits, and tricks like shared-[prefix cache](/shared/glossary/#prefix-cache) routing matter so much.

### Multi-turn conversation {#multi-turn-conversation}
A chat where the user and the AI take turns talking back and forth, building on what was said earlier, like a natural human conversation. For example, if you ask "What's a good movie?" and then ask "Who stars in it?", the AI remembers the movie from the first turn. Instead of starting from scratch every time, the system keeps the past conversation in its [KV cache](/shared/glossary/#kv-cache) — like keeping an open notebook on your desk instead of erasing the whiteboard after every question.

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

### N-gram {#n-gram}
A run of `n` tokens (or words) sitting next to each other. Here a *gram* just means one item — one word or token (the word comes from Greek *gramma*, "something written") — and the *n* says how many of them in a row, so "the cat sat" is a 3-gram (three words in a row) and "cat" on its own is a 1-gram. By matching the most recent few tokens against earlier text, you can often guess what comes next from what followed the same phrase before, which is exactly how prompt-lookup speculative decoding builds its drafts for free.

### nn.Module {#nnmodule}
PyTorch's base class for all neural network components; acts as a registry that automatically tracks sub-modules, parameters, and buffers assigned in `__init__`

### Node (distributed) {#node-distributed}
One physical machine (server) in a distributed job, usually holding several GPUs; multi-node training spreads work across several of them over a network.

### non_blocking {#non_blocking}
The `non_blocking=True` flag on `.to()` / `.cuda()` that lets a host→device copy run asynchronously from pinned memory

### Normalization {#normalization}
Rescaling a layer's outputs so they keep a consistent size — typically zero mean and unit variance (LayerNorm) or unit root-mean-square ([RMSNorm](/shared/glossary/#rmsnorm)). Like adjusting every photo to the same brightness before comparing them, it stops numbers from ballooning or vanishing as they flow through a deep network, which is what keeps training stable.

### Normalizing flow {#normalizing-flow}
A generative model that starts from simple random noise (usually a plain Gaussian "bell curve") and pushes it through a chain of *reversible* steps to reshape it into realistic data — like kneading a smooth ball of dough into a detailed shape, where you can always un-knead it back. Why can you *always* un-knead it? Because every step is deliberately built to be undoable: it only ever stretches, shifts, or folds the dough in a way that has an exact opposite, and it never merges two blobs into one or throws any dough away. For example, if a step's rule is "double this number and add 3," its reverse is simply "subtract 3, then halve" — feed the output back through and you recover the original number exactly, with nothing lost. (An ordinary neural network is *not* like this: it mashes information together — like flattening the dough — so there is no way to run it backwards.) Because every step can be run backwards exactly, a flow can also report the precise probability of any data point, which most generative models cannot do. The price for that exactness is that each step must stay reversible, which heavily constrains the architecture; examples include [Real NVP](/shared/glossary/#real-nvp) and [Glow](/shared/glossary/#glow).

### NVLink {#nvlink}
NVIDIA's GPU-GPU interconnect; much faster than PCIe

### NVSwitch {#nvswitch}
NVLink switch chip; full-bandwidth all-to-all within a node

### Observability {#observability}
The practice of making a running system's inner state visible from the outside — through metrics, logs, and traces — so you can ask new questions about *why* it is misbehaving without adding new code. Like the dashboard and warning lights in a car: you can tell what is wrong while still driving, instead of pulling the engine apart. For a serving stack it is the difference between knowing "p99 [latency](/shared/glossary/#latency) tripled at 9 a.m." and finding out only when users complain.

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
Filling shorter sequences with a placeholder value so that every [sample](/shared/glossary/#sample) in a batch has the same length.

### Parameters {#parameters}
The numbers a model *learns* during training — its adjustable internal settings. Picture thousands of tiny knobs on a giant mixing board: training nudges each knob a little at a time until the whole board produces good output, and the final knob positions *are* what the model "knows." They come in two kinds — [weights](/shared/glossary/#weights) and [biases](/shared/glossary/#biases) — are stored as [tensors](/shared/glossary/#tensor), and are adjusted by the [optimizer](/shared/glossary/#optimizer) during training. (When people say a "7B model," they mean 7 billion of these knobs.) In PyTorch they are `nn.Parameter` objects, registered automatically when assigned to an [`nn.Module`](/shared/glossary/#nnmodule).

### Partial derivative {#partial-derivative}
How much a function changes when you nudge just one of its inputs and hold all the others still — the [derivative](/shared/glossary/#derivative) taken one input at a time. If a recipe's tastiness depends on both salt and sugar, the partial derivative with respect to salt tells you the effect of adding a pinch more salt while keeping the sugar fixed. A [gradient](/shared/glossary/#gradients) is simply the full list of these one-at-a-time slopes, one per [parameter](/shared/glossary/#parameters).

### PagedAttention {#pagedattention}
A way of storing the [KV cache](/shared/glossary/#kv-cache) for many concurrent requests by splitting each request's cache into small fixed-size "pages" that the engine can scatter freely around GPU memory and look up through a per-request page table — the same idea operating systems use for virtual memory. It removes the wasted space and fragmentation you get when each request needs its own contiguous chunk, which is why [vLLM](/shared/glossary/#vllm) made it the default scheme.

### Patchification {#patchification}
Splitting a (latent) tensor into a sequence of patch tokens for a transformer

### PCA (principal component analysis) {#pca-principal-component-analysis}
A technique that finds the few directions along which data varies the most and uses them to compress many numbers down to a handful, so high-dimensional data can be drawn on a 2D plot. Imagine photographing a 3D object from the angle that reveals its shape best — PCA picks that most-informative "camera angle" automatically. It is a quick, standard first step for *seeing* the structure in data, such as checking whether real images cluster together while random noise scatters apart.

### PCIe {#pcie}
The standard CPU-GPU connection (and slower GPU-GPU when no NVLink)

### Percentile {#percentile}
A way to describe where a value ranks in a sorted list: the p99 [latency](/shared/glossary/#latency) is the time that 99% of requests beat, with only the slowest 1% taking longer. Unlike an average, which a single huge outlier can hide, percentiles expose the slow tail that users actually feel — like reporting "even the slowest of the top 99% of diners was served within 20 minutes" instead of a misleading table-wide average. Serving teams quote p50, p95, and p99 rather than the mean for exactly this reason.

### Perceptual loss (LPIPS) {#perceptual-loss-lpips}
Loss computed in the feature space of a pretrained classifier; sharper than pixel MSE

### permute {#permute}
Reorders all of a tensor's dimensions by rewriting strides — never copies

### Perplexity {#perplexity}
A score for how *surprised* a language model is by a piece of text — roughly, how many words it was effectively choosing between at each step. Lower is better: a perplexity of 1 means the model knew exactly what came next, while a high number means it was guessing wildly. Because it is cheap to compute and rises the moment a model gets worse, it is a common first [tripwire](/shared/glossary/#tripwire) in a [quality gate](/shared/glossary/#quality-gate) after [quantization](/shared/glossary/#quantization).

### PID {#pid}
Proportional-Integral-Derivative — the workhorse linear controller

### Pinned memory {#pinned-memory}
Page-locked CPU memory that enables faster, asynchronous transfers to the GPU; enabled with `pin_memory=True` on a [DataLoader](/shared/glossary/#dataloader).

### Pinocchio {#pinocchio}
Fast rigid-body dynamics library (CRBA, RNEA, ABA)

### PixelCNN {#pixelcnn}
An [autoregressive](/shared/glossary/#autoregressive-model) image model that draws a picture one pixel at a time, predicting each pixel from the pixels already drawn above it and to its left — like filling in a coloring grid square by square, always glancing back at what you have already colored to decide the next color. The image quality is strong and it can report an exact [probability](/shared/glossary/#probability-density) for any picture, but generating one is slow because the pixels must come out strictly in order, each waiting on the one before it.

### Plücker coordinates {#plücker-coordinates}
6D representation of a camera ray; standard for camera-conditioning

### PoC {#poc}
Proof of Concept — a small, rough build whose only job is to show that an idea *can* work, before anyone invests in a polished version. Like frying one test pancake to check the batter before making the whole stack: you are not trying to serve it, just to learn whether the approach is sound.

### Point cloud {#point-cloud}
A loose scatter of dots in space, where each dot is one data item placed by its numbers. Turn every image in a batch into a [feature vector](/shared/glossary/#inception-network) — a single point — and the whole batch becomes a cloud of such points. Comparing two clouds (say, real images vs. generated ones) is how a metric like [FID](/shared/glossary/#fid) measures similarity: it is like comparing two swarms of bees and asking whether they are hovering in the same spot and spread out in the same shape.

### Policy {#policy}
In reinforcement learning, the model being trained to choose what to do next — for an [LLM](/shared/glossary/#llm), the network that picks the next token. "Improving the policy" just means making those choices earn more reward.

### Position bias {#position-bias}
A judge's tendency to pick an answer based on *where* it sits rather than *what* it says — for example, an [LLM-as-judge](/shared/glossary/#llm-as-judge) that quietly prefers whichever response appears first (or last) when shown two side-by-side. Like a job interviewer who can't help favoring the candidate they meet right after lunch, regardless of qualifications. The standard fix is to ask the judge twice with the two answers swapped and accept the verdict only if both runs name the same winner.

### Position interpolation {#position-interpolation}
Extending a model's context length by linearly rescaling [RoPE](/shared/glossary/#rope) position indices so longer sequences fall within the trained range

### Posterior collapse {#posterior-collapse}
VAE failure mode: encoder collapses to the prior; latent carries no information

### Postmortem {#postmortem}
A written review done after an incident — an outage, a slowdown — that lays out what happened, how it was detected and fixed, and what will stop it recurring. A good one is *blameless*: it focuses on the system and the process, not on punishing a person, like an air-crash investigation whose goal is safer future flights rather than someone to fire.

### PPO {#ppo}
Proximal Policy Optimization — the [workhorse](/shared/glossary/#workhorse) [on-policy](/shared/glossary/#on-policy) RL algorithm, used in classic RLHF

### Prefill {#prefill}
The first stage of LLM inference: reading the *entire* prompt at once to fill the [KV cache](/shared/glossary/#kv-cache), before any new tokens are generated. Because all the prompt's tokens can be processed together in a single [forward pass](/shared/glossary/#forward-pass), prefill is compute-heavy and fast per token — like a reader skimming a whole page at a glance to grasp it before starting to write a reply. It is the opposite of [decode](/shared/glossary/#decode), which then produces the answer one token at a time, and prefill time is what sets the [time to first token](/shared/glossary/#ttft).

### Prefix cache {#prefix-cache}
Sharing KV cache across requests that begin with the same tokens (e.g., system prompts)

### Pretraining {#pretraining}
Self-supervised training on a large unlabeled corpus to predict the next token

### PRM {#prm}
Probabilistic Roadmap — multi-query sampling-based planner

### PRM800K {#prm800k}
A public dataset of about 800,000 human labels that mark each step of a math solution as right or wrong, released by OpenAI to train [process reward models](/shared/glossary/#process-reward-model). Rather than only checking whether the final answer was correct, human graders read each worked solution line by line — like a math teacher putting a check or an X next to every step of a student's proof, not just the boxed answer at the bottom. Because the feedback is step-level, a model trained on it learns to spot exactly where the reasoning went off the rails instead of whether the ending happened to be lucky. It is the standard training set for the step-by-step scorers used in [Best-of-N](/shared/glossary/#best-of-n) re-ranking.

### Probability density {#probability-density}
A function that says how *likely* each possible value is — high where real data points pile up, low in the empty regions where they rarely fall. For a 2D dataset you can picture it as a heatmap: bright ridges over the crowded spots, dark valleys over the bare ones. It must stay non-negative everywhere, and all of it added up (the total volume under the surface) equals exactly 1, since *some* value always occurs. Most generative models can only *draw* new samples; a [normalizing flow](/shared/glossary/#normalizing-flow) is special because it can also report the exact probability density of any point you hand it.

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

### Quality gate {#quality-gate}
An automatic check that a model must pass *before* it is allowed to serve real traffic — like a bouncer at the door who turns away anyone failing the dress code. It runs a fixed set of evaluations (such as [perplexity](/shared/glossary/#perplexity) and capability tests) and blocks the deploy if any score drops too far from the trusted baseline, which is how teams catch silent [quantization](/shared/glossary/#quantization) regressions before users do.

### Quantization {#quantization}
Reducing weight / activation precision (FP16, BF16, FP8, INT8, INT4) to save memory and bandwidth

### RadixAttention {#radixattention}
sglang's KV cache organized as a radix tree keyed on prompt prefixes for automatic sharing

### RAG {#rag}
Retrieval-Augmented Generation — give the model an "open-book exam" instead of asking it to answer from memory alone. First a search step fetches the documents most relevant to the question (from a company wiki, a manual, the web), then those documents are pasted into the prompt, and only then does the model write its answer using them as notes. This lets it use fresh or private facts it was never trained on, and makes it easy to check where an answer came from.

### rank {#rank}
The unique integer ID of a process in a distributed job. `RANK` is the global ID across all machines; `LOCAL_RANK` is the ID within one machine; `WORLD_SIZE` is the total number of processes.

### RDMA {#rdma}
Remote Direct Memory Access — letting one machine read or write another machine's memory directly over the network, without either CPU stopping to copy the data. Like a pneumatic tube that drops a package straight onto a coworker's desk instead of handing it to a courier who walks it over. In disaggregated serving it is how a [prefill](/shared/glossary/#prefill) node ships a multi-gigabyte [KV cache](/shared/glossary/#kv-cache) to a [decode](/shared/glossary/#decode) node fast enough to be worth splitting them.

### Real NVP {#real-nvp}
Short for "Real-valued Non-Volume Preserving" — an early, influential [normalizing flow](/shared/glossary/#normalizing-flow) design. Its trick at each step: split the numbers into two halves, leave one half completely untouched, and use that untouched half to decide how to stretch and shift the other half. Because the untouched half is still right there, the step is trivially *reversible* (you can recompute the stretch-and-shift and undo it) and its effect on [probability density](/shared/glossary/#probability-density) is cheap to calculate. This made flows practical to train and inspired later models like [Glow](/shared/glossary/#glow).

### Reasoning model {#reasoning-model}
An [LLM](/shared/glossary/#llm) trained to think out loud at length — writing a long [chain of thought](/shared/glossary/#cot) before its final answer — to solve harder problems (math, code, logic). Like a student who fills a page of scratch work before writing the answer, it is far more capable on tough questions but also far more expensive to serve, because one hard problem can produce 10× the tokens of a normal chat reply. Managing that swing in output length is the main serving challenge it creates.

### ReAct {#react}
A simple [agent](/shared/glossary/#agent) pattern that interleaves **Rea**soning and **Act**ing: the model writes a thought, takes an action with a tool, reads the observation, then repeats — the loop most basic agents are built on.

### Reciprocal rank fusion {#reciprocal-rank-fusion}
A simple, robust way to merge several ranked lists into one: each item scores the sum of `1 / (rank + constant)` across the lists, so items ranked highly by more than one retriever rise to the top. Common for combining dense and sparse search in [hybrid retrieval](/shared/glossary/#hybrid-retrieval).

### Rectified flow {#rectified-flow}
A flow-matching parameterization with straight-line trajectories; popular in 2024+ models

### Reference model {#reference-model}
A frozen copy of the starting model that [RLHF](/shared/glossary/#rlhf) and [DPO](/shared/glossary/#dpo) measure against (through a [KL](/shared/glossary/#kl-divergence) term) so the model being trained does not drift too far from sensible behavior — a "before" photo to compare every change against.

### Rejection sampling {#rejection-sampling}
A way to draw samples from a target distribution by proposing easy guesses and keeping or throwing away each one with just the right probability, so the survivors are distributed exactly as if they came from the hard distribution directly. The "right probability" of keeping a guess is `min(1, p ÷ q)`, where `p` is how likely the [target model](/shared/glossary/#target-model) thinks that token is and `q` is how likely the [draft model](/shared/glossary/#draft-model) thought it was. The rule is intuitive: if the target wants the token at least as much as the draft did (`p ≥ q`), always keep it; if the target wants it only half as much (`p` is half of `q`), keep it half the time and otherwise draw a replacement. For example, the draft proposes "cat" with `q = 0.6` but the target only gives it `p = 0.3`, so you keep "cat" with probability `0.3 ÷ 0.6 = 0.5` — a coin flip — which exactly cancels the draft's over-eagerness for that word. In [speculative decoding](/shared/glossary/#speculative-decoding) this is the step that lets a draft model's guesses be reused for random [sampling](/shared/glossary/#sampling) without changing the target model's true output distribution.

### ReLU {#relu}
Rectified Linear Unit — the most common and simplest [activation function](/shared/glossary/#activations): it keeps positive numbers unchanged and turns every negative number into 0 (`max(0, x)`). Like a one-way valve that lets water through in one direction and blocks it in the other. That single sharp bend is enough to give a network its non-linear power, and because it is so cheap to compute it was the default for years; newer models often swap it for smoother curves like [Swish](/shared/glossary/#swish) or [GELU](/shared/glossary/#gelu).

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

### Right-sizing {#right-sizing}
Choosing the smallest, cheapest model that still clears your quality bar for a task, instead of defaulting to the biggest one available. A well-trained 8B model often passes the same eval as a 70B at a fraction of the [cost per million tokens](/shared/glossary/#cost-per-million-tokens) — like hiring a capable specialist instead of an expensive all-rounder for a job that doesn't need one. Most production teams over-serve, so right-sizing is one of the easiest cost wins.

### Ring attention {#ring-attention}
A way to run [attention](/shared/glossary/#attention) over a very long sequence that is split across several GPUs ([context parallelism](/shared/glossary/#context-parallelism)): each GPU passes its slice of the keys and values to its neighbor around a circle, round after round, until every GPU has seen every other slice. Like people seated around a dinner table passing dishes one seat at a time so everyone eventually tastes every dish. This lets the GPUs handle a sequence far longer than any one of them could hold alone.

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

### Router model {#router-model}
A small, cheap model that sits at the front of a serving stack and decides *which* model should answer each request — for example, sending an easy question to a fast 1B model and only escalating hard ones to a slow, expensive 70B model. Like a hospital triage nurse who handles simple cases on the spot and forwards the serious ones to a specialist, it saves money because most queries never need the biggest model.

### RRT {#rrt}
Rapidly-exploring Random Tree — single-query sampling-based planner

### SAC {#sac}
Soft Actor-Critic — maximum-entropy continuous-control algorithm; the modern default

### SAE {#sae}
Sparse Autoencoder — interpretability tool decomposing activations into monosemantic features

### Sample {#sample}
A single example in a dataset or [batch](/shared/glossary/#batch) — one sentence, one image, one prompt. If a batch is a carton of eggs, a sample is one egg. The word can confuse beginners because [sampling](/shared/glossary/#sampling) in text generation means something else entirely (randomly drawing the next token); here it simply means "one item."

### Sampler {#sampler}
The component that decides the order in which a [DataLoader](/shared/glossary/#dataloader) visits dataset examples (e.g. random, sequential, or class-weighted).

### Sandbox {#sandbox}
An isolated, throwaway environment — like a fenced-off playground — where an [agent](/shared/glossary/#agent) or program can run commands, create files, and make mistakes without affecting your real computer. If the agent breaks something inside the sandbox, you just throw the sandbox away; nothing outside it is touched. Containers (like Docker) and virtual machines are common ways to build one.

### Sampling {#sampling}
Drawing the next token from the model's predicted probability distribution instead of always taking the most likely one; [temperature](/shared/glossary/#temperature), [top-k](/shared/glossary/#top-k), and [top-p](/shared/glossary/#top-p) control how random the choice is.

### Scaling laws {#scaling-laws}
The empirical finding that a model's [loss](/shared/glossary/#loss-function) drops in a smooth, predictable curve as you add [parameters](/shared/glossary/#parameters), training data, and compute — like a growth chart that lets you forecast a bigger model's quality from smaller ones before you ever build it.

### Scheduler {#scheduler}
The part of an inference server that decides, at every step, which requests to start, which to keep generating, and which to pause when memory runs low — like an air-traffic controller choosing which planes take off, keep flying, or circle, so the runway (the GPU) is always busy but never overloaded. A good scheduler is often worth more real-world [throughput](/shared/glossary/#throughput) than any single clever kernel.

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

### SiLU {#silu}
Sigmoid Linear Unit — just another name for [Swish](/shared/glossary/#swish), the activation `x · σ(x)`. The two words mean the exact same function: you will see "SiLU" in code (PyTorch's `nn.SiLU`) and "Swish" in papers.

### SIMT {#simt}
Single Instruction Multiple Threads; NVIDIA's execution model

### SLAM {#slam}
Simultaneous Localization and Mapping

### SLI {#sli}
Service Level Indicator — the actual measured number for how well a service is doing, such as the real percentage of requests that succeeded or answered within 500 ms. The [SLO](/shared/glossary/#slo) is the *target*; the SLI is the *measurement* you compare against it — like the speedometer reading (SLI) versus the posted speed limit (SLO).

### SLO {#slo}
Service Level Objective — a specific, measurable promise about how a service should perform, such as "p95 [TTFT](/shared/glossary/#ttft) under 500 ms" or "99.9% of requests succeed." It is the target you design toward and get alerted on, like a delivery company promising most parcels arrive within two days. The measured reality you check it against is the [SLI](/shared/glossary/#sli), and the slack it allows for failure is the [error budget](/shared/glossary/#error-budget).

### SM {#sm}
Streaming Multiprocessor; the GPU's "core"

### softmax {#softmax}
The function that turns a vector of scores into a probability distribution — each value squeezed into 0–1, and all of them summing to 1; the core of [attention](/shared/glossary/#attention) and classification heads. The name means a *soft* version of `max`: instead of the hard "winner takes all" of [argmax](/shared/glossary/#argmax), which hands the single biggest score 100% and the rest nothing, softmax gives most of the weight to the biggest score while still leaving a little for the others. That smoothness — a dimmer switch rather than an on/off toggle — is what lets the model be trained by gradients.

### Special tokens {#special-tokens}
Reserved [vocabulary](/shared/glossary/#vocabulary) entries that mark structure rather than text — e.g. `<bos>`, `<eos>`, `<pad>`, and chat-boundary tokens like `<|im_start|>`

### Speculative decoding {#speculative-decoding}
A trick to make [decode](/shared/glossary/#decode) faster for free: a small, fast "draft" model guesses the next few tokens, and the big "target" model checks all of them in a single parallel pass, keeping every guess that matches what it would have produced and discarding the rest. Like an editor who reads a sentence a junior writer drafted and approves the part that is already correct rather than writing every word from scratch — the answer is identical to what the target alone would say, just reached in fewer slow steps. It works because decode is starved for memory bandwidth, so the GPU has spare compute to verify several guesses at once.

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
The activation used in most modern transformer [MLPs](/shared/glossary/#mlp): a [GLU](/shared/glossary/#glu) gate whose non-linearity is [Swish](/shared/glossary/#swish), written `(xW) · Swish(xV)`. In plain terms, the input is projected two ways — one path is the content, the other is squeezed through Swish to become a soft gate — and the two are multiplied so the gate dials each value up or down. It replaced plain [ReLU](/shared/glossary/#relu) feed-forward layers because, for the same size, it tends to learn a little better; it is the default [FFN](/shared/glossary/#ffn) in Llama-style models.

### Swish {#swish}
A smooth [activation function](/shared/glossary/#activations), `x · σ(x)`, also called [SiLU](/shared/glossary/#silu). It does roughly the same job as [ReLU](/shared/glossary/#relu) — squashing large negatives toward 0 and passing positives through — but with a gentle curve instead of a sharp corner (and it even dips a little below 0 for small negatives before recovering). Think of a soft-closing drawer that eases shut instead of slamming at exactly zero: that smoothness gives the network cleaner [gradients](/shared/glossary/#gradients) to learn from. It is the non-linearity used as the gate inside [SwiGLU](/shared/glossary/#swiglu).

### System prompt {#system-prompt}
A message placed at the very start of a chat conversation that tells the model how to behave — its role, tone, rules, and the tools it can call — before the user's first turn ever arrives. Like a stage director's note to an actor before the curtain rises: *"You're a polite customer-support agent who answers only refund questions."* System prompts are usually long and shared across many requests, which is why caching their KV state (see [prefix cache](/shared/glossary/#prefix-cache)) saves so much repeated work.

### Systolic array {#systolic-array}
Data-flow matmul fabric used in TPUs

### T2V {#t2v}
Text-to-Video

### Tail latency {#tail-latency}
The [latency](/shared/glossary/#latency) of the slowest requests (for example the p95 or p99 percentiles) rather than the median (p50); it is what users notice most.

### Target model {#target-model}
In [speculative decoding](/shared/glossary/#speculative-decoding), the big, accurate model whose output you actually want — it checks the small [draft model](/shared/glossary/#draft-model)'s guesses and has the final say on every token. Like the senior editor who must approve the assistant's draft: slow and expensive to consult, so the trick is to bother it as rarely as possible while still letting it decide the real answer.

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
A grid of numbers — the basic container deep learning uses for almost everything. A single number is a 0-D tensor, a list of numbers is 1-D (a *vector*), a table is 2-D (a *matrix*), and you can keep stacking into 3-D and beyond — for example a color image is a 3-D tensor of height × width × 3 color channels. Under the hood it is a (storage, shape, stride, offset, dtype, device, requires_grad) tuple viewing a 1-D [storage](/shared/glossary/#storage) buffer, but the everyday idea is simply "an N-dimensional array of numbers the GPU can crunch in parallel."

### Tensor Core {#tensor-core}
Specialized matmul unit in NVIDIA GPUs since [Volta](/shared/glossary/#volta)

### Tensor parallelism (TP) {#tensor-parallelism-tp}
Splitting each layer's [weights](/shared/glossary/#weights) across several GPUs so they each do part of the math, then combining their partial results with an [all-reduce](/shared/glossary/#allreduce). "At attention/[MLP](/shared/glossary/#mlp) boundaries" means that combining happens at two natural seams in every [transformer](/shared/glossary/#transformer) block — once at the end of the [attention](/shared/glossary/#attention) sublayer and once at the end of the MLP sublayer — because *within* a sublayer the GPUs can work independently, but at its edge their pieces must be added back together before the next step can start. Like four cooks each preparing part of a dish and merging everything at two fixed points before it moves on.

### TFLOPs {#tflops}
Tera (10¹²) floating-point operations per second

### TGI {#tgi}
Short for **Text Generation Inference** — Hugging Face's open-source LLM serving engine, similar in role to [vLLM](/shared/glossary/#vllm). It implements [continuous batching](/shared/glossary/#continuous-batching), [PagedAttention](/shared/glossary/#pagedattention), and quantized inference behind a simple HTTP API, and is one of the two engines most commonly used to put an LLM in front of real users.

### Thinking budget {#thinking-budget}
A cap on how many tokens a [reasoning model](/shared/glossary/#reasoning-model) is allowed to spend thinking before it must give an answer — like telling a student "you have ten minutes of scratch work, then write your answer." It lets a serving system trade accuracy for cost and [latency](/shared/glossary/#latency): a bigger budget usually means better answers on hard problems but slower, pricier responses.

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

### Tool call {#tool-call}
When an [LLM](/shared/glossary/#llm), instead of answering directly, emits a structured request to run an external function — search the web, query a database, run code — and then continues once it sees the result. Like a person pausing mid-task to look something up or use a calculator, it is the basic action an [agent](/shared/glossary/#agent) takes in its loop.

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

### Trained or just prompted {#trained-or-just-prompted}
A choice in how you create a small helper model (like a [router model](/shared/glossary/#router-model)). You can either "train" it (by fine-tuning its weights on thousands of examples, like sending someone to medical school) or "just prompt" it (by taking an existing smart model and simply giving it a written instruction like "You are a router, decide if this question is hard or easy," like handing a smart assistant a checklist). Training takes more effort upfront but is faster and cheaper to run; prompting is quick to set up but costs more per request since you process the instructions every time.

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

### Tripwire {#tripwire}
A cheap, fast check whose only job is to sound the alarm the instant something goes wrong — named after the thin wire that, when stepped on, sets off a trap or flare. In model deployment a quick metric like [perplexity](/shared/glossary/#perplexity) is used as a tripwire: it won't tell you *what* broke, but it spikes the moment quality drops, so it catches a bad build before the slower, fuller tests even run.

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
The main, larger group of learned [parameters](/shared/glossary/#parameters) in a layer — the `W` in `y = xW + b` — that decide how strongly each input affects each output. Think of the volume sliders on a soundboard: a big weight turns an input way up, a near-zero weight mutes it, and a negative weight flips it. During training the [optimizer](/shared/glossary/#optimizer) keeps nudging these sliders to lower the [loss](/shared/glossary/#loss-function), and they make up the bulk of a model's size.

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
