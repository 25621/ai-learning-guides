# Glossary

Terms from all guides in this repository, sorted alphabetically. Each guide's own concepts are included here; see individual guides for deeper context.

---

### (2+1)D {#21d}
A way to build a video network cheaply by *factorizing* a full 3D convolution (which mixes space and time at once) into two smaller steps: first a 2D spatial layer that processes each frame on its own, then a separate 1D temporal layer that mixes information across time at each pixel location. The name reads "2 plus 1": two spatial dimensions handled together, plus one time dimension handled separately. Splitting them this way costs far less compute than a true 3D layer and — crucially — lets you initialize the spatial half from a pretrained image model and add the temporal half fresh (see [temporal inflation](/shared/glossary/#temporal-inflation)). The trade-off is that space and time never interact *within* a single layer, which can miss fast, complex motion that a full spatiotemporal layer would catch.

### 3D VAE {#3d-vae}
A [VAE](/shared/glossary/#vae) (variational autoencoder — a network that squeezes data into a small code and reconstructs it) built for video, so it compresses along *time* as well as the two spatial dimensions. A plain image VAE shrinks each frame's height and width; a 3D VAE *also* merges groups of nearby frames, exploiting the fact that consecutive frames barely differ. Typical ratios are about 4× in time and 8× in each spatial direction, cutting a clip's data by roughly 100× overall (the spatial compression applies to both height and width, so it shrinks 4 × 8 × 8 = 256 times in size, but the number of channels usually grows, e.g., from 3 RGB channels to 8 latent channels, so the final footprint is about 100× smaller). This is what makes modern video diffusion affordable: the [diffusion model](/shared/glossary/#diffusion-model) runs on the small compressed [latent](/shared/glossary/#latent-space) grid instead of raw pixels, so it sees maybe 30 latent "frames" where the original clip had 120. Because the same compressor is reused for every clip, the heavy work of learning to reconstruct video is paid once while the VAE is trained, not on every generation.

### ABA {#aba}
Articulated-Body Algorithm — `O(n)` forward dynamics for rigid-body chains

### Ablation {#ablation}
A controlled experiment that changes exactly one factor (a data step, a layer, a hyperparameter) while holding everything else fixed, to measure that factor's true effect.

### Acceptance rate {#acceptance-rate}
In [speculative decoding](/shared/glossary/#speculative-decoding), the share of the [draft model](/shared/glossary/#draft-model)'s guessed tokens that the big [target model](/shared/glossary/#target-model) agrees with and keeps — `accepted ÷ proposed`. Like a junior writer drafting sentences that the editor either approves or crosses out: the higher the approval rate, the less the editor has to redo and the faster the work goes. Higher acceptance means bigger speedups.

### Action conditioning {#action-conditioning}
Telling a generative video model not just *what* scene to show but *what just happened to it* — you feed in an extra input encoding an action (a button press, a steering angle, a chosen game move) alongside the current frames, and the model predicts the frames that should follow *that* action. Turning the generic "predict the next frames" task into "predict the next frames **given this action**" is exactly what turns an ordinary video generator into a [world model](/shared/glossary/#world-model): the action becomes the knob a user (or a [policy](/shared/glossary/#policy)) turns to steer what happens next. Like the difference between a film that simply plays and a video game that reacts to the controller in your hands.

### Activation checkpointing {#activation-checkpointing}
A memory-saving trick that throws away the intermediate [activations](/shared/glossary/#activations) from the forward pass and recomputes them during the [backward pass](/shared/glossary/#backward-pass) — trading a little extra compute for a lot less memory. Also called [gradient checkpointing](/shared/glossary/#gradient-checkpointing).

### Activations {#activations}
The intermediate outputs that flow *between* the layers of a network — the numbers each layer hands to the next during the forward pass. If [weights](/shared/glossary/#weights) are the fixed recipe a model learned, activations are the half-finished dish moving down the kitchen line, changing with every new input. Unlike weights, they are not saved after training; they are recomputed fresh each time the model runs on a new input.

### AdaGN (adaptive group normalization) {#adagn-adaptive-group-normalization}
The trick diffusion models use to push a condition — like a class label or the current denoising step — into the network through its [normalization](/shared/glossary/#normalization) layers. First *group normalization* wipes a group of [activations](/shared/glossary/#activations) clean to mean 0 and variance 1, erasing their current style; then a *tiny layer* — a single small linear layer, just one little matrix of [weights](/shared/glossary/#weights) rather than a deep stack of layers — reads the condition and predicts two numbers, a **scale** and a **shift** ([scale-and-shift](/shared/glossary/#scale-and-shift)), that re-stretch and re-center those activations. The layer can stay this small because its only job is to translate the condition into those two knobs — not to do any heavy image work — so a lightweight layer is plenty, and being tiny means it costs almost nothing even when one is dropped in at every normalization layer. Picture resetting a photo to neutral brightness and contrast, then letting the label "cat" turn those two knobs to a setting the model learned for cats. It is the group-norm cousin of [AdaIN](/shared/glossary/#adaptive-instance-normalization-adain) and [AdaLN](/shared/glossary/#adaln), and it is how a single model can be steered to generate one chosen class on demand. The [**adaptive**](/shared/glossary/#adaptive) part is exactly this: those two knobs are not frozen constants but are re-predicted for whatever condition you hand in, so the layer re-tunes itself for "cat" versus "dog" instead of behaving the same way every time.

### AdaLN {#adaln}
[**Adaptive**](/shared/glossary/#adaptive) layer normalization; the conditioning mechanism in [DiT](/shared/glossary/#dit). It is the [layer-normalization](/shared/glossary/#normalization) cousin of [AdaGN](/shared/glossary/#adagn-adaptive-group-normalization) and [AdaIN](/shared/glossary/#adaptive-instance-normalization-adain): a normalization layer first wipes a block's [activations](/shared/glossary/#activations) to a neutral mean-0, variance-1 state, then a tiny layer reads the condition (usually the current denoising step plus a class label) and predicts a fresh [scale-and-shift](/shared/glossary/#scale-and-shift). It is called *adaptive* because those scale and shift numbers are not fixed once and reused — they are re-computed for each condition, so the layer adapts itself to whatever you are asking it to generate. See also [AdaLN-Zero](/shared/glossary/#adaln-zero).

### AdaLN-Zero {#adaln-zero}
The conditioning trick that powers [DiT](/shared/glossary/#dit). Start with plain **AdaLN** (*adaptive layer normalization*): a normalization layer first wipes a block's [activations](/shared/glossary/#activations) to a neutral mean-0, variance-1 state, then a *tiny layer* reads the condition — usually the current denoising step plus a class label — and predicts a fresh [scale-and-shift](/shared/glossary/#scale-and-shift) to re-stretch and re-center them, the [layer-normalization](/shared/glossary/#normalization) cousin of [AdaGN](/shared/glossary/#adagn-adaptive-group-normalization) and [AdaIN](/shared/glossary/#adaptive-instance-normalization-adain). The **"-Zero"** part makes two changes. First, the tiny layer predicts a *third* number — a **gate** — that multiplies the whole block's output before it is added back onto the [residual stream](/shared/glossary/#residual-stream) (a [gated](/shared/glossary/#gated) path). Second, it *initializes the layer that produces scale, shift, and gate so they all start at zero*. With the gate at zero, every [transformer](/shared/glossary/#transformer) block contributes *nothing* at the very start of training — the input slides straight through untouched, exactly like the identity shortcut of a [residual connection](/shared/glossary/#residual-connection). As training proceeds the gate gradually lifts off zero, so each block learns to add its effect *gently* instead of jolting a fragile, freshly-initialized network. Picture adding a new musician to a band but starting them on mute: you slowly turn up their volume from zero as they learn the song, rather than letting them blast over everyone on the first beat. That gentle start is a big part of why deep DiTs train stably.

### Adam {#adam}
Adaptive Moment Estimation — gradient-descent optimizer that maintains per-parameter running averages of the first (mean) and second (uncentered variance) moments of the gradients to compute individual adaptive learning rates.

### AdamW {#adamw}
Adam optimizer with decoupled weight decay: the regularization term shrinks the parameter directly rather than being folded into the gradient update

### Adapter {#adapter}
A small trainable layer slipped into an otherwise [frozen](/shared/glossary/#frozen) pretrained network so it can pick up a new skill — or accept a brand-new input [modality](/shared/glossary/#modality) — without the cost of retraining the whole thing. The usual design is a *bottleneck*: squeeze the incoming features down to a tiny dimension, push them through a nonlinearity, expand them back to the original size, and add the result onto the untouched main path, so at the start the adapter outputs almost nothing and only gradually learns its small correction. Like a travel plug adapter that lets your existing appliance work in a foreign socket — the appliance (the big pretrained model) is left exactly as it is, and the cheap little adapter does all the converting. Example: bolt an adapter onto a frozen [CLIP](/shared/glossary/#clip) image encoder so it can suddenly read [depth maps](/shared/glossary/#depth-map), training only the adapter's handful of [weights](/shared/glossary/#weights) while millions of frozen ones stay put. [LoRA](/shared/glossary/#lora) is one popular variety of adapter.

### Adaptive {#adaptive}
"Adaptive" means a layer does not use one frozen setting for every input — it *adjusts its setting on the fly* based on what you ask for. In a plain [normalization](/shared/glossary/#normalization) layer the [scale-and-shift](/shared/glossary/#scale-and-shift) knobs are learned once during training and then locked: every single input gets the exact same two numbers, like a radio permanently soldered to one station. The word **adaptive** flips that. Instead of fixed knobs, a *tiny layer reads a condition you hand in* — a style code, a class label like "cat," or the current denoising step — and **predicts fresh knobs for that specific input**, so the same layer re-tunes itself every time. Picture the difference between an old thermostat bolted to 70°F no matter who walks in, and a smart thermostat that reads the room — who is home, the time of day, the weather — and picks a new target temperature on its own. Same machine, but its behavior *adapts* to the situation. That is exactly what the "Ada" stands for in [AdaGN](/shared/glossary/#adagn-adaptive-group-normalization), [AdaIN](/shared/glossary/#adaptive-instance-normalization-adain), and [AdaLN](/shared/glossary/#adaln): the scale and shift are predicted from the condition instead of staying frozen, so the network adapts to whatever you are generating right now.

### Adaptive instance normalization (AdaIN) {#adaptive-instance-normalization-adain}
A way to push a "style" into a network's features: first normalize a feature map so it has mean 0 and variance 1 (wiping out its current style), then rescale and shift it using two numbers — a scale and a bias — predicted from a style code. Like erasing a drawing down to a plain pencil outline and then re-coloring it from a palette you hand in. [StyleGAN](/shared/glossary/#stylegan) applies AdaIN at every layer so a single style code can steer image features at every scale. The [**adaptive**](/shared/glossary/#adaptive) in the name means the scale and bias are not baked-in constants: they change with each style code you provide, so the very same layer re-paints features differently for every style instead of applying one fixed look.

### ADD (Adversarial Diffusion Distillation) {#add-adversarial-diffusion-distillation}
The recipe behind few-step models like SDXL Turbo: [distill](/shared/glossary/#distillation) a slow multi-step [diffusion model](/shared/glossary/#diffusion-model) into a 1–4-step student, but add a [GAN](/shared/glossary/#gans)-style [discriminator](/shared/glossary/#discriminator) that judges whether the student's quick output looks real. Plain distillation alone makes few-step images blurry, because regressing toward an average washes out detail; the discriminator punishes that blur and forces crisp results. Like training a sprinter to copy a marathoner's route in a fraction of the strides while a sharp-eyed judge rejects any shortcut that looks sloppy — so speed rises without the output going soft. Compared with an [LCM](/shared/glossary/#lcm) (pure consistency distillation), ADD trades a fiddlier training setup for sharper few-step samples.

### Admission control {#admission-control}
Refusing requests early when capacity is saturated, to protect SLOs for accepted requests

### Advantage {#advantage}
`A(s, a) = Q(s, a) − V(s)` — how much better than the baseline this action is

### Aesthetic score {#aesthetic-score}
A single number predicting how visually *pleasing* a human would find an image — used to judge generators when realism metrics like [FID](/shared/glossary/#fid) miss the question of "is it beautiful?" You produce it with a small predictor — usually a tiny linear head bolted on top of frozen [CLIP](/shared/glossary/#clip) image [embeddings](/shared/glossary/#embedding) — that has been fit to a dataset of images people rated on a 1–10 scale (the LAION-Aesthetics predictor is the best-known example). To score a new image you embed it with CLIP and pass that embedding through the trained head; the output approximates the average rating a person would give. Think of a film critic who has watched thousands of movies alongside their audience scores and can now glance at a new one and guess its rating. Because it is a learned proxy for taste, it inherits the biases of whoever did the original rating.

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

### Alt-text {#alt-text}
The short text description attached to an image in a web page's HTML so screen readers can announce it and so the text still shows if the picture fails to load (the "alt" is short for *alternative* text). Because it sits right next to billions of web images, alt-text is the free, ready-made caption that web-scraped datasets like [LAION](/shared/glossary/#laion) use as each image's label — which is why it is the raw material the whole multimodal-data pipeline starts from. The catch is that it was written for accessibility, not for training: it is often missing, a bare filename like "IMG_2025.jpg", keyword spam stuffed in for search ranking, or simply unrelated to the picture — which is exactly why pipelines filter it by [CLIP](/shared/glossary/#clip) score and rewrite it into [synthetic captions](/shared/glossary/#synthetic-captions). Like the one-line label taped to the back of a photo in a shoebox: handy when it is accurate, useless when someone scribbled the wrong date.

### AMP {#amp}
Automatic Mixed Precision — running operations in 16-bit floats ([float16](/shared/glossary/#float16) or [bfloat16](/shared/glossary/#bfloat16)) where it is safe, to save memory and speed up training while keeping a [float32](/shared/glossary/#float32) copy of the weights.

### AnimateDiff {#animatediff}
A 2023 technique that adds motion to an existing [Stable Diffusion](/shared/glossary/#stable-diffusion) image model *without retraining it*. The trick is a separately trained [motion module](/shared/glossary/#motion-module) — a small stack of time-aware ([temporal](/shared/glossary/#temporal-inflation)) layers — that you slide in between the [frozen](/shared/glossary/#frozen) image model's blocks: the image model still draws each frame, and the motion module makes consecutive frames move together coherently. Because the module is trained once on generic video and then frozen, you can drop it into almost any community [checkpoint](/shared/glossary/#checkpoint) (a custom-art-style fine-tune, say) and animate that style for free. It is the most popular concrete instance of [temporal inflation](/shared/glossary/#temporal-inflation) packaged as a reusable add-on rather than a full model.

### Anomaly detection {#anomaly-detection}
A debugging mode (`torch.autograd.set_detect_anomaly(True)`) that makes [autograd](/shared/glossary/#autograd) check each operation and raise an error at the exact line that first produces a [NaN](/shared/glossary/#nan) or infinite [gradient](/shared/glossary/#gradients).

### AOTInductor {#aotinductor}
Ahead-of-Time Inductor — a deployment path built on [`torch.export`](/shared/glossary/#torchexport) that compiles a captured model graph into a standalone shared library (`.so`) ahead of time, enabling C++-only inference without a Python runtime.

### AnyRes {#anyres}
A way to feed a [VLM](/shared/glossary/#vlm) images of *any* size and shape instead of squashing every picture to one fixed square. AnyRes splits the image into a grid of tiles at its native aspect ratio, runs the image encoder on each tile separately, and concatenates all the resulting [image tokens](/shared/glossary/#token-visualaudio) — usually alongside one extra down-scaled copy of the whole image for global context. Analogy: rather than shrinking a newspaper page until the text is an unreadable blur, you photograph it column by column at full zoom and lay the close-ups side by side. Example: a tall 768×1536 screenshot might be cut into a 1×2 grid of two 768×768 tiles, doubling the tokens but keeping small text legible — which is why AnyRes (used by Qwen2-VL and InternVL2) sharply improves [OCR](/shared/glossary/#ocr-optical-character-recognition)-heavy and dense-chart benchmarks, at the cost of a longer, slower token sequence.

### Application {#application}
The specific real-world job a model is being built to do — for example, "answer customer-support questions about our refund policy," "summarize internal engineering tickets," or "write product descriptions in our brand voice." A model that scores high on a generic public [benchmark](/shared/glossary/#benchmark) can still flop on *your* application if the two don't match, the way a chef who aces a fine-dining contest may still be the wrong hire for your taco truck. That mismatch is why teams build a small targeted eval shaped like their application instead of trusting a famous leaderboard number.

### AprilTag {#apriltag}
Square fiducial marker with a known code; widely used for pose ground truth

### Arena {#arena}
A way to rank chat models by having them go head-to-head: two models answer the same prompt, a human or [LLM judge](/shared/glossary/#llm-as-judge) picks the winner, and many such duels are turned into [Elo](/shared/glossary/#elo) ratings — the scoring system used for chess players. The public [LMSys Chatbot Arena](https://lmarena.ai/) is the best-known example.

### argmax {#argmax}
The "which one is biggest?" operation: given a list of scores it returns the *position* of the largest one, not the value itself. The name is short for *argument of the maximum* — in math the "argument" is the input you hand a function, so `argmax` answers "which input gives the biggest output?" and returns that input's position. If the [logits](/shared/glossary/#logits) are `[1.2, 4.8, 0.3]`, `argmax` is `1` — the index of `4.8` — which the model reads as "pick token #1." Like scanning a class's test scores and naming the top student rather than reading out their mark. [Greedy decoding](/shared/glossary/#greedy-decoding) is just `argmax` applied to the logits at every step, so it always makes the same choice and never gambles.

### Artifact {#artifact}
An unwanted distortion that a process adds to a signal — something that was not in the original but appears in the output, usually because detail was thrown away to save space. In lossy audio or image compression (an MP3, a JPEG, or a [neural codec](/shared/glossary/#neural-codec)), squeezing the data too hard leaves audible smears, metallic ringing, or muffled detail in sound, and blocky squares or halos around edges in images. Like a photocopy of a photocopy — each pass loses fidelity and adds smudges the original never had. Example: encoding music at a very low [bitrate](/shared/glossary/#bitrate) can make cymbals sound watery or add a faint "underwater" warble — those are compression artifacts.

### Aspect-ratio bucketing {#aspect-ratio-bucketing}
A training trick for image generators: instead of forcing every training image into one square shape by cropping (which slices off the edges of tall portraits and wide landscapes, so the model never learns to compose anything but squares), you sort images into a handful of "buckets" by their shape — tall, wide, square — and build each [batch](/shared/glossary/#batch) from a single bucket so every image in it shares one resolution. This is necessary because a batch must be a single tensor of one [shape](/shared/glossary/#shape), so images of different sizes cannot share a batch unless they are grouped first. The model then learns to generate at many aspect ratios, not just 1:1. Like a photo lab that sorts prints into 4×6, 5×7, and 8×8 trays before processing, so each tray runs through the machine at its own size instead of every photo being trimmed square. Example: a 1280×720 photo goes in the 16:9 bucket; afterward you can ask for a 16:9 image and get a properly-framed one instead of a cropped square.

### ASR (Automatic Speech Recognition) {#asr-automatic-speech-recognition}
Automatic Speech Recognition — the task of turning recorded speech into written text, what your phone does when it transcribes a voice message. A modern ASR model such as [Whisper](/shared/glossary/#whisper) reads a [mel spectrogram](/shared/glossary/#mel-spectrogram) of the audio and emits the words one piece at a time, just as a person listens and types along. Concrete example: feed it a clip of someone saying "turn on the lights" and it returns the string "turn on the lights". Because the model must map a long, wobbly sound wave onto a short line of text, the hard parts are accents, background noise, and rare words — which is why [fine-tuning](/shared/glossary/#fine-tuning) on a specific domain or language helps so much.

### Atari {#atari}
A family of simple 1970s–80s arcade and home-console video games (Pong, Breakout, Space Invaders, and dozens more) that has become a standard testing ground in reinforcement learning. Each game offers a small screen of pixels, a handful of joystick actions, and a score, which makes the set a cheap, convenient [benchmark](/shared/glossary/#benchmark) for agents and [world models](/shared/glossary/#world-model) — hard enough to be interesting, small enough to train on a single GPU. "The Atari benchmark" usually refers to the Arcade Learning Environment, a software package that runs ~57 of these games behind one common interface.

### ATen {#aten}
The C++ tensor library underneath PyTorch's Python frontend

### Attention {#attention}
The operation `softmax(QKᵀ/√d) V` — [content-addressable token mixing](/shared/glossary/#content-addressable-token-mixing); the core of every [transformer](/shared/glossary/#transformer). The [softmax](/shared/glossary/#softmax) step turns the raw query–key match scores into weights between 0 and 1 that decide how much each earlier token contributes to the next one.

### Attention sink {#attention-sink}
The first few tokens of a sequence, which [attention](/shared/glossary/#attention) heads keep putting weight on no matter what those tokens actually say. They are called a *sink* in the plumbing sense — a drain where leftover water collects: on every step the [softmax](/shared/glossary/#softmax) has to spread a full 100% of attention across the tokens, so when a head has nothing important to look at, that spare attention drains into these first tokens. Because the model leans on them, [KV cache](/shared/glossary/#kv-cache) eviction schemes deliberately keep these tokens even when they look unimportant, which keeps quality stable in long-context serving.

### AudioSet {#audioset}
A large public dataset from Google of about two million 10-second clips taken from YouTube, each tagged with the kinds of sound it contains (dog bark, guitar, rain, speech) drawn from a vocabulary of 527 labels. Think of it as "[ImageNet](/shared/glossary/#imagenet) for sound" — a big, broadly labeled collection people use to teach models what everyday audio events sound like. Because each clip comes with short descriptive tags, AudioSet is also a handy source of (audio, caption) pairs for training a model to describe what it hears.

### Autoencoder {#autoencoder}
A neural network that learns to copy its input to its output through a narrow middle layer. It has two halves: an *encoder* that squeezes the input down to a small set of numbers, and a *decoder* that rebuilds the original from those numbers. Because the middle is much smaller than the input, the network cannot simply memorize — it is forced to keep only the most important features, like writing a short summary of a long article and then reconstructing the article from the summary. That small middle representation is called the [latent space](/shared/glossary/#latent-space).

### autograd {#autograd}
The [reverse-mode](/shared/glossary/#reverse-mode) automatic differentiation engine

### Autoregressive model {#autoregressive-model}
A model that generates a sequence one piece at a time, where each new piece is predicted from all the pieces produced so far — like writing a sentence word by word, with every word depending on the words already on the page. The name says what it does: *auto* means "self" and *regression* means "predicting a value from earlier values," so the model predicts each new piece by regressing on its own previous outputs — it feeds on itself. For images, an autoregressive model (such as PixelCNN) draws pixel by pixel in a fixed order. This makes the math clean and the samples sharp, but generation is slow because each step has to wait for the previous one, with no way to compute them all at once.

### AWQ {#awq}
Activation-aware Weight Quantization — preserve weights important to large activations

### AV1 {#av1}
A modern, royalty-free [video codec](/shared/glossary/#video-codec) — the rules for squeezing video into a small file. AV1 compresses noticeably better than older codecs like [H.264](/shared/glossary/#h264) — often 30–50% smaller files at the same quality — but it is much slower to decode, so reading AV1 video back into frames costs more CPU time. Analogy: it is like a denser ZIP format that saves disk space but takes longer to unzip. Example: storing 100 clips as AV1 `.webm` files might use a third of the disk of the same clips as H.264 `.mp4`, but take several times longer to decode each clip during training.

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
The rule that ties the [value](/shared/glossary/#value-function) of a state to the value of the states it leads to: `V(s) = E[r + γV(s')]`, read as *"the worth of being here equals the reward I get now plus the [discounted](/shared/glossary/#discount-factor) worth of wherever I land next."* It is called *recursive* because a state's value is defined in terms of the same quantity one step later, and a *consistency condition* because the true value function is the one set of numbers that makes both sides agree in every state at once. Almost every RL algorithm is some way of forcing this equation to hold when you cannot compute the right-hand side exactly. Named after Richard Bellman, who developed this style of recursive optimization in the 1950s.

### Bellman operator {#bellman-operator}
One application of the [Bellman equation](/shared/glossary/#bellman-equation)'s right-hand side, used as an update rule: take your current guess of the [value function](/shared/glossary/#value-function) and, for every state, replace it with *"reward now plus [discounted](/shared/glossary/#discount-factor) value of the next state."* One such sweep over all states is called a **Bellman backup**. Repeating backups is the engine of [value iteration](/shared/glossary/#value-iteration) and policy evaluation, and it is guaranteed to converge because the operator is a [contraction mapping](/shared/glossary/#contraction-mapping) — each backup pulls every guess strictly closer to the true answer. Think of it as one round of "update each cell from its neighbors," repeated until nothing changes.

### Benchmark {#benchmark}
A fixed, shared test set used to measure and compare models on a task — like a standardized exam everyone sits so scores line up side by side. [MMLU](/shared/glossary/#mmlu) tests knowledge and [GSM8K](/shared/glossary/#gsm8k) tests math; a benchmark is only meaningful while models have not already seen its answers (see [contamination](/shared/glossary/#contamination)).

### Best-of-N {#best-of-n}
An inference trick that samples `N` candidate answers to the same prompt and keeps the single one a scorer — usually a [reward model](/shared/glossary/#reward-model) or [verifier](/shared/glossary/#verifier) — rates highest, like writing several drafts of an email and sending only the best.

### bfloat16 {#bfloat16}
16-bit float with fp32's exponent range — the modern default for training (also written bf16, BF16)

### Bias correction {#bias-correction}
An adjustment applied in the Adam family of optimizers to counteract the zero-initialization of moment estimates; without it, early steps would be artificially small

### Bias-variance tradeoff {#bias-variance-tradeoff}
A way of describing the two different reasons an estimate can be wrong. *Bias* is being consistently off in the same direction — like a bathroom scale that always reads 2 kg heavy; taking more readings never fixes it. *Variance* is being noisy — readings that scatter widely, so any single one might land far from the truth even though they average out correctly. In RL this is the [Monte Carlo](/shared/glossary/#monte-carlo-method) vs [temporal-difference](/shared/glossary/#temporal-difference-learning) choice: a Monte Carlo [return](/shared/glossary/#return) is *unbiased* (it is a real sampled outcome) but *high-variance* (it depends on every random step of a whole episode), while a one-step TD target is *low-variance* (only one step's randomness) yet *biased* (it leans on a current, still-wrong value estimate). Most modern methods accept a little bias to buy a large drop in variance, which is why [bootstrapped](/shared/glossary/#bootstrapping) targets dominate.

### Biases {#biases}
The smaller, additive group of learned [parameters](/shared/glossary/#parameters) in a layer — the `b` in `y = xW + b`. After the [weights](/shared/glossary/#weights) combine the inputs, each output neuron adds its own bias: a fixed offset that shifts the result up or down no matter what the input was. Like the `+ b` that lets a line `y = mx + b` sit above or below the origin, or a starting balance in a bank account before any transactions — it gives each neuron a baseline to lean toward.

### BigGAN {#biggan}
A large [class-conditional GAN](/shared/glossary/#conditional-gan-cgan) from 2018 that was the first to make GAN-generated images look convincingly realistic at high resolution across the thousand everyday categories of ImageNet (dogs, mushrooms, coffee mugs, and so on). Its recipe was mostly "make everything bigger and steadier": much larger [batches](/shared/glossary/#batch), more [parameters](/shared/glossary/#parameters), and a [projection discriminator](/shared/glossary/#projection-discriminator) to feed in the class label cleanly. Like discovering that a decent home cake recipe just needed a bigger oven, more eggs, and a steadier hand to reach bakery quality. Its best-known trick is the *truncation trade-off*: drawing the input noise closer to the average gives cleaner, more typical images at the cost of variety, so you can dial between "safe and pretty" and "wild and diverse."

### Bitrate {#bitrate}
How many bits are used to represent one second of sound (or video) — the data budget per second. A higher bitrate keeps more detail and sounds closer to the original; a lower bitrate saves storage and bandwidth but blurs detail and adds [artifacts](/shared/glossary/#artifact). For a [neural codec](/shared/glossary/#neural-codec) it maps directly to how many [tokens](/shared/glossary/#token-visualaudio) per second the audio is turned into — fewer tokens, lower bitrate. Like the quality slider when you export a photo: more data per image captures finer detail at the cost of a bigger file. Example: MP3 music is often 128–320 kbps (kilobits per second), while EnCodec can compress speech down to about 1.5 kbps — far smaller, but with more audible loss.

### Blackjack {#blackjack}
A simple card-game environment in [Gymnasium](/shared/glossary/#gymnasium) (`Blackjack-v1`) used to teach [Monte Carlo](/shared/glossary/#monte-carlo-method) methods. The state is small and easy to read — your hand's total, whether you hold a usable ace, and the dealer's one visible card — and each game ends quickly in a win, loss, or draw, so you can collect many complete episodes fast. Because the deck's odds are known, the true [value](/shared/glossary/#value-function) of some situations can be worked out exactly, giving you a ground-truth answer to check a sampled estimate against.

### Blackwell {#blackwell}
NVIDIA's 2024 GPU architecture (B100, B200, B200 Ultra) and the successor to [Hopper](/shared/glossary/#hopper). Like swapping a sports car engine for a more powerful one of the same shape, it keeps the same overall design as Hopper but doubles down on low-precision math — better [FP8](/shared/glossary/#fp8) throughput and brand-new FP4 [Tensor Cores](/shared/glossary/#tensor-core) — which is what makes it the preferred chip for the largest 2025-era training and serving runs.

### BM25 (Best Matching 25) {#bm25}
A classic keyword-search ranking — short for **Best Matching 25** — that scores a document by how often the query's words appear in it, weighting rare words more heavily. Think of it like a librarian scanning pages for your exact search words and ranking pages where those words appear most often (especially unusual words) higher on the list. It is the *sparse* (exact-word) counterpart to dense [embedding](/shared/glossary/#embedding) search.

### Bootstrapping {#bootstrapping}
Updating a value estimate using *another current estimate* rather than waiting for the true, fully observed outcome — you "pull yourself up by your bootstraps" by leaning partly on a guess. In [temporal-difference learning](/shared/glossary/#temporal-difference-learning) the target `r + γ V(s′)` bootstraps because `V(s′)` is itself a still-imperfect estimate, not a real measured [return](/shared/glossary/#return). Bootstrapping is what lets an agent learn from a single step instead of a whole episode; it cuts variance but adds [bias](/shared/glossary/#bias-variance-tradeoff), and combined with function approximation and off-policy data it forms the [deadly triad](/shared/glossary/#deadly-triad).

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

### Camera control {#camera-control}
Steering not just *what moves* in a generated video but where the virtual camera goes — [pan](/shared/glossary/#pan), [zoom](/shared/glossary/#zoom), [orbit](/shared/glossary/#orbit), [dolly](/shared/glossary/#dolly) — by feeding the model an explicit camera path. The dominant method represents each frame's camera as [Plücker coordinates](/shared/glossary/#plücker-coordinates) (a six-number description of the ray every pixel looks along) and adds those as an extra input, so the model can keep objects placed consistently as the viewpoint moves. Named systems that do this include **CameraCtrl** and **MotionCtrl**. It is one of the control surfaces — alongside the [motion score](/shared/glossary/#motion-score) and [depth](/shared/glossary/#depth-map)- or pose-conditioning — that turn a raw video generator into a directable tool.

### Canny edge detector {#canny-edge-detector}
A classic (non-neural) algorithm that reduces a photo to a clean black-and-white map of its outlines — the lines where brightness changes sharply, such as the border between a face and the background. It works by measuring the *gradient* (how fast pixel brightness changes) at every point, keeping only the local peaks so the edges come out one pixel thin, and then linking those peaks into continuous contours. Picture tracing all the hard boundaries of a photo with a fine pen and throwing away the shading in between. [ControlNet](/shared/glossary/#controlnet) uses such an edge map as a conditioning signal: the outline says *where* shapes must go while the prompt decides *what* fills them. Named after its inventor, John Canny.

### Catastrophic forgetting {#catastrophic-forgetting}
When training a model on new data erases skills it had already learned, because the new [gradients](/shared/glossary/#gradients) overwrite the old [weights](/shared/glossary/#weights).

### Cascaded diffusion {#cascaded-diffusion}
A way to generate high-resolution images or video by chaining several [diffusion models](/shared/glossary/#diffusion-model) in sequence rather than asking one model to do everything at once: the first model produces a small, coarse result, and each later model takes that output and adds detail or resolution, conditioned on the previous stage's blurry version. "Cascade" is the picture of water falling down a series of steps — each pool feeds the next. Splitting the work this way lets each model specialize (rough layout at low resolution, fine texture at high resolution) and was the dominant recipe for high-resolution video before [latent](/shared/glossary/#latent-space) models; Imagen Video and Make-A-Video both built [super-resolution](/shared/glossary/#super-resolution) cascades. Modern latent-diffusion systems mostly dropped it because compressing into a small latent space up front already makes full-resolution generation affordable inside a single model.

### Causal 3D VAE {#causal-3d-vae}
A [3D VAE](/shared/glossary/#3d-vae) built so that each frame is encoded using only *itself and earlier frames*, never later ones. In machine learning, "causal" means "respecting the flow of time" — just as an effect cannot precede its cause, a causal model cannot look into the future to process the present. This is the same "look only backward" rule a [causal mask](/shared/glossary/#causal-mask) enforces in language models. A plain 3D VAE merges a fixed block of frames together (e.g., always merging 4 frames into 1), so it has no clean way to handle a lone still image (because it expects a full block, a lone image has no later neighboring frames to merge with). The causal version sidesteps this: because the very first frame depends on nothing after it, a single-image input (`T=1`) compresses to a single latent frame (`T'=1`), and the one model can encode both still images and full video. This is what lets frontier video systems train a single shared compressor on a mix of images and clips (see [joint image-video training](/guides/video-generation/projects/joint-image-video-training/)) instead of maintaining separate image and video encoders.

### Causal mask {#causal-mask}
A mask applied to [attention](/shared/glossary/#attention) scores that hides future positions, so each token can attend only to itself and the tokens before it. In machine learning, "causal" means "respecting the flow of time" — just as an effect cannot precede its cause, a causal model cannot look into the future to process the present.

### CausVid {#causvid}
A method for fast, [streaming](/shared/glossary/#streaming-video-generation) video generation that [distills](/shared/glossary/#distillation) a slow, high-quality diffusion "teacher" — which denoises a whole clip at once and needs dozens of steps — into a *causal* [autoregressive](/shared/glossary/#autoregressive-model) "student" that emits frames in time order using only a few steps each. "Caus" is short for *causal*: like the [causal mask](/shared/glossary/#causal-mask) in a language model, each frame may look only at frames already generated, never future ones, which is what lets the model run indefinitely and start showing output immediately. It is one of the current recipes (alongside [Self-Forcing](/shared/glossary/#self-forcing)) for real-time and infinite-length video.

### CBF {#cbf}
Control Barrier Function — runtime safety filter via a constraint on `ḣ`

### CDNA / RDNA {#cdna--rdna}
AMD's datacenter / consumer GPU architectures

### CelebA {#celeba}
A dataset of about 200,000 photos of celebrity faces, each labeled with attributes such as "smiling," "wearing glasses," or "blond hair." Because every image is a face, it is a favorite for studying generative models of a single, well-defined kind of picture — you can easily judge whether a generated face looks real, and the attribute labels let you check whether the model learned to control features like hair or expression.

### CFG (classifier-free guidance) {#cfg-classifier-free-guidance}
Classifier-free guidance — the standard inference trick for making a [diffusion model](/shared/glossary/#diffusion-model) follow its prompt more closely. The model is trained to run both *with* the condition (the prompt or label) and *without* it; at sampling time you take the difference between the two predictions and amplify it, pushing the output away from "generic" and toward "matches the prompt." Unlike [classifier guidance](/shared/glossary/#classifier-guidance), it needs no separate classifier — the same generator provides both signals — which is why it became universal in text-to-image models. A guidance-scale knob trades diversity for prompt adherence.

### CFG fusion {#cfg-fusion}
A diffusion-serving optimization for [classifier-free guidance](/shared/glossary/#cfg-classifier-free-guidance), which normally needs *two* model passes per denoising step — one conditioned on the prompt, one unconditioned. CFG fusion runs both in a single batched forward pass (stacking them as a batch of two) instead of two separate calls, so the GPU is launched once per step rather than twice. Like cooking two portions in one pan instead of washing up between them — same result, far less overhead.

### Chain rule {#chain-rule}
A calculus principle used to compute the derivative of a composite function by multiplying the derivatives of its parts.

### Chameleon {#chameleon}
Meta's family of [native multimodal](/shared/glossary/#native-multimodal) models that treat text and images as one single stream of [tokens](/shared/glossary/#token-visualaudio): pictures are turned into image tokens by a [VQ-VAE](/shared/glossary/#vq-vae), mixed in with ordinary text tokens, and a single [transformer](/shared/glossary/#transformer) is trained from scratch over the combined sequence with one plain [next-token-prediction](/shared/glossary/#next-token-prediction) objective. This is the [early-fusion](/shared/glossary/#fusion-earlymiddlelate) recipe taken to its extreme — there is no separate vision encoder bolted on, so the model can read and write any interleaving of words and image patches. Analogy: instead of a writer and an illustrator passing a notebook back and forth, one person who was taught from the start to "write" in both words and pictures sketches and types along the same flowing line. Example: handed a recipe that is half text and half photos, Chameleon continues it by emitting the next word *or* the next patch of an image, whichever comes next; the name nods to the lizard that blends seamlessly into any surroundings — here, any mix of modalities.

### Character consistency {#character-consistency}
Keeping the *same* person or object looking like themselves across the many separate shots that make up a long video — the same face, hairstyle, and clothing in shot 5 as in shot 1. The failure mode is called **drift**: because each shot is generated somewhat independently, small differences pile up until the character slowly morphs into someone else. Fixes pin the identity to a fixed reference — an [IP-Adapter](/shared/glossary/#ip-adapter) that feeds a reference image's appearance into every shot, or a character [LoRA](/shared/glossary/#lora) fine-tuned on a few pictures of that character. You measure the leftover drift by turning each generated face into an [embedding](/shared/glossary/#embedding) and checking how far it travels from shot to shot. Like a film's continuity supervisor making sure an actor's costume and haircut match between takes shot weeks apart. Related named systems include DreamBooth-Video and ID-Animator.

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

### Class conditioning {#class-conditioning}
Telling a generative model *which* category to produce instead of leaving it to chance. You feed the model a label (for example, the digit "7" or the class "cat") alongside its usual input, so at generation time you can ask for exactly that class. Without it, the model draws a random sample from everything it learned; with it, you steer the output — like ordering a specific flavor instead of accepting whatever scoop you are handed.

### Classifier guidance {#classifier-guidance}
An early technique for steering a [diffusion model](/shared/glossary/#diffusion-model) toward a chosen class or label: you train a separate image classifier that can read *noisy* images, then at each denoising step add a nudge in the direction of its [gradient](/shared/glossary/#gradients) — the direction that makes the target class more likely. Like a critic standing over a painter and pointing "more toward a cat" at every brushstroke, it trades a little sample diversity for much stronger adherence to the condition. Its drawback is the extra cost of training and running that dedicated noisy classifier, which [classifier-free guidance (CFG)](/shared/glossary/#cfg-classifier-free-guidance) later eliminated by getting the same steering from the generator itself.

### Cliff Walking {#cliff-walking}
A small gridworld environment ([Gymnasium](/shared/glossary/#gymnasium)'s `CliffWalking-v0`, from Sutton & Barto) built to expose the gap between [on-policy](/shared/glossary/#on-policy) and [off-policy](/shared/glossary/#off-policy) learning. The agent walks from start to goal along the top of a cliff; every step costs −1, and stepping into the cliff costs −100 and snaps it back to the start. The shortest route hugs the cliff edge, but any random exploratory move there is disastrous — so [SARSA](/shared/glossary/#sarsa), which accounts for its own exploration, learns a cautious path one row up, while [Q-learning](/shared/glossary/#q-learning), which assumes greedy future behavior, learns the risky edge path. Reproducing its two learning curves is Sutton & Barto's Figure 6.5.

### CLIP {#clip}
Contrastive Language-Image Pretraining — a model that learns to match pictures with the words that describe them. It has two separate encoders: one reads an image, the other reads text, and both map their input into the same shared space of [embeddings](/shared/glossary/#embedding), so a photo of a dog and the caption "a dog" land near each other while the caption "a bicycle" lands far away. It is trained on hundreds of millions of image–caption pairs scraped from the web with a *contrastive* objective: pull each true image–caption pair together and push every mismatched pair apart. Think of it as teaching two translators — one who only speaks "image" and one who only speaks "text" — to agree on a common language, so any picture and its description end up pointing at the same spot. Once trained you can measure how well a caption fits an image (a *CLIP score*), classify images with no extra training by comparing them to label phrases (*zero-shot*), or extract just the text encoder and feed it into a generator. Why add a text encoder to a generator when CLIP is already trained to match text and images? Because the generator itself doesn't inherently understand words; by borrowing CLIP's text encoder—which has already learned an excellent representation of language—you give the generator a deep understanding of descriptive language so it can follow prompts. It is the text encoder inside early [Stable Diffusion](/shared/glossary/#stable-diffusion).

### CLIP-L {#clip-l}
A specific, larger version of the [CLIP](/shared/glossary/#clip) model. The "L" stands for Large (compared to the standard or Base versions), meaning it has more [parameters](/shared/glossary/#parameters) and can grasp more nuanced relationships between text and images. Because of its stronger understanding, it is often used as the text encoder in powerful generators to ensure they follow complex prompts accurately.

### Closed-form {#closed-form}
A solution you can write down and compute directly with a fixed formula, instead of reaching it through many rounds of trial-and-error. Solving `2x = 10` by writing `x = 5` is closed-form; nudging `x` up and down until both sides match is not. In [DPO](/shared/glossary/#dpo) a closed-form objective lets the model learn straight from preference pairs with one training [loss](/shared/glossary/#loss-function), skipping the slow [reward model](/shared/glossary/#reward-model)-plus-[PPO](/shared/glossary/#ppo) loop of classic [RLHF](/shared/glossary/#rlhf).

### CLS token {#cls-token}
A special extra "summary" token (short for *classification*) glued to the front of a [transformer](/shared/glossary/#transformer)'s input sequence whose only job is to soak up information from all the real tokens, so its final output vector can stand in for the whole input. In a [ViT](/shared/glossary/#vit) it has no [patch](/shared/glossary/#patch) of its own — it starts as a learned placeholder and, through [attention](/shared/glossary/#attention), gathers a single image-wide description you then hand to a classifier. Like a meeting secretary who owns none of the agenda items but listens to every speaker and writes the one-line summary everyone refers to afterward. (Many models instead average all token outputs — *mean pooling* — which often works just as well.)

### CNN {#cnn}
Convolutional Neural Network — a neural network built mainly from [convolution layers](/shared/glossary/#convolution-layers); the standard architecture for image tasks. Instead of staring at the whole picture at once, a CNN slides a small magnifying glass across the image, checking one little patch at a time for simple features — an edge here, a splash of color there. Early layers spot these tiny patterns; deeper layers stitch them into bigger ideas (edges become a whisker, whiskers become a cat). Because the *same* magnifying glass is reused over every patch, a CNN needs far fewer [parameters](/shared/glossary/#parameters) than a network that wired up every pixel separately — and it can recognize a cat whether it sits in the corner or the center of the photo.

### COCO {#coco}
Common Objects in Context — a widely used image dataset of roughly 120,000 everyday photos, each paired with five short human-written captions plus labeled object outlines, so it serves as a shared benchmark for both captioning and object detection. Think of it as the field's standard "practice set," the way students everywhere drill the same well-known textbook problems so their results can be compared. In this guide, COCO's image-caption pairs are the convenient small-scale fuel for training toy [CLIP](/shared/glossary/#clip), [Q-Former](/shared/glossary/#q-former), and captioning models — big enough to be realistic, small enough to fit a weekend.

### Codebook {#codebook}
The fixed list of code vectors a [VQ-VAE](/shared/glossary/#vq-vae) is allowed to use to describe an image — think of it as a numbered paint set, where every patch of the picture must be painted using one of the colors on the palette rather than any color imaginable. The encoder looks at a patch, finds the closest entry in this list, and stores just that entry's index, which is what makes the latent code *discrete*. A bigger codebook offers more "colors" (finer detail) but is harder to use fully — see [codebook collapse](/shared/glossary/#codebook-collapse).

### Codebook collapse {#codebook-collapse}
A failure where a [VQ-VAE](/shared/glossary/#vq-vae) ends up using only a few entries of its [codebook](/shared/glossary/#codebook) and ignores the rest — like owning a 64-color crayon box but only ever drawing with three. The unused entries are wasted capacity, so the model stores less detail than its codebook size suggests and reconstructions stay blurry. Common fixes are [EMA](/shared/glossary/#ema-weights) codebook updates, re-initializing dead (never-chosen) entries near popular ones, and k-means warmup. It is the discrete-latent cousin of [mode collapse](/shared/glossary/#mode-collapse) in [GANs](/shared/glossary/#gans).

### Collate function {#collate-function}
The function a [DataLoader](/shared/glossary/#dataloader) uses to combine a list of individual samples into one batched tensor; a custom one can pad variable-length data.

### Collective operation {#collective-operation}
A communication step that all processes ([ranks](/shared/glossary/#rank)) in a distributed job perform together — such as [AllReduce](/shared/glossary/#allreduce); if one rank skips it, the others wait forever.

### Collision mesh {#collision-mesh}
Simplified geometry used for collision tests, distinct from visual mesh

### Column-wise partitioning {#column-wise-partitioning}
Splitting a weight matrix along its column (output) dimension so that each GPU holds a vertical slice and computes part of the output independently — the standard first step in [Megatron](/shared/glossary/#megatron)-style [tensor parallelism](/shared/glossary/#tensor-parallelism-tp).

### Concatenation {#concatenation}
The most basic way to fuse two [modalities](/shared/glossary/#modality): just stick their feature vectors end to end into one longer vector and hand that to the next layer. If an image [embedding](/shared/glossary/#embedding) has 512 numbers and a text embedding has 512, concatenation glues them into one 1024-number vector — like taping two index cards side by side and reading them as a single wider card. It adds almost no parameters and is a surprisingly strong baseline, but the two streams never actually *look* at each other the way [cross-attention](/shared/glossary/#cross-attention) lets them; they only get combined once the next layer mixes the stacked numbers, which is why richer [fusion](/shared/glossary/#fusion-earlymiddlelate) often wins when the task needs the modalities to interact.

### Conditional GAN (cGAN) {#conditional-gan-cgan}
A [GAN](/shared/glossary/#gans) that is told *which* kind of image to make instead of producing a random one. The class label (for example, the digit "7") is fed to both the [generator](/shared/glossary/#generator) and the [discriminator](/shared/glossary/#discriminator), so generation becomes [class-conditioned](/shared/glossary/#class-conditioning) — you ask for a category and get it. Like a vending machine where you press a button for the snack you want rather than taking whatever drops. See also [projection discriminator](/shared/glossary/#projection-discriminator), an efficient way to feed the label to the critic.

### Consistency model {#consistency-model}
A [diffusion](/shared/glossary/#diffusion-model)-derived model trained so that *every* point along a noisy-to-clean denoising path maps directly to the same final clean image — so at sampling time you can jump from pure noise to a finished picture in one (or a handful of) steps instead of the usual dozens. It is built by *consistency [distillation](/shared/glossary/#distillation)*: a student learns to agree with itself at neighboring noise levels along a teacher's [ODE](/shared/glossary/#ode) trajectory. Like a winding park path where every bench has a sign pointing straight to the exit — wherever you start, one glance gets you to the end. Trade-off: a huge speedup (1–4 steps vs ~50) for a modest dip in quality. The latent-space version is the [LCM](/shared/glossary/#lcm).

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

### Contraction mapping {#contraction-mapping}
A function that, every time you apply it, pulls any two inputs *closer together* by at least a fixed ratio — so repeating it drags everything toward a single unmovable point (its *fixed point*). This matters in RL because the [Bellman operator](/shared/glossary/#bellman-operator) is a contraction with ratio γ (the [discount factor](/shared/glossary/#discount-factor)): each [backup](/shared/glossary/#bellman-operator) shrinks the gap between your current estimate and the true [value function](/shared/glossary/#value-function) by a factor of γ, so the error falls like `γ, γ², γ³, …` and the estimate is guaranteed to converge. Analogy: photocopying a copy of a copy at 90% size each time — the images shrink toward one point no matter what you started with. This single property justifies every iterative value-based algorithm in RL.

### ControlNet {#controlnet}
An add-on that gives a frozen [diffusion model](/shared/glossary/#diffusion-model) precise *spatial* control. It clones the [U-Net](/shared/glossary/#u-net)'s encoder into a parallel branch that reads an auxiliary conditioning image — a [depth map](/shared/glossary/#depth-map), a pose skeleton, a [Canny edge map](/shared/glossary/#canny-edge-detector), a [segmentation map](/shared/glossary/#segmentation-map) — and feeds that branch's features back into the original network so the output follows the supplied structure. The base model stays untouched (so its quality and prompt-following are preserved) and only the new branch is trained; the connections use [zero-convolutions](/shared/glossary/#zero-conv) so the branch contributes nothing at first and is learned gradually. Like laying tracing paper with an outline over a painter's canvas: the prompt still chooses colors and texture, but every shape must follow the lines you drew. Adapting ControlNet to *video* (sometimes called ControlNet-Video) feeds a per-frame conditioning map — e.g. a depth map for every frame — into a video diffusion model; the extra challenge is keeping the control [temporally consistent](/shared/glossary/#temporal-consistency) so the result does not flicker frame to frame.

### ConvLSTM {#convlstm}
An [LSTM](/shared/glossary/#lstm) that swaps its internal matrix multiplications for [convolutions](/shared/glossary/#convolution-layers), so it can carry memory across time *and* keep the 2D spatial layout of each frame instead of flattening it into a single vector. A plain LSTM treats its input as a flat list of numbers, which throws away which pixel sat next to which; a ConvLSTM keeps the grid intact, so a local fact like "this corner is getting brighter" stays local. That makes it a natural fit for [future frame prediction](/shared/glossary/#future-frame-prediction), where both *what* changes and *where* it changes matter. It was the standard baseline for video prediction before transformers and diffusion took over, and later recurrent variants such as PredRNN refined the same idea with extra memory paths between layers.

### Convolution Layers
These are the foundational building blocks of a [Convolutional Neural Network (CNN)](/shared/glossary/#cnn). Their job is to scan an image and hunt for specific patterns.

Each layer uses a small grid of numbers—called a *filter* or *kernel*—that acts like a tiny pattern detector. The network slides this filter systematically, step by step, across the entire image. At every pause, the filter looks exclusively at the small patch of the image directly *underneath* it, checks how well that patch matches the pattern it is hunting for, and spits out a single "match score." As the filter sweeps over the whole image, it records these scores onto a new, blank grid called a **feature map**.

Picture a small, transparent stencil painted with red-and-white stripes. You drag this stencil step by step over a crowded "Where's Waldo?" poster:

* When the stencil is *underneath* a patch of blue sky or a green tree, the patterns don't match, so it leaves a "0" (a dark mark) on your feature map.
* But when you slide the stencil directly over Waldo's shirt, the stripes align perfectly, leaving a high score (a bright mark) on your feature map.

By the end of the sweep, your feature map acts as a glowing treasure map, lighting up exactly where Waldo's shirt is located.

In a real network, one filter might hunt for stripes, another for glasses, and another for the curve of a beanie cap. By stacking many of these convolution layers together, the network pieces together simple clues to eventually recognize a complex object like Waldo himself. Because the network reuses the same tiny filter across the entire poster, the process stays incredibly efficient—and ensures that the pattern is found no matter where it is hiding in the picture.

### copy {#copy}
A tensor that owns its own storage, independent of any source tensor; created by `.clone()`, or automatically by operations like `.contiguous()` and `reshape` when a view is not possible

### Cosine decay {#cosine-decay}
A [learning-rate](/shared/glossary/#learning-rate) schedule that, after [warmup](/shared/glossary/#warmup), lowers the rate along the smooth downward half of a cosine curve until it reaches near zero by the end of training. The step size starts large and eases off gently — like braking smoothly as you coast up to a stop sign instead of slamming the pedal at the last moment — which helps the model settle into a good solution. It is the long-standing default schedule, before newer recipes like [WSD](/shared/glossary/#wsd).

### Cosine similarity {#cosine-similarity}
A score from −1 to +1 for how closely two vectors point in the *same direction*, ignoring how long they are. You get it by taking the [dot product](/shared/glossary/#dot-product) of the two vectors and dividing by both of their lengths — which is the same as first [L2-normalizing](/shared/glossary/#l2-normalization) each vector (rescaling it to length 1 so it sits on the unit sphere) and then taking a plain dot product. **Worked example:** for `a = [3, 4]` (length 5) and `b = [4, 3]` (length 5), the dot product is `3·4 + 4·3 = 24`, so cosine similarity is `24 / (5·5) = 0.96` — nearly 1, meaning they point almost the same way. A value of 1 means identical direction, 0 means unrelated (at right angles), and −1 means exactly opposite. Analogy: two people pointing at the night sky — cosine similarity asks only "are your arms aimed at the same star?", not "whose arm is longer." This is the standard way to compare [embeddings](/shared/glossary/#embedding), because in most models *meaning* lives in a vector's direction, not its magnitude; it is the score inside [CLIP](/shared/glossary/#clip) matching and the building block of [InfoNCE](/shared/glossary/#infonce).

### Cost per million tokens {#cost-per-million-tokens}
The standard price unit for running a model in production: how many dollars it costs to generate one million tokens of output. You get it by dividing the hardware's hourly cost by how many tokens it produces per hour — like working out a car's cost per mile from its fuel bill and the distance it covers. Almost every serving optimization, from [batching](/shared/glossary/#batching) to [quantization](/shared/glossary/#quantization), is ultimately a way to push this one number down.

### CoT {#cot}
Chain of Thought — prompting or training a model to write out its reasoning step by step before giving a final answer, the way a student shows their work on a math problem instead of blurting out just the result.

### Covariance {#covariance}
A measure of how two quantities move *together*: when one is above its average, does the other tend to be above too (positive covariance), below (negative), or neither (near zero)? Stacked up for many quantities at once it becomes a *covariance matrix*, which describes the overall shape and spread of a [cloud of points](/shared/glossary/#point-cloud) — how wide it is in each direction and how tilted. Picture a scatter of darts on a board: the covariance tells you whether the cloud is a tight circle, a wide oval, or a diagonal streak. [FID](/shared/glossary/#fid) compares the covariances of real and generated image features to check that the two clouds have the same *shape*, not just the same center.

### CQL {#cql}
Conservative Q-Learning — offline RL with a pessimistic Q penalty

### Cross-attention {#cross-attention}
A form of [attention](/shared/glossary/#attention) that lets one stream of data look at and pull in information from a *different* source. In ordinary self-attention a sequence attends to itself; in cross-attention the *queries* come from one place (say, the image being denoised) while the *keys* and *values* come from another (say, the text prompt's [embeddings](/shared/glossary/#embedding)) — often a different [modality](/shared/glossary/#modality) entirely. Picture a painter who keeps glancing at a written description while working: each patch of canvas asks the words "which of you matters to me?" and pulls in the answer to decide what to paint. This is exactly how [diffusion models](/shared/glossary/#diffusion-model) inject a text prompt into the image: inside the [U-Net](/shared/glossary/#u-net) the image patches are the queries and the text tokens are the keys and values, so every region of the picture can attend to the words most relevant to it.

### Cross-encoder {#cross-encoder}
A model that reads a query and one candidate document *together* in a single pass and outputs one relevance score — far more accurate than comparing their separate [embeddings](/shared/glossary/#embedding), but too slow to run over a whole corpus, so it is used to [rerank](/shared/glossary/#reranker) a short candidate list.

### Cross-entropy {#cross-entropy}
A [loss function](/shared/glossary/#loss-function) that scores how surprised a model is by the correct answer: it stays small when the model gave the true next word a high probability and grows large when it was confidently wrong. Like grading a weather forecaster on confidence and not just on being right — announcing "90% chance of sun" and then getting rain costs far more points than a hedged "50%." Training an [LLM](/shared/glossary/#llm) means adjusting the [weights](/shared/glossary/#weights) to push this surprise as low as it will go.

### Cross-modal retrieval {#cross-modal-retrieval}
Searching with one [modality](/shared/glossary/#modality) to find matches in another — typing a caption to pull up the right photo, or handing in an image to find the text that describes it. It works by mapping both modalities into one shared space (for instance with [CLIP](/shared/glossary/#clip)), so that a query and its true match land near each other; you then keep the few stored items with the highest [cosine similarity](/shared/glossary/#cosine-similarity) to the query (a [top-k](/shared/glossary/#top-k) nearest-neighbor lookup). Because every item is encoded just once, answering a query is only a batch of [dot products](/shared/glossary/#dot-product) — one [matmul](/shared/glossary/#matmul) — which is why it scales to huge collections. Analogy: a library where books and their summaries are shelved by *meaning* instead of by title, so a summary in your hand leads you straight to the shelf holding the matching book. It is the first of the four canonical multimodal tasks and the thing [dual encoders](/shared/glossary/#dual-encoder) are best at.

### Cross product {#cross-product}
A way to combine two 3D vectors that — unlike the [dot product](/shared/glossary/#dot-product), which boils them down to a single number — returns a *third vector*. That new vector points at a right angle (perpendicular) to both of the originals. For `a = [a₁, a₂, a₃]` and `b = [b₁, b₂, b₃]` it is computed slot by slot as `a × b = [a₂·b₃ − a₃·b₂, a₃·b₁ − a₁·b₃, a₁·b₂ − a₂·b₁]`. For example, `[1, 0, 0] × [0, 1, 0] = [0, 0, 1]`: two arrows lying flat on a table (one pointing "east", one "north") produce one pointing straight *up*, out of the table.

**What it does (the effect).** Where the [dot product](/shared/glossary/#dot-product) measures how *aligned* two vectors are, the cross product hands you the axis they are *not* aligned along. Analogy: Imagine laying two pens flat on a desk so their ends touch, forming a "V" shape. Now, imagine taking a pencil and standing it perfectly upright exactly where the two pens meet, pointing directly at the ceiling. That standing pencil is the cross product. Furthermore, the length of that pencil depends on how wide you open the "V" shape. If you open the pens to a perfect 90-degree corner, the pencil grows to its maximum height. If you close the pens together so they overlap and point the same way, the pencil vanishes entirely (its length shrinks to zero). 

**Why [Plücker coordinates](/shared/glossary/#plücker-coordinates) use it.** If you just say "a line pointing North," you haven't given enough information to pin it down — imagine two parallel train tracks that both point North, but sit in completely different places. You need a way to tell them apart.

This is where the cross product comes in. By taking the cross product of a [position vector](/shared/glossary/#position-vector) (pointing to *any* spot on the track) and the track's *direction*, you create a new vector called the line's **moment**. Think of this moment as a unique fingerprint for the line's exact location in space. The magic of this math is that no matter which spot you pick along that specific track, the cross product always spits out the exact same fingerprint. But if you do the math on the *other* parallel track, you get a completely different fingerprint. So, by keeping just two things — the direction and this fingerprint — you perfectly lock down exactly which line you are talking about.

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

### DALL·E 3 {#dalle-3}
OpenAI's [text-to-image](/shared/glossary/#text-to-image) model, best known for following long, detailed prompts faithfully — it reliably places the right objects, counts, and spatial relationships you asked for. Its standout trick was training on [synthetic captions](/shared/glossary/#synthetic-captions): instead of messy web [alt-text](/shared/glossary/#alt-text), the team rewrote the training captions to richly describe each image, so the model learned exactly which words map to which pictures. Like a student who finally aces reading comprehension once their textbook is rewritten in clear, complete sentences. It is the proprietary counterpart to open models like [Stable Diffusion](/shared/glossary/#stable-diffusion), and the public demonstration that *better captions can beat a bigger model*.

### Data parallelism {#data-parallelism}
The default way to train across many GPUs: put a full copy of the model on each GPU, feed each one a different slice of the batch, then average their [gradients](/shared/glossary/#gradients) so all copies stay identical — like several graders each marking part of an exam pile and then pooling the scores. (See [DDP](/shared/glossary/#ddp).)

### DataLoader {#dataloader}
PyTorch's iterator that pulls samples from a Dataset, groups them into batches, and can load them in parallel using [worker processes](/shared/glossary/#worker-processes).

### DCGAN {#dcgan}
Deep Convolutional [GAN](/shared/glossary/#gans) — the 2015 recipe that first made GAN training reliable, by building both the [generator](/shared/glossary/#generator) and [discriminator](/shared/glossary/#discriminator) out of [convolution layers](/shared/glossary/#convolution-layers) with a few simple rules (batch normalization, no pooling layers, specific activations). Before it, GANs often fell apart mid-training; DCGAN's architecture became the default starting point that almost every later image GAN built on.

### DDIM {#ddim}
Denoising Diffusion Implicit Models — a way to sample from an already-trained [DDPM](/shared/glossary/#ddpm) far faster. The word "Implicit" means that instead of relying on a strict, random step-by-step chain to add noise, the math implicitly defines a non-random shortcut that reaches the same noisy result, allowing us to skip many steps when generating an image backward. Where DDPM's reverse process is *stochastic* (it injects fresh randomness at every step and may need ~1000 steps), DDIM makes the path *deterministic*: the same starting noise always yields the same image, and the smooth path lets you skip most steps, so ~50 steps match 1000-step quality. Crucially it reuses the same trained network — DDIM changes only how you *sample*, not how you *train*. Like taking a few long, confident strides across a room instead of many tiny shuffles.

### DDIM inversion {#ddim-inversion}
Running the deterministic [DDIM](/shared/glossary/#ddim) sampler *in reverse* to find the starting noise that would regenerate a given real image. Normal sampling goes noise → image by removing a little noise each step; inversion walks the same path backward, image → noise, *adding* the noise the model would have removed. Once you hold that noise you can change the prompt and denoise forward again, and because the path is largely reused the edit keeps the original's layout and pose. The catch is drift: each backward step is only approximate, so the recovered noise does not reconstruct the photo perfectly. A follow-up technique called *null-text inversion* fixes this by tweaking the *empty-prompt embedding* — the baseline "blank canvas" signal the model uses when given no text — until the reconstruction perfectly matches the original photo. Think of it like reverse-engineering the exact base batter recipe (the empty prompt) for a finished cake. Once you tweak that base batter so the cake bakes perfectly, you can swap in one new ingredient (the new text prompt) to change just the flavor, while the shape and texture come out exactly the same.

### DDP {#ddp}
Distributed Data Parallel — replicate model, split batch, all-reduce gradients

### DDPG {#ddpg}
Deep Deterministic Policy Gradient — the first deep-RL continuous-control algorithm

### DDPM {#ddpm}
Denoising Diffusion Probabilistic Models — the foundational 2020 paper and recipe that kicked off the modern [diffusion](/shared/glossary/#diffusion-model) era. Training is disarmingly simple: take a clean image, add a known amount of random (Gaussian) noise, and teach a network (usually a [U-Net](/shared/glossary/#u-net)) to predict that noise so it can be subtracted back off; the [loss](/shared/glossary/#loss-function) is just [mean squared error](/shared/glossary/#mse-mean-squared-error) on the noise. To generate, start from pure static and repeat the learned "remove a little noise" step many times (classically 1000) until an image appears. Because there is no adversarial game, it sidesteps the [mode collapse](/shared/glossary/#mode-collapse) that plagues [GANs](/shared/glossary/#gans).

### Deadly triad {#deadly-triad}
Function approximation + bootstrapping + off-policy data → instability

### Decode {#decode}
The token-by-token half of LLM inference: after [prefill](/shared/glossary/#prefill) digests the prompt, the model generates one new token per [forward pass](/shared/glossary/#forward-pass), each step reading the whole [KV cache](/shared/glossary/#kv-cache) before producing the next [logits](/shared/glossary/#logits). Like writing a sentence one word at a time while glancing back over every word already written — fast per step, but the constant re-reading of the page is what bounds speed. Decode is [memory-bandwidth-bound](/shared/glossary/#roofline) on a GPU, the opposite of [prefill](/shared/glossary/#prefill), and is what most serving optimizations target.

### decord {#decord}
A fast video-reading library that decodes frames straight into [tensors](/shared/glossary/#tensor), built for deep-learning data loaders. Its key trick is efficient random access: you can ask for "frames 0, 30, and 90" and it jumps to them without decoding everything in between, which is exactly what frame sampling needs. Analogy: a regular video player reads a movie front to back like a cassette tape, while decord works like a book with an index — it flips straight to the page you want. Example: `vr.get_batch([0, 30, 90])` returns just those three frames as a single tensor, ready for the model.

### Decoupled {#decoupled}
A training technique where two effects that are mathematically equivalent in standard SGD are separated into independent operations. In AdamW, weight decay is decoupled from the gradient update so that the regularization strength is not scaled by the adaptive learning rate.

### Deduplication {#deduplication}
Removing repeated or near-repeated documents from a training corpus; one of the highest-return cleaning steps in [pretraining](/shared/glossary/#pretraining).

### Deep network {#deep-network}
A neural network with *many* layers stacked one after another, so the input passes through a long chain of transformations before reaching the output — "deep" literally refers to that depth (the number of stacked layers), in contrast to a "shallow" network of just one or two. Each layer builds on the features the previous one produced: in an image model the early layers might pick out edges, the middle layers shapes, and the later layers whole objects — like an assembly line where every station adds a little more refinement. Depth is what lets these models learn rich, abstract patterns, but it also makes them hard to train, because the learning signal ([gradients](/shared/glossary/#gradients)) has to travel back through every layer and tends to fade or blow up along the way — which is precisely the problem [residual connections](/shared/glossary/#residual-connection) and [normalization](/shared/glossary/#normalization) were invented to tame.

### Deepfake {#deepfake}
A fake but convincingly realistic image, video, or audio clip of a *real* person — produced by an AI model — that shows them doing or saying something they never did. The name blends "[deep](/shared/glossary/#deep-network) learning" with "fake." Like a forged signature, but for someone's face and voice: the danger is that a viewer cannot tell it apart from genuine footage, which is why deepfakes drive misinformation, fraud, and non-consensual imagery. They are the central threat that [watermarking](/shared/glossary/#watermarking) and detection tools like [SynthID](/shared/glossary/#synthid) try to counter, by tagging or spotting synthetic media so a clip's origin can be checked.

### DeepSpeed {#deepspeed}
Microsoft's open-source library for training very large models efficiently. It is best known for [ZeRO](/shared/glossary/#zero), which shards a model's [parameters](/shared/glossary/#parameters), [gradients](/shared/glossary/#gradients), and [optimizer state](/shared/glossary/#optimizer-state) across GPUs so no single GPU has to hold the whole model — the same idea as PyTorch's [FSDP](/shared/glossary/#fsdp). Think of it as a moving company that splits one giant load across several trucks instead of trying to cram everything into one.

### Depth map {#depth-map}
A grayscale image that records *how far away* each pixel is rather than its color — near things are drawn light and far things dark (or the reverse), like a black-and-white fog where closer objects glow brighter. It throws away texture and color and keeps only the 3D shape of a scene: which parts stick out toward the camera and which recede into the distance. You can estimate one from an ordinary photo with a depth-prediction network, or capture it directly with a depth sensor. [ControlNet](/shared/glossary/#controlnet) uses a depth map as a conditioning signal so a generated image keeps the same sense of near-and-far layout — the prompt repaints the surfaces, but a person standing in front of a wall stays in front of the wall.

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

### Dial {#dial}
Used throughout these guides as a verb: *to dial* a value up or down means to adjust it smoothly anywhere along a continuous range — the way you turn a knob — instead of flipping a switch that is only ever fully on or fully off (the same idea as a [soft gate](/shared/glossary/#soft-gate)). The word borrows from a physical dial: a round gauge you rotate, so its setting *is* an angle. That rotating-angle picture is also why a dial is an apt image for [RoPE](/shared/glossary/#rope), which marks a token's position by rotating its vectors through an angle. Examples elsewhere in this glossary: a [temperature](/shared/glossary/#temperature) dial that trades safe answers for creative ones, or a [motion score](/shared/glossary/#motion-score) that dials how much a generated video moves.

### Diffusion Forcing {#diffusion-forcing}
A training trick that gives *each frame its own independent noise level* instead of noising the whole clip to the same amount. Because the model learns to denoise a sequence whose frames sit at different stages of cleanup, at generation time it can hold earlier frames clean while denoising later ones — letting one model both denoise a full clip at once *and* roll out frame-by-frame like an [autoregressive model](/shared/glossary/#autoregressive-model). It is the bridge between full-sequence [diffusion](/shared/glossary/#diffusion-model) and next-frame prediction, and underlies several long-form and [streaming](/shared/glossary/#streaming-video-generation) video methods. The name captures the idea that each frame is independently "forced" to a chosen noise level.

### Diffusion model {#diffusion-model}
A generative model that learns to *un-noise* an image (or video, or audio). 
**The Key Intuition:** The model *only* learns the reverse process. The forward process (adding noise) is a fixed, mathematical destruction (like randomly shuffling a puzzle) that requires no learning. The reverse process is the actual learning phase: the network is handed a scrambled image along with a strict label of *how much* noise is currently present (often measured by a timestep `t` or standard deviation `σ`). This noise level acts as a critical condition, physically injected into the network via mechanisms like [AdaGN](/shared/glossary/#adagn-adaptive-group-normalization) so the model knows whether to focus on forming broad outlines (high noise) or tweaking fine details (low noise).

### DINOv2 {#dinov2}
A strong, off-the-shelf image encoder from Meta trained in a [self-supervised](/shared/glossary/#self-supervised) way via [self-distillation](/shared/glossary/#self-distillation) — it learns purely from images, with no human labels, by teaching the network to give two different crops of the same photo matching internal descriptions. The result is a general-purpose [ViT](/shared/glossary/#vit) backbone whose [features](/shared/glossary/#embedding) work well for many tasks (classification, segmentation, depth) right out of the box, often beating label-trained encoders on a [linear probe](/shared/glossary/#linear-probe). Like a student who learns to recognize objects just by looking at millions of pictures and noticing what stays the same when an object is moved or cropped, never being told any object's name. The "v2" marks the second, larger and cleaner-data version; the name comes from *self-**di**stillation with **no** labels*.

### Disaggregated serving {#disaggregated-serving}
Running prefill and decode on separate GPU pools with KV cache transfer between them

### Discount factor {#discount-factor}
The number γ (gamma), between 0 and 1, that says how much a reward is worth *per step of delay*: a reward `k` steps in the future counts as `γᵏ` times its face value. With γ close to 0 the agent is short-sighted and chases only immediate reward; with γ close to 1 it is patient and plans far ahead. Two reasons it exists: it keeps the infinite sum of future rewards from blowing up (the geometric series `1 + γ + γ² + … = 1/(1−γ)` is finite only when γ < 1), and it encodes a real preference for sooner over later, the way money today is worth more than money next year. Its size sets the [effective horizon](/shared/glossary/#effective-horizon) — roughly how many steps ahead the agent effectively cares about. It is the `γ` in the [Bellman equation](/shared/glossary/#bellman-equation) and one of the five parts of an [MDP](/shared/glossary/#mdp).

### Discriminator {#discriminator}
The "critic" half of a [GAN](/shared/glossary/#gans): a network that looks at an image and outputs how likely it is to be real rather than made by the [generator](/shared/glossary/#generator). It is trained like a detective spotting fakes, and its verdicts are the only teaching signal the generator ever gets — as the discriminator sharpens, the generator is forced to make more convincing images. In [Wasserstein GANs](/shared/glossary/#wasserstein-gan-wgan) it outputs an unbounded score instead of a 0–1 probability and is usually called a *critic*.

### Dispatcher {#dispatcher}
The PyTorch component that routes `torch.foo(...)` calls to the right backend/dtype kernel

### Distillation {#distillation}
Training a smaller "student" model to copy the output of a larger, more capable "teacher" so the student inherits most of the teacher's behavior at a fraction of the cost. Like a junior cook shadowing a head chef and learning each recipe by mimicking the dish — they may never match the master, but they can plate most of the menu for far less money. Distillation works for skills the teacher already has but cannot conjure new abilities the teacher lacks.

### Distribution drift {#distribution-drift}
When the kind of data a model sees in production slowly changes away from the data it was tuned on — like a store whose regular customers gradually change their tastes, so last year's best-selling stock starts to sit on the shelf. For a [quantized](/shared/glossary/#quantization) model it matters because [calibration](/shared/glossary/#calibration) was fitted to the old traffic, so quality can quietly slip as the new traffic drifts further away.

### DiT {#dit}
Diffusion Transformer — Peebles & Xie's diffusion backbone that replaces the [U-Net](/shared/glossary/#u-net) with a pure [transformer](/shared/glossary/#transformer). It chops the noisy image (really its [VAE](/shared/glossary/#vae) [latent](/shared/glossary/#latent-space)) into a grid of small [patches](/shared/glossary/#patchification), turns each patch into a token, and lets [attention](/shared/glossary/#attention) mix them — the same recipe that took over language modeling, now pointed at denoising. The name simply joins "diffusion" (the denoising task) with "transformer" (the architecture). Its big draw is [scaling](/shared/glossary/#scaling-laws): make it wider or deeper and quality improves along a predictable curve, the way a bigger language model reliably gets better. Like swapping a custom-built, image-shaped machine (the U-Net) for a general-purpose assembly line you can just make longer to produce more. Sizes are named DiT-S (small), DiT-B (base), DiT-L (large), and a suffix like "/2" gives the [patch](/shared/glossary/#patchification) size — DiT-S/2 is the small model with 2×2 patches.

### Dolly {#dolly}
A camera move where the whole camera physically travels toward or away from the subject — the name comes from the wheeled cart (a "dolly") that camera operators roll along a track. Unlike a [zoom](/shared/glossary/#zoom), which only magnifies the image from a fixed spot, a dolly actually changes the camera's position, so the background shifts relative to the foreground and you get a real sense of moving through the scene. It is one of the moves a video model can be directed through with [camera control](/shared/glossary/#camera-control).

### Dot product {#dot-product}
A way to boil two equal-length lists of numbers (two *vectors*) down to a single number: multiply them position by position, then add up all the products. For `[1, 2, 3] · [4, 5, 6]` you compute `1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32`.

**What it does (the effect).** The dot product works as a *similarity score* between two vectors. It comes out large and positive when the two lists "point the same way" — their big numbers sit in the same slots — near zero when they are unrelated, and negative when they pull in opposite directions. Analogy: imagine two friends each rate ten movies from −5 to +5. Multiply their scores movie by movie and add them up: if they both loved and both hated the same films the total is a big positive number (very alike); if their tastes are unrelated the pluses and minuses cancel out near zero; if one loved what the other hated it goes negative. The dot product is exactly that "how aligned are we?" number.

**How to calculate it with vectors (and matmul).** Doing one dot product is the multiply-and-sum above. Doing *many* at once is precisely what [matrix multiplication](/shared/glossary/#matmul) (`matmul`, written `A @ B`) is built from: each number in the output grid is the dot product of one row of the left matrix with one column of the right matrix. So a single `matmul` is just a big batch of dot products computed together. This is why, in a [projection discriminator](/shared/glossary/#projection-discriminator), "taking a dot product between the image's features and a learned class vector" is simply measuring how much the image lines up with that class — a high dot product means "this really looks like that category."

### Double backward {#double-backward}
Computing the gradient of a gradient by tracking the backward pass operations in a new computation graph.

### Downstream {#downstream}
The later, real-world tasks a model is eventually judged on — such as question answering or coding — as opposed to the [pretraining](/shared/glossary/#pretraining) objective it was trained on. "Downstream scores" measure how much a change (cleaner data, a better [learning rate](/shared/glossary/#learning-rate)) actually pays off on those end tasks, the way a river's health downstream reflects what happened upstream at the source.

### DPM-Solver {#dpm-solver}
Short for **Diffusion Probabilistic Model Solver** — a fast sampler for [diffusion models](/shared/glossary/#diffusion-model). While the baseline [Euler method](/shared/glossary/#euler-method) blindly takes small straight-line steps based on the immediate slope (requiring hundreds of tiny steps to avoid wandering off-path), DPM-Solver exploits the *known* mathematical curvature of the ODE. By calculating the exact linear parts ahead of time, it can take large, confident strides, effectively reaching the same quality in just 10–20 steps.
**DPM-Solver++:** An upgraded version optimized for high-[CFG](/shared/glossary/#cfg-classifier-free-guidance) environments. High CFG makes the predicted noise direction (ε) oscillate wildly, which confuses the standard DPM-Solver. DPM-Solver++ fixes this by mathematically pivoting the equation to predict the stable *final clean image* (x₀) instead of the wavering noise direction. By anchoring its steps to this unmoving final destination, it safely prevents *image burn and artifacts* — the oversaturated, blown-out colors and blotchy fake textures that show up when too-high guidance pushes pixel values past their valid range, like overexposing a photo until the bright spots turn into harsh white patches — even under aggressive guidance.

### DPO {#dpo}
Direct Preference Optimization — a way to do [RLHF](/shared/glossary/#rlhf)-style alignment that skips the usual two-step machinery (first train a separate [reward model](/shared/glossary/#reward-model), then optimize against it with [PPO](/shared/glossary/#ppo)) and instead tunes the model *directly* on pairs of (chosen, rejected) answers to the same prompt. The trick is a math shortcut — a [closed-form](/shared/glossary/#closed-form) result, meaning an exact formula you can write down directly instead of searching for the answer by trial and error. It proves you never have to build the separate reward model at all: the score that reward model *would* have given is already hidden inside the language model's own answer probabilities (how likely the model thinks each answer is). So one simple training step does the whole job — nudge the model to make the human-preferred answer a little more likely and the rejected answer a little less likely. To stop it from over-correcting and wandering off into nonsense, every nudge is measured *relative to a frozen [reference model](/shared/glossary/#reference-model)* — a saved, unchanging copy of the model from before tuning — so the tuned model can only drift a small distance from where it started, like a climber clipped to a fixed anchor who can move around but not fall far. Like teaching a cook by repeatedly showing them two plates and saying "this one, not that one," instead of first writing a detailed scoring rubric (the reward model) and then training against the rubric. Example: given a prompt and two responses a human marked better/worse, one DPO step nudges the model toward the better one — no reward network and no RL rollouts required, which makes it far simpler and cheaper to run than [PPO](/shared/glossary/#ppo).

### DQN {#dqn}
Deep Q-Network — a reinforcement learning algorithm that combines [Q-learning](/shared/glossary/#q-learning) with a deep neural network, enabling an agent to learn actions in complex environments with large state spaces (like raw game screen pixels). Rather than keeping a giant, impractical lookup table of action values for every possible situation, DQN uses a neural network to *predict* the values of different actions based on the current state. To keep training stable, it relies on two key tricks: **experience replay** (saving past experiences in a 'memory bank' and sampling random mixes of them so the agent does not just learn from its most recent moves) and a **target network** (a frozen, slow-to-update copy of the model used to estimate future rewards, preventing the training targets from shifting constantly). Analogy: learning to drive. Using a lookup table is like memorizing exactly what to do at every coordinate on Earth—impossible. DQN is like learning *rules of thumb* (when a car appears ahead, slow down) that generalize to new roads. The experience replay is like studying a mixed set of video logs from previous driving sessions at night, so you do not forget how to drive in the dark just because you spent the afternoon driving in the sun.

### Draft model {#draft-model}
In [speculative decoding](/shared/glossary/#speculative-decoding), a small, fast model that *guesses* the next few tokens so the big [target model](/shared/glossary/#target-model) can check them all at once. Like a quick assistant who scribbles a rough draft for the expert to approve or correct — cheap to run, and most of its guesses turn out right, so the slow expert is consulted far less often.

### DreamBooth {#dreambooth}
A personalization recipe that [fine-tunes](/shared/glossary/#fine-tuning) the *whole* [diffusion model](/shared/glossary/#diffusion-model) on just 3–5 photos of one subject and binds it to a rare trigger word, so you can afterwards prompt "a photo of [V] dog surfing." Because every [weight](/shared/glossary/#weights) is updated the likeness is excellent, but the saved model is full-sized — the opposite trade-off from a lightweight [LoRA](/shared/glossary/#lora). To stop the model from [catastrophically forgetting](/shared/glossary/#catastrophic-forgetting) what other dogs look like, it adds a [prior-preservation loss](/shared/glossary/#prior-preservation-loss) that keeps training on the model's own generic class images. Like memorizing one specific face in such detail that you must consciously remind yourself other faces still exist.

### DreamerV3 {#dreamerv3}
The third version of *Dreamer* (Hafner et al.), a reinforcement-learning agent that first learns a compact [world model](/shared/glossary/#world-model) of its environment and then trains its [policy](/shared/glossary/#policy) almost entirely *inside* that learned model — "imagining" thousands of future [rollouts](/shared/glossary/#rollout) rather than acting in the real (slow, expensive) environment. This guide borrows only the generative half: predict the next [latent](/shared/glossary/#latent-space) state given the current one and an [action](/shared/glossary/#action-conditioning); the policy-learning loop that sits on top is reinforcement learning's job. DreamerV3 is famous for clearing a wide range of very different tasks — from [Atari](/shared/glossary/#atari) to collecting diamonds in Minecraft — with a *single* fixed set of hyperparameters, which had been an open challenge in RL. The "Dreamer" name captures the core idea: the agent learns by dreaming up experience in its head instead of living through all of it.

### Drift {#drift}
The slow accumulation of small errors across a sequence — frame after frame, shot after shot, or step after step — until the result wanders noticeably away from where it began. Each individual step is only *slightly* off, but because every new piece is built on top of the previous (already-imperfect) one, those tiny mistakes pile up and compound. In long video generation it shows up as colors creeping, a character's face or outfit slowly morphing into someone else's, or a scene losing its original layout the further you travel from the opening frame. Like photocopying a photocopy of a photocopy: any single copy looks fine, but after a hundred generations the image has visibly degraded. Fixes work by repeatedly re-anchoring to a fixed reference so the errors cannot keep stacking up — for example pinning a character's identity with an [IP-Adapter](/shared/glossary/#ip-adapter) or character [LoRA](/shared/glossary/#lora), or overlapping and blending clips in [sliding-window generation](/shared/glossary/#sliding-window-generation).

### dtype {#dtype}
A tensor's element data type — e.g. `float32`, `float16`, `bfloat16`, `int8`, `bool`

### Dual encoder {#dual-encoder}
An architecture with *two* separate encoders — one per [modality](/shared/glossary/#modality), e.g. an image tower and a text tower — that each map their input into the *same* shared space, where the two are compared by [cosine similarity](/shared/glossary/#cosine-similarity). The key trait is that the modalities never mix until the very end ([late fusion](/shared/glossary/#fusion-earlymiddlelate)): each side is encoded entirely on its own. [CLIP](/shared/glossary/#clip) is the classic example. Analogy: two translators who never talk to each other but have both been trained to render anything into one common interlingua, so their outputs can be lined up afterward. This separation is exactly what makes dual encoders fast for [cross-modal retrieval](/shared/glossary/#cross-modal-retrieval) — you encode the whole image collection *once*, in advance, and a new text query only has to be encoded and compared — but it also means they can only *match* or *score*, never *reason over* or *generate* the other modality the way a [VLM](/shared/glossary/#vlm) can.

### Dynamic computation graph {#dynamic-computation-graph}
A graph of operations built on-the-fly as code executes, representing the forward pass used for autograd.

### Dynamic programming {#dynamic-programming}
In RL, the family of algorithms — [policy iteration](/shared/glossary/#policy-iteration) and [value iteration](/shared/glossary/#value-iteration) — that solves an [MDP](/shared/glossary/#mdp) *exactly* using a fully known model: the [transition probabilities](/shared/glossary/#transition-function) and the [reward function](/shared/glossary/#reward-function). "Dynamic programming" is Richard Bellman's term for cracking a hard problem by breaking it into smaller overlapping subproblems and reusing their answers; here the subproblem is "what is each state worth?" and the reuse comes from the [Bellman backup](/shared/glossary/#bellman-operator), which writes a state's value in terms of its neighbors' values. It is the *planning* baseline that the rest of RL approximates once the model is unknown (meaning the agent does not know the rules of the world, like exactly where a step will take it or what reward it will get) and you can only sample [transitions](/shared/glossary/#transition-function) by actually acting in the environment.

### Dynamic quantization {#dynamic-quantization}
A [quantization](/shared/glossary/#quantization) method that stores [weights](/shared/glossary/#weights) as [int8](/shared/glossary/#int8) ahead of time but computes each layer's [activation](/shared/glossary/#activations) scale at runtime, just before the layer runs.

### Eager mode {#eager-mode}
PyTorch's default execution, where each operation runs immediately as its Python line is reached — flexible and easy to debug, but without the cross-operation optimizations a compiler can apply.

### EAGLE / Medusa {#eagle--medusa}
Self-speculation: extra heads on the target model propose tokens, no separate draft model

### Earth Mover's Distance {#earth-movers-distance}
A way to measure how far apart two distributions are by the smallest amount of "work" needed to reshape one pile into the other — imagine shovelling a heap of dirt into the shape of a second heap, where work is dirt moved times distance carried. Also called the Wasserstein distance, it gives a smooth, meaningful number even when the two piles barely overlap, which is exactly why [Wasserstein GANs](/shared/glossary/#wasserstein-gan-wgan) use it in place of the original [GAN](/shared/glossary/#gans) loss that goes flat in that case.

### Edge inference {#edge-inference}
Running a model directly on the device in front of the user — a phone, laptop, car, or small embedded board — instead of sending the request to a data-center GPU. Like cooking at home rather than ordering delivery: it is private and works without a network, but you are limited to the small "kitchen" the device has, so models are kept small (1–8B), heavily [quantized](/shared/glossary/#quantization), and tuned to sip battery and fit in shared memory.

### EDM {#edm}
A cleaned-up reformulation of [diffusion](/shared/glossary/#diffusion-model) from the paper *Elucidating the Design Space of Diffusion-Based Generative Models* (Karras et al. 2022) that strips away historical baggage and makes training and sampling much easier to tune. Two ideas carry it: index noise by its standard deviation σ rather than a discrete timestep (the [σ-schedule](/shared/glossary/#σ-schedule-karras)), and *precondition* the network — rescale its input, output, and per-σ loss weight so it always sees roughly unit-variance signals no matter how much noise is present. The result is a flat, forgiving hyperparameter surface. A useful rule of thumb: if a 2020-era diffusion paper feels obscure, restate it in EDM's language and it usually becomes obvious.

### Effective horizon {#effective-horizon}
A rough measure of how many steps into the future an agent's choices actually take into account, set by the [discount factor](/shared/glossary/#discount-factor): about `1/(1−γ)` steps. The intuition is that rewards beyond this point have been shrunk by so many factors of γ that they barely move the decision. For example γ = 0.9 gives `1/(1−0.9) = 1/0.1 = 10` steps, γ = 0.99 gives `1/0.01 = 100` steps, and γ = 0.999 gives 1,000 — so nudging γ from 0.99 to 0.999 makes the agent plan ten times farther ahead. Like a flashlight beam that γ widens or narrows: it sets how far down the road you can see well enough to steer by.

### EKF {#ekf}
Extended Kalman Filter — Kalman filter linearized about the current estimate

### ELBO {#elbo}
Evidence Lower Bound — a mathematical score used to train generative models like [VAEs](/shared/glossary/#vae). It balances two goals: recreating the original input accurately, and keeping the model's internal representation organized. Think of it like packing for a trip: you want to bring everything you need (accurate recreation) but also pack it neatly so the suitcase closes easily (organized representation). The ELBO is the score that measures how well the model balances both tasks.

### Elementwise operation {#elementwise-operation}
An operation applied independently to each element of a tensor (e.g. add, multiply, ReLU), where output position `i` depends only on input position `i`.

### Eligibility traces {#eligibility-traces}
A short-term memory that lets a single [TD error](/shared/glossary/#td-error) update many past states at once, instead of only the state where the error appeared. Each state carries a "trace" — a number that jumps up when the state is visited and then fades by a factor of `γλ` every step — and when a TD error arrives, every state is nudged in proportion to its current trace, so more recently visited states get more credit. 

The decay parameter λ tunes the [bias–variance](/shared/glossary/#bias-variance-tradeoff) balance continuously: λ = 0 gives plain one-step [TD(0)](/shared/glossary/#temporal-difference-learning), λ = 1 reproduces [Monte Carlo](/shared/glossary/#monte-carlo-method), and the method in between is called TD(λ). Think of this like diagnosing food poisoning at the end of the day: if λ = 0 (one-step TD), you only blame the very last thing you ate (a late-night mint); if λ = 1 (Monte Carlo), you blame everything you ate all day equally; and if λ is in between, you mostly blame the recent dinner, less so lunch, and barely blame breakfast. 

*Replacing* traces cap a revisited state's trace at 1 rather than adding to it, which stops a state visited in a loop from earning unrealistically large credit. Picture a fading scent trail behind you: when reward finally appears, the freshest parts of the trail feel the strongest pull, but replacing traces ensures a spot you walked in a circle over doesn't smell overpoweringly strong, just "fresh."

### Elo {#elo}
A rating system borrowed from chess that turns a series of head-to-head wins and losses into a single number per player: beat a strong opponent and your rating jumps, lose to a weak one and it drops. LLM [arenas](/shared/glossary/#arena) use it to rank chat models from pairwise comparisons instead of from a fixed-answer [benchmark](/shared/glossary/#benchmark).

### EMA weights {#ema-weights}
Exponential moving average of model weights; samples better than the live weights

### Embedding {#embedding}
A dense vector that represents a token (or other item) so the model can compute over it; each token ID maps to one row of the [embedding matrix](/shared/glossary/#embedding-matrix)

### Embedding matrix {#embedding-matrix}
The lookup table `E ∈ ℝ^{V×d}` that turns each token ID into a dense vector by selecting its row; growing the [vocabulary](/shared/glossary/#vocabulary) means adding rows

### Embedding space {#embedding-space}
The shared multi-dimensional space that all [embeddings](/shared/glossary/#embedding) live in, where every item is a point and *direction and distance carry meaning* — items that mean similar things sit close together and point the same way. Talking about the *geometry* of the embedding space means asking how those points are arranged: are the true image–caption pairs bunched into tight clusters, spread evenly over the unit sphere, or collapsed into one indistinct blob? Analogy: a city where related shops naturally form neighborhoods — you learn a lot about a model by inspecting the *shape* of its map, not just whether it gets answers right. In [CLIP](/shared/glossary/#clip), a well-chosen [temperature](/shared/glossary/#temperature) pulls matching pairs into tight clusters on the sphere while keeping mismatches pushed apart, so the geometry itself reveals how confidently the model separates right from wrong.

### EMO {#emo}
Emote Portrait Alive — a specific [talking-head](/shared/glossary/#talking-head) model that turns a single portrait photo and an audio clip into a highly expressive, singing or talking video. Unlike older models that only move the mouth, EMO predicts motion directly without relying on intermediate 3D face models or facial landmarks.

### Enormous on paper {#enormous-on-paper}
Describes a model like a [Mixture-of-Experts (MoE)](/shared/glossary/#moe) that has a massive total number of parameters (e.g., 100 billion), but only uses a small fraction of them (e.g., 10 billion) for any single word. Like a giant university with 5,000 courses listed in its catalog (enormous on paper)—no single student takes all 5,000 courses. Each student only takes a few classes at a time, so the cost per student remains low, even though the total catalog is huge.

### Epsilon-greedy {#epsilon-greedy}
The simplest way to balance [exploration and exploitation](/shared/glossary/#exploration-vs-exploitation), usually written ε-greedy (ε is the Greek letter *epsilon*, standing for a small probability). With probability `1 − ε` the agent acts [greedily](/shared/glossary/#greedy-policy) — it picks the action its current estimates rate best — and with probability ε it instead picks a uniformly random action, just to keep trying options it might be undervaluing. ε is normally *annealed* (decayed) from near 1 down to a small floor over training, so the agent explores wildly at first and increasingly exploits what it has learned. It is "good enough" when rewards are dense but fails badly when reward is rare, because random jitter almost never stumbles onto a far-off goal.

### Error budget {#error-budget}
The small amount of failure an [SLO](/shared/glossary/#slo) allows. If your target is 99.9% success, the remaining 0.1% — about 43 minutes a month — is your error budget. Like a monthly data allowance on a phone plan: you can "spend" it on risky deploys and experiments, but once it runs out you stop taking risks until it resets. It turns reliability from a vague goal into a balance you can watch.

### Euler method {#euler-method}
The simplest way to numerically solve a differential equation: look at the slope where you currently are, take one straight-line step in that direction, then repeat. It is easy to implement but accumulates error quickly because it ignores how the slope changes *during* the step, so diffusion samplers built on it need many steps to stay accurate. Like steering a car by only ever looking at the road directly under the bumper. Contrast with [Heun's method](/shared/glossary/#heuns-method) and [DPM-Solver](/shared/glossary/#dpm-solver), which correct for the changing slope and so need far fewer steps.

### Evaluation harness {#evaluation-harness}
A ready-made framework that runs a model through many [benchmarks](/shared/glossary/#benchmark) automatically, with the prompts, answer parsing, and scoring all fixed so every model is tested the exact same way. Like a standardized testing center that hands every candidate the same paper and grades it with the same answer key, instead of each examiner writing their own quiz. For [VLMs](/shared/glossary/#vlm) the two common ones are *lmms-eval* and *VLMEvalKit*: you point them at a model and a list of benchmarks ([MMBench](/shared/glossary/#mmbench), [MMMU](/shared/glossary/#mmmu), DocVQA, …) and they return one comparable table of scores. This matters because tiny differences in prompt wording or in how a multiple-choice letter is pulled out of the answer can swing a score by several points, so sharing one harness is what makes two papers' numbers actually comparable. Example: a single command like `python -m lmms_eval --model llava --tasks mmbench,mmmu` evaluates the model on both and prints an accuracy for each.

### ExecuTorch {#executorch}
PyTorch's lightweight runtime for running models on mobile and edge devices, built on the graph captured by [`torch.export`](/shared/glossary/#torchexport).

### Expert {#expert}
In a [Mixture-of-Experts (MoE)](/shared/glossary/#moe), one of several parallel [MLP](/shared/glossary/#mlp) sub-networks; a router sends each token to only the top few experts instead of all of them. Like a hospital triage desk that routes each patient to the right specialist rather than making everyone see every doctor — lots of expertise on hand, but only a little used per case.

### Expert parallelism (EP) {#expert-parallelism-ep}
For MoE models, distributing experts across GPUs with [all-to-all token routing](/shared/glossary/#all-to-all-token-routing)

### Exploration vs exploitation {#exploration-vs-exploitation}
The core tension in RL: should the agent *exploit* — take the action it currently believes is best — or *explore* — try something uncertain that might turn out better? Pure exploitation can lock onto a mediocre habit because the agent never gathers the evidence that a better option exists; pure exploration never cashes in what it has learned. It is like always reordering your favorite dish versus occasionally trying something new on the menu — you need a mix of both. [Epsilon-greedy](/shared/glossary/#epsilon-greedy), softmax (Boltzmann) action selection, and UCB (which adds an "optimism" bonus to rarely tried actions) are common ways to strike the balance.

### Exponent {#exponent}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that records its *scale* — how many places to shift the decimal point. In scientific notation like `3.5 × 10¹²`, the `12` is the exponent (using base 10 instead of base 2). More exponent bits give a wider range of representable magnitudes, from astronomically large to vanishingly small; fewer exponent bits mean values overflow or [underflow](/shared/glossary/#underflow) more easily. This is why [FP8](/shared/glossary/#fp8) has two flavors: **E5M2** (5 exponent bits) for gradients that can swing wildly in size, and **E4M3** (4 exponent bits) for activations that stay in a tighter range. See also [mantissa](/shared/glossary/#mantissa).

### Extrapolation {#extrapolation}
Extending a learned pattern *beyond* the range of values seen during training — working out what comes *outside* the examples, rather than filling a gap *between* two of them (that in-between case is *interpolation*). A model that **extrapolates** gracefully keeps behaving sensibly on inputs that are larger, longer, or differently shaped than anything in its training data. This is the whole reason [RoPE](/shared/glossary/#rope) encodes position as a *rotation*: a rotation angle is defined for *any* position, including ones past the longest sequence the model ever trained on, so a model trained on short clips can still place a token at a never-seen position and generate longer videos at [variable resolution](/shared/glossary/#variable-resolution). Contrast a fixed lookup table of learned position vectors, which simply has no entry for a position it never saw. Like a tide chart built from one month of readings: *interpolating* tells you the water level at a moment between two marks, while *extrapolating* predicts next week's tide — beyond every reading you have.

### FCFS {#fcfs}
First-Come, First-Served — the simplest scheduling rule: handle requests in the exact order they arrive, like a single queue at a bakery where nobody can skip ahead. It is fair and easy to build, but it has no sense of deadlines, so one slow request at the front can make everyone behind it late.

### FFN {#ffn}
Feed-Forward Network — the small [MLP](/shared/glossary/#mlp) inside each [transformer](/shared/glossary/#transformer) block. *Position-wise* means it is applied to each token (each position in the sequence) on its own, reusing the **same** weights at every position — like one cashier serving each customer in line one at a time at the same till, never letting them interact. That is the opposite of the [attention](/shared/glossary/#attention) sublayer, where tokens *do* look at each other; the FFN just lets each token "think" by itself.

### F/T sensor {#ft-sensor}
Force/Torque sensor — six-axis force and moment at a wrist or fingertip

### Farnebäck optical flow {#farnebäck-optical-flow}
A classical (non-neural) algorithm for computing dense [optical flow](/shared/glossary/#optical-flow), named after its inventor Gunnar Farnebäck. It estimates motion by approximating the brightness around each pixel with a small quadratic (a smooth curved surface) in both frames and solving for the shift that lines them up. Analogy: it slides a tiny transparent patch of the first frame around the second until it clicks into place, and records how far it had to move. It is fast and needs no training, but it is less accurate on large or blurry motion than a learned model like [RAFT](/shared/glossary/#raft). Example: OpenCV's `cv2.calcOpticalFlowFarneback` returns a `(H, W, 2)` array giving each pixel's left–right and up–down movement.

### FID {#fid}
Fréchet Inception Distance — the standard sample-quality metric for image generation. ("Fréchet," after the mathematician Maurice Fréchet, names the *Fréchet distance*: a way to measure how far apart two probability distributions sit.) It runs both real and generated images through a pretrained [Inception network](/shared/glossary/#inception-network) to turn each image into a feature vector, then measures how far apart the two [clouds](/shared/glossary/#point-cloud) of features sit by comparing their means and [covariances](/shared/glossary/#covariance) (their centers and spreads). A lower FID means the generated images look statistically more like the real ones — picture two overlapping clouds of dots: the more they overlap, the smaller the distance. The real images here are only a *yardstick*, not an ingredient: your model invents brand-new images from random noise and never copies the real ones — FID simply needs a pile of real photos to compare those inventions against so it can score how convincing they are.

### FILM {#film}
FILM (Frame Interpolation for Large Motion) is a neural [frame-interpolation](/shared/glossary/#frame-interpolation) model from Google that, given two real frames, generates the frames in between — and it is specifically built to cope when objects move a *long* way between the two shots, the case where older methods smear or tear. It estimates motion at several scales at once (a coarse pass catches big jumps, finer passes catch small ones) and warps both frames toward the middle before blending them. Think of an animation assistant who can fill in the missing "in-between" drawings between two key poses even when the character has leapt clear across the scene. It is a convenient pretrained model for seeing, firsthand, the artifacts that fast motion produces.

### Filterbank {#filterbank}
A stack of band-pass filters that each measure how much energy a signal carries in one narrow frequency range — together they split a sound into a set of frequency "buckets." A *mel filterbank* is the specific set used to build a [mel spectrogram](/shared/glossary/#mel-spectrogram): commonly 80 filters, each shaped by [triangular weights](/shared/glossary/#triangular-weights) and spaced on the perceptual mel scale, all stored as one fixed matrix. Applying it is a single matrix multiply that collapses the [STFT](/shared/glossary/#stft)'s hundreds of evenly spaced frequency rows down to a handful of [mel bands](/shared/glossary/#mel-bands). Like a row of differently tuned wine glasses, each ringing only for the note near its own pitch: play a chord and you can read off how much of each note is present from how loudly each glass hums. Example: a 1024-point FFT produces ~513 frequency values per frame; multiplying by an 80×513 mel filterbank matrix turns each time frame into just 80 numbers.

### Fine-tuning {#fine-tuning}
Taking a model that was already trained on a huge dataset and training it a little further on a small, specific dataset so it picks up a new skill, subject, or style. The big initial training is expensive and done once; fine-tuning is cheap and reuses all that knowledge — like hiring an experienced cook and teaching them your three house recipes rather than training someone from scratch. In image generation you might fine-tune [Stable Diffusion](/shared/glossary/#stable-diffusion) on 20 photos of your pet so it can draw that specific pet. Fine-tuning can update *every* [weight](/shared/glossary/#weights) (as in [DreamBooth](/shared/glossary/#dreambooth)) or just a tiny added piece (as in [LoRA](/shared/glossary/#lora)); the less you change, the smaller and more shareable the result, at some cost in how much new behavior you can absorb.

### FineWeb-Edu {#fineweb-edu}
A large, openly released [pretraining](/shared/glossary/#pretraining) dataset built by running a [quality filter](/shared/glossary/#quality-filter) over crawled web pages and keeping only the educational-looking ones — like skimming a huge pile of internet text and saving just the pages that read like a textbook. Models trained on it often beat models trained on far more unfiltered text, making it a go-to example that data quality can matter more than raw quantity.

### FK / IK {#fk--ik}
Forward / Inverse Kinematics — compute end-effector pose from joints or vice versa

### Flamingo {#flamingo}
DeepMind's 2022 [vision-language model](/shared/glossary/#vlm) that pioneered *[gated](/shared/glossary/#gated) [cross-attention](/shared/glossary/#cross-attention)*: it leaves a big pretrained language model entirely [frozen](/shared/glossary/#frozen) and inserts brand-new cross-attention layers between its blocks so the text can look at image features. The clever part is the gate — a learned multiplier that starts at exactly zero, so on the very first training step the new layers contribute *nothing* and the model behaves identically to the original language model, then the gate slowly opens as training teaches it how much image information to let in. Like adding a new water line to a working house but keeping its valve shut until you have checked every joint, then easing it open. This "don't break what already works, blend the new capability in gradually" trick is why Flamingo could bolt vision onto a frozen LLM without destabilizing it, and it became a template later VLMs copied. The [projector](/shared/glossary/#projector)-only approach of [LLaVA](/shared/glossary/#llava) is the simpler rival design.

### FlashAttention {#flashattention}
A much faster way to compute [attention](/shared/glossary/#attention) that never writes the giant token-by-token score table to slow [HBM](/shared/glossary/#hbm) memory. Plain attention builds the full `T × T` grid of how strongly every token attends to every other token, parks it in HBM, then reads it back — a flood of slow memory traffic. FlashAttention instead works on small tiles inside the chip's fast on-chip memory (SRAM) and keeps a running total, so the huge grid never has to be stored at all. Like adding up a long column of numbers in your head as you go instead of writing every subtotal on paper — same answer, far fewer trips to the slow notebook. Every modern inference engine relies on it.

### FlashDecoding {#flashdecoding}
A version of [FlashAttention](/shared/glossary/#flashattention) tuned for the [decode](/shared/glossary/#decode) step, where there is just one new query token but a long [KV cache](/shared/glossary/#kv-cache) to read. It splits that long read across many GPU workers so the [HBM](/shared/glossary/#hbm) bandwidth stays fully used instead of one worker plodding through the cache alone — the trick that lets engines like [vLLM](/shared/glossary/#vllm) hit near-peak bandwidth on decode-heavy traffic.

### float16 {#float16}
16-bit floating-point format (`fp16`); saves memory and can be fast on GPUs, but has a limited range (max ~65,504) that can cause [underflow](/shared/glossary/#underflow) when accumulating very small values

### float32 {#float32}
32-bit floating-point format (`fp32`); the standard default precision for PyTorch tensors — wide enough range and enough precision for most training and inference tasks

### FLOPs {#flops}
Floating-Point Operations — a count of the individual arithmetic steps (additions and multiplications on decimal numbers) a model performs, used as a hardware-independent measure of how much compute one forward pass costs. You estimate it by adding up the work in each layer: a [matrix multiply](/shared/glossary/#matmul) of an M×K matrix by a K×N one, for instance, takes about 2·M·K·N FLOPs (each of the M·N outputs needs K multiplies and K adds). Like counting the total pencil strokes a calculation requires, regardless of how fast the person writing them is. More FLOPs means a slower, costlier model — exactly the price you pay when a [ViT](/shared/glossary/#vit) uses smaller [patches](/shared/glossary/#patch). (Note: "FLOPs" = operations; "FLOP/s" with a slash = operations *per second*, a speed.)

### Flow matching {#flow-matching}
Training a *velocity field* — a model that, given a half-noisy image and a time, predicts which direction and how fast to move it toward a clean image — so that following those arrows turns pure noise into data. Concretely, you draw a straight line between a real image `x_0` and random noise `ε`, pick a random point on that line, and train the model to output the line's direction `ε - x_0`; at generation time you start at noise and repeatedly step along the predicted arrows (solving an [ODE](/shared/glossary/#ode)) until you arrive at a clean image. It is a simpler, more modern alternative to [DDPM](/shared/glossary/#ddpm): there is no [noise schedule](/shared/glossary/#noise-schedule) to tune, just one clean regression target. Like learning the wind currents over a map so that, dropped anywhere in the fog, you always know which way blows toward home.

### Flux {#flux}
A family of state-of-the-art open-weight text-to-image models released in 2024 by Black Forest Labs (a team that included original [Stable Diffusion](/shared/glossary/#stable-diffusion) researchers). Flux is built on a large [MMDiT](/shared/glossary/#mmdit) backbone trained with [rectified flow](/shared/glossary/#rectified-flow), so text and image tokens share the same [attention](/shared/glossary/#attention) layers and the model denoises along nearly straight paths — which is why it follows detailed prompts and renders legible text unusually well. It ships in a few flavors: a top-quality "pro" version, an open "dev" version for tinkering, and a distilled "schnell" (German for *fast*) version that trades a little quality for very few sampling steps. Think of it as the generation of image models that arrived just after SD3 and pushed open-weight quality a notch higher.

### Forensics {#forensics}
Working backward from a training failure to the operation that first caused it, instead of chasing the visible symptom. In PyTorch this means turning on [autograd](/shared/glossary/#autograd) [anomaly detection](/shared/glossary/#anomaly-detection) to halt at the first [NaN](/shared/glossary/#nan) or bad [gradient](/shared/glossary/#gradients).

### FP4 {#fp4}
4-bit floating point — half the bits of [FP8](/shared/glossary/#fp8) again, so a weight takes a quarter of the space of [bfloat16](/shared/glossary/#bfloat16). With only 4 bits there are just 16 possible values, so it sits near the edge of usable precision and needs careful checking; newer [Blackwell](/shared/glossary/#blackwell) GPUs accelerate it in hardware, making it attractive for squeezing huge models onto fewer chips.

### Forward hook {#forward-hook}
A callback registered on an `nn.Module` that PyTorch calls automatically after the module's forward pass, receiving the input and output tensors; used for capturing activations and debugging

### Forward pass {#forward-pass}
One complete run of an input through the *whole* network — every layer in order, from the first to the last — to produce an output (for an [LLM](/shared/glossary/#llm), the [logits](/shared/glossary/#logits) for the next token). It means start-to-finish through *all* the layers, not a single layer. Like running a part down an entire assembly line once to get the finished product. The reverse direction, used in training to compute [gradients](/shared/glossary/#gradients), is the [backward pass](/shared/glossary/#backward-pass).

### Fourier transform {#fourier-transform}
A math tool that takes a signal that changes over *time* — like a sound wave — and reveals which pure *frequencies* (pitches) it is secretly built from, and how much of each. Like a glass prism splitting white light into its rainbow of colors, the Fourier transform splits a messy sound into the simple sine-wave "tones" hidden inside it. Concrete example: feed it a recording of a piano chord and it answers "this is mostly 262 Hz (middle C) + 330 Hz (E) + 392 Hz (G)". **How it works:** it slides every candidate frequency past the signal, multiplies the two together point by point, and adds up the products (a [dot product](/shared/glossary/#dot-product)); when a test frequency really is present the bumps line up and the sum comes out large, and when it is absent the products cancel to near zero — so a big result means "yes, that pitch is in here." The catch is that it tells you *which* frequencies are present across the whole clip but not *when* each one happened, which is exactly why the [STFT](/shared/glossary/#stft) runs it on short overlapping slices instead. It is named after Joseph Fourier, who showed that any repeating signal can be rebuilt by adding up enough simple sine waves.

### FP8 {#fp8}
8-bit floating point — half the bits of [bfloat16](/shared/glossary/#bfloat16). Comes in two flavors: **E4M3** (4 [exponent](/shared/glossary/#exponent) bits + 3 [mantissa](/shared/glossary/#mantissa) bits) keeps a bit more precision and is used for [weights](/shared/glossary/#weights) and the forward [activations](/shared/glossary/#activations); **E5M2** (5 [exponent](/shared/glossary/#exponent) + 2 [mantissa](/shared/glossary/#mantissa)) trades precision for a wider range and is used for gradients, which can be very large or very small. Supported by [Hopper](/shared/glossary/#hopper) and later NVIDIA GPUs, it is rapidly becoming the modern default serving precision.

### Fragmentation {#fragmentation}
Memory wasted in gaps too small to reuse, left behind when each request is given its own contiguous chunk — like a parking lot full of single empty spaces where no bus can fit even though there is plenty of total room. Paged schemes such as [PagedAttention](/shared/glossary/#pagedattention) avoid it by handing out small fixed-size pages instead of one big block per request.

### Frame interpolation {#frame-interpolation}
Generating new frames *between* two existing ones to make motion smoother or a clip slower — turning, say, 24 frames per second into 60. It is sometimes called "video generation lite" because the model only has to invent the short motion between two anchors it can already see, not a whole scene from nothing. The classic analogy is hand-drawn animation: a lead artist draws the key poses and an assistant fills in the "in-between" frames — the industry literally calls this *inbetweening*. Modern neural versions such as [FILM](/shared/glossary/#film) and Super SloMo estimate how each pixel moves between the two frames (closely related to [optical flow](/shared/glossary/#optical-flow)) and warp the images toward the midpoint.

### Frame rate (fps) {#frame-rate-fps}
How many still frames a video shows per second — "fps" stands for *frames per second* (e.g. 24, 30, 60). It sets how much real-world time sits between two neighboring frames, so the *same* motion looks bigger and choppier at low fps and smoother at high fps. Analogy: a flipbook drawn with 12 pages per second looks jerky; the same drawings at 60 pages per second look fluid. Example: sampling 16 frames evenly from a 2-second clip at 8 fps covers the whole clip, but grabbing 16 *consecutive* frames from a 60-fps clip covers only a quarter-second — so a model must be told which fps it is seeing.

### Frontier run {#frontier-run}
A training run for one of the largest, most capable models at the leading edge of what is currently possible — the kind that ties up thousands of GPUs for weeks and costs millions of dollars. Because the stakes are so high, a [loss spike](/shared/glossary/#loss-spike) that cannot be recovered cleanly can throw away days of that compute, which is why teams rehearse [checkpoint](/shared/glossary/#checkpoint) recovery on small models first.

### Frozen {#frozen}
A layer or whole sub-network is *frozen* when its [weights](/shared/glossary/#weights) are held fixed during training — the [optimizer](/shared/glossary/#optimizer) is told to skip them, so no [gradients](/shared/glossary/#gradients) update them — while other parts of the model keep learning. Like renovating one room of a house while the rest stays sealed off and untouched. Freezing is how you reuse an expensive pretrained component (a [CLIP](/shared/glossary/#clip) image encoder, a big language model) as a fixed feature extractor and train only a small new piece — a [projector](/shared/glossary/#projector), an [adapter](/shared/glossary/#adapter), or a [LoRA](/shared/glossary/#lora) — on top: it saves memory and compute and protects the pretrained knowledge from being overwritten by a small new dataset. The opposite is leaving a part *trainable* (or "unfrozen"), as [fine-tuning](/shared/glossary/#fine-tuning) does.

### FrozenLake {#frozenlake}
A small gridworld environment in [Gymnasium](/shared/glossary/#gymnasium) (`FrozenLake-v1`) where the agent crosses a frozen lake from start to goal without falling into holes. In its default *slippery* mode the ice is stochastic: the chosen direction only happens part of the time, and the agent often slides to a perpendicular square instead — which makes the [transition probabilities](/shared/glossary/#transition-function) genuinely random and the problem a real [MDP](/shared/glossary/#mdp) rather than a deterministic maze. Because its dynamics are small and fully known, it is the standard first testbed for both [value iteration](/shared/glossary/#value-iteration) (planning with the model) and [Q-learning](/shared/glossary/#q-learning) (learning without it).

### F.scaled_dot_product_attention {#fscaled_dot_product_attention}
PyTorch's built-in fused [attention](/shared/glossary/#attention) function (in `torch.nn.functional`) that computes `softmax(QKᵀ/√d)·V` in a single call, dispatching to an optimized [backend](/shared/glossary/#backend) such as [FlashAttention](/shared/glossary/#flashattention).

### FSDP {#fsdp}
Fully Sharded Data Parallel — shard params, grads, and optimizer state across [ranks](/shared/glossary/#rank)

### FSQ {#fsq}
Finite Scalar Quantization — a way to make discrete image [tokens](/shared/glossary/#token-visualaudio) *without* a learned [codebook](/shared/glossary/#codebook). Instead of looking up the nearest entry in a trained table, it simply rounds each coordinate of the latent to the nearest value on a fixed grid, like snapping every measurement to the nearest tick on a ruler. Because there is nothing to train in the quantizer, it is simpler and sidesteps [codebook collapse](/shared/glossary/#codebook-collapse), yet stays competitive with [VQ-VAE](/shared/glossary/#vq-vae).

### Function calling {#function-calling}
The mechanism by which a model uses a tool: it emits a structured request (such as JSON naming a function and its arguments), an external program runs that request, and the result is handed back to the model. Also called tool use.

### Fusion (early/middle/late) {#fusion-earlymiddlelate}
*Where* in the network the information from different [modalities](/shared/glossary/#modality) is combined. **Late fusion** encodes each modality fully on its own and only compares the two finished [embeddings](/shared/glossary/#embedding) at the very end ([CLIP](/shared/glossary/#clip) matching an image vector to a text vector). **Middle fusion** encodes each separately but then lets one stream attend to the other partway through, usually with [cross-attention](/shared/glossary/#cross-attention) (a [VLM](/shared/glossary/#vlm) feeding image features into a language model). **Early fusion** turns every modality into one shared stream of [tokens](/shared/glossary/#token-visualaudio) from the very start and runs a single model over the mix ([native multimodal](/shared/glossary/#native-multimodal) models like [Chameleon](/shared/glossary/#chameleon)). Think of three ways to combine a recipe's flavors: stir two finished sauces together at the table (late), blend them while each is still simmering (middle), or throw every raw ingredient into one pot from the beginning (early). The earlier the fusion, the more freely the modalities can shape each other — but the more compute and data it takes to train.

### Future frame prediction {#future-frame-prediction}
The task of, given the first few frames of a video, predicting the frames that come next — an early benchmark for whether a model has learned how things move. It is the video cousin of next-word prediction in language: the model is trained to continue a sequence it has only partly seen. The classic toy benchmark is [Moving MNIST](/shared/glossary/#moving-mnist) and the classic baseline architecture is the [ConvLSTM](/shared/glossary/#convlstm). Because the future is genuinely uncertain, a simple model trained with [mean squared error](/shared/glossary/#mse-mean-squared-error) tends to hedge by *blurring* — averaging all the plausible futures into one fuzzy guess rather than committing to a single sharp one.

### FVD {#fvd}
Fréchet Video Distance — the standard automatic quality metric for video generation, and the direct extension of image [FID](/shared/glossary/#fid) to clips. ("Fréchet," after mathematician Maurice Fréchet, names the *Fréchet distance* between two probability distributions.) Like FID, it runs both real and generated videos through a pretrained network — here a video-understanding network (I3D) that watches a clip and summarizes its appearance *and* motion into a feature vector — then measures how far apart the two clouds of feature vectors sit by comparing their means and spreads ([covariances](/shared/glossary/#covariance)); a lower FVD means the generated clips look and move more like real ones. It is widely criticized because the score often disagrees with human judgment — a clip people clearly dislike can still post a good FVD — which is exactly why suites like [VBench](/shared/glossary/#vbench) break quality into many separate dimensions instead of trusting one number.

### GAE {#gae}
Generalized Advantage Estimation — TD(λ) for advantages

### GameNGen {#gamengen}
A 2024 Google system (the name reads "Game-N-Gen", i.e. *game engine*) that showed a neural network can run a playable game entirely by generating its frames, with no traditional game code underneath. It is an [action-conditioned](/shared/glossary/#action-conditioning) [diffusion model](/shared/glossary/#diffusion-model) trained on recorded play of the classic shooter DOOM: given the recent frames plus the player's current button press, it predicts the next frame, fast enough to play in real time at about 20 [frames per second](/shared/glossary/#frame-rate-fps). It is a landmark demonstration that a learned [world model](/shared/glossary/#world-model) can stand in for a hand-written engine — the screen you see is being *dreamed*, frame by frame, in response to your controls.

### GAN inversion {#gan-inversion}
Running a [GAN](/shared/glossary/#gans) backwards: given a real photo, find the input [latent](/shared/glossary/#latent-space) code that makes the [generator](/shared/glossary/#generator) reproduce it. A trained generator only goes code → image, so inversion recovers the missing code either by optimizing it to lower reconstruction error or by training an encoder to predict it in one shot. It is the step that lets you *edit* a real image — once you have its code, nudging the code changes the picture.

### GANs (Generative Adversarial Networks) {#gans}
A class of generative models that trains two networks in a contest. A [generator](/shared/glossary/#generator) turns random noise into fake images, and a [discriminator](/shared/glossary/#discriminator) tries to tell those fakes from real ones; each one makes the other better, like a counterfeiter and a detective locked in an arms race. At the end you keep the generator, which by then makes images realistic enough to fool a well-trained critic. GANs produce sharp samples but are famously unstable to train — see [mode collapse](/shared/glossary/#mode-collapse).

### Gated {#gated}
An operation where one path of a neural network controls how much of another path gets through, by multiplying the two together value-by-value (*element-wise multiplication*). Picture a row of dimmer switches — or the valves on a bank of faucets. The main path carries the information; the second path produces a "gate" number for each value, and that number turns the corresponding value up or down. A gate near **0** shuts a value off (nothing passes), a gate near **1** lets it through untouched, and anything in between is a partial dribble. Because the multiply happens one number at a time, *every* feature gets its own private valve, so the network can wave some details through while damping others — all decided on the fly from the input. This is the trick behind [LSTM](/shared/glossary/#lstm) "forget/input gates" (deciding what to keep vs. drop from memory) and modern [MLP](/shared/glossary/#mlp) blocks like [SwiGLU](/shared/glossary/#swiglu), where one half of the layer gates the other. It is closely related to [scale-and-shift](/shared/glossary/#scale-and-shift) conditioning, except a pure gate only *scales* (multiplies) rather than also adding an offset.

### GCG {#gcg}
Short for **Greedy Coordinate Gradient** — a gradient-based attack that finds an adversarial suffix (a short string of seemingly random tokens) which, when appended to a harmful question, causes an aligned [LLM](/shared/glossary/#llm) to comply anyway. It works by swapping one suffix token at a time for whatever the gradient says raises the probability of an unsafe answer most. Like picking a combination lock by feeling each dial until the click; once one model is unlocked the same suffix often opens other models, which is why GCG is the standard benchmark attack in [jailbreak](/shared/glossary/#jailbreak) research.

### GELU {#gelu}
Gaussian Error Linear Unit — a smooth activation function widely used in transformer [MLPs](/shared/glossary/#mlp).

### GEMM {#gemm}
GEneral Matrix Multiply — the workhorse operation `C = A × B` on two matrices, and the single most common heavy computation inside a neural network. GPUs are built to do GEMMs fast; nearly every layer's forward pass is one. When one input is very "skinny" (a tiny batch, as in single-token [decode](/shared/glossary/#decode)) the GPU's [Tensor Cores](/shared/glossary/#tensor-core) sit half-idle, so that case needs a different kernel from a big, square prefill GEMM.

### Generalization {#generalization}
How well a model performs on inputs it has *never seen*, as opposed to merely repeating its training examples. A model that generalizes has captured the underlying *pattern*; one that has only memorized has captured the *examples* — the difference between a student who learned how multiplication works and one who memorized a single times-table and is lost on any new numbers. For a style [LoRA](/shared/glossary/#lora), generalization means the learned look transfers to prompts that never appeared in training; its opposite is [overfitting](/shared/glossary/#overfitting). You measure it by checking performance on held-out inputs, not on the training set.

### Generalized policy iteration {#generalized-policy-iteration}
The unifying idea that almost every RL algorithm is some interleaving of two processes: *policy evaluation* (make the [value function](/shared/glossary/#value-function) more accurate for the current [policy](/shared/glossary/#policy)) and *policy improvement* (make the policy [greedier](/shared/glossary/#greedy-policy) with respect to the current values). [Policy iteration](/shared/glossary/#policy-iteration) runs evaluation to completion before each improvement; [value iteration](/shared/glossary/#value-iteration) does a single evaluation step per improvement; but the two can be blended in *any* proportion and still converge to the [optimal policy](/shared/glossary/#optimal-policy). Think of tuning a piano: policy iteration is like perfectly tuning one string before moving to the next, while value iteration is like giving every string a tiny twist and repeating the process until the whole chord sounds right. Either way, you eventually reach a perfectly tuned instrument. The processes both compete (improving the policy makes the old values stale) and cooperate (each hands the other something better to work with), and they come to rest only when neither can change anything — which is exactly the optimal solution.

### Generator {#generator}
The half of a [GAN](/shared/glossary/#gans) that actually makes images: it takes a vector of random noise and maps it to a picture, learning to fool the [discriminator](/shared/glossary/#discriminator) into judging its output as real. It never sees the real images directly — it learns only from whether the discriminator was fooled, like a forger who improves purely from a detective's reactions. After training, the generator alone is what you keep and sample from.

### GenEval {#geneval}
A [benchmark](/shared/glossary/#benchmark) that measures how faithfully a [text-to-image](/shared/glossary/#stable-diffusion) model obeys the *structured content* of a prompt — the right number of objects, the right colors, the right spatial arrangement ("a red cube to the left of a blue sphere"). Instead of asking a person, it runs an object detector on each generated image and checks automatically whether every requested object, count, color, and position is present; the score is the fraction of prompts whose requirements were *all* satisfied. Like an exam graded against a fixed answer key rather than on handwriting: "two cats? — yes; one is orange? — no, fail." It targets compositional skills (counting, positioning, attribute binding) that beauty metrics like [FID](/shared/glossary/#fid) completely ignore.

### Genie {#genie}
A line of DeepMind "foundation world models" (Genie, Genie 2) that learn to generate playable, [action-conditioned](/shared/glossary/#action-conditioning) video from large amounts of ordinary internet video — *without* ever being told which action produced each frame. The trick is a [latent action model](/shared/glossary/#latent-action-model): Genie infers a small set of consistent "actions" purely from how consecutive frames change, so afterwards a person can press one of those discovered actions and watch the model roll the world forward. This is what lets it build a [world model](/shared/glossary/#world-model) you can actually control out of unlabeled footage alone, instead of needing recorded controller inputs.

### GGUF {#gguf}
A single-file format for storing a [quantized](/shared/glossary/#quantization) model — weights plus all the metadata needed to run it — popularized by [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Like a self-contained zip that a laptop or phone can open and run without extra setup, it is the format of choice for [edge and on-device inference](/shared/glossary/#edge-inference).

### Glow {#glow}
A well-known [normalizing flow](/shared/glossary/#normalizing-flow) model (from OpenAI, 2018) that improved on [Real NVP](/shared/glossary/#real-nvp) by adding [learnable](/shared/glossary/#learnable) 1×1 [convolutions](/shared/glossary/#convolution-layers) that shuffle and mix the channels between steps, letting it generate sharp, high-resolution faces. It showed that flows could produce convincing images and smoothly morph one face into another, though they were later overtaken by [diffusion models](/shared/glossary/#diffusion-model) on hard, real-world images.

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

### Gradient penalty {#gradient-penalty}
An extra [loss](/shared/glossary/#loss-function) term used in [Wasserstein GANs](/shared/glossary/#wasserstein-gan-wgan) (WGAN-GP) that keeps the critic 1-[Lipschitz](/shared/glossary/#lipschitz-constraint) — meaning its output cannot change faster than its input. It works by measuring the size of the critic's [gradient](/shared/glossary/#gradients) with respect to its input image and pushing that size toward 1. This replaces the original WGAN's blunt trick of clipping weights to a fixed range, which often hurt quality, and is the main reason WGAN-GP trains so stably.

### GradScaler {#gradscaler}
A helper used with [float16](/shared/glossary/#float16) mixed-precision training that multiplies the loss before the backward pass, preventing small gradients from rounding to zero ([underflow](/shared/glossary/#underflow)).

### Graph break {#graph-break}
A point where [`torch.compile`](/shared/glossary/#torchcompile) cannot trace the code (e.g. a `print` or a data-dependent branch), forcing it to split the model and fall back to [eager mode](/shared/glossary/#eager-mode) — a common cause of lost speedup.

### Greedy decoding {#greedy-decoding}
The simplest [sampling](/shared/glossary/#sampling) rule: at every step, pick the single most likely next token (the [`argmax`](/shared/glossary/#argmax) of the [logits](/shared/glossary/#logits)) and never roll the dice. Like always ordering the most popular dish on the menu — boring but predictable. Useful when reproducibility matters, though on a GPU even greedy decoding is not bit-for-bit deterministic across batch sizes because floating-point sums reorder.

### Greedy policy {#greedy-policy}
The policy that, in every state, picks the action with the highest current [value](/shared/glossary/#value-function) estimate — no randomness, no looking past the numbers in front of it ("greedy" because it grabs the locally best-looking action). Acting greedily with respect to the *optimal* value function yields the [optimal policy](/shared/glossary/#optimal-policy), which is why so many algorithms loop between improving value estimates and reading off the greedy policy. On its own a greedy policy never explores, so during learning it is usually softened into an [ε-greedy](/shared/glossary/#epsilon-greedy) policy that occasionally acts at random.

### GridWorld {#gridworld}
A deliberately tiny reinforcement-learning environment: a 2D grid of cells in which an agent steps up/down/left/right, with a few walls, a goal cell, and maybe a hazard. Because the entire state is just "which cell am I in," it is trivial to simulate and to draw, which makes it the standard first sandbox for testing an idea — a [world model](/shared/glossary/#world-model), a [policy](/shared/glossary/#policy), a planning algorithm — before scaling up to something as rich as [Atari](/shared/glossary/#atari) or a 3D game.

### Grounding {#grounding}
Making a [VLM](/shared/glossary/#vlm) point at *where* something is in an image, not just say *that* it is there — the model outputs spatial references like a *bounding box* (a rectangle `(x1, y1, x2, y2)` around an object) or a single point, instead of only words. The common trick is to add [special tokens](/shared/glossary/#special-tokens) (e.g. `<box>`) plus tokens for quantized coordinates to the [vocabulary](/shared/glossary/#vocabulary), so a location becomes a few extra tokens the model emits with ordinary [next-token prediction](/shared/glossary/#next-token-prediction) — no new architecture needed. In modern [native multimodal](/shared/glossary/#native-multimodal) models, this alignment is leveraged during the [decode](/shared/glossary/#decode) phase by attending heavily to visual features stored in the [KV cache](/shared/glossary/#kv-cache) during [prefill](/shared/glossary/#prefill). Analogy: the difference between a tour guide who says "there's a fountain in this plaza" and one who actually points their finger at it. Example: asked "where is the dog?", a grounded model answers "`<box>` 0.10 0.20 0.45 0.80", which a viewer can draw as a rectangle on the photo; this is what benchmarks like RefCOCO measure.

### GRPO {#grpo}
Group Relative Policy Optimization — [value-function](/shared/glossary/#value-function)-free PPO variant; DeepSeek lineage

### GSM8K {#gsm8k}
A benchmark of about 8,000 grade-school math word problems, widely used to test step-by-step reasoning because each problem has a single checkable numeric answer.

### GTSAM {#gtsam}
Factor-graph SLAM library; the standard back-end for many modern systems

### Gymnasium {#gymnasium}
The standard Python library of reinforcement-learning environments — the maintained successor to OpenAI Gym. It defines the common API every environment follows: `reset()` to start an episode and `step(action)` to take an action and get back the next observation, reward, and whether the episode ended. Because classic testbeds like [FrozenLake](/shared/glossary/#frozenlake), [Blackjack](/shared/glossary/#blackjack), [Cliff Walking](/shared/glossary/#cliff-walking), CartPole, and the MuJoCo control tasks all share this interface, the same training loop can be pointed at any of them with almost no changes.

### H.264 {#h264}
The most common [video codec](/shared/glossary/#video-codec) on the internet, also called AVC (Advanced Video Coding) — the rules used to compress almost every `.mp4` you have ever streamed. It compresses well and decodes fast on nearly all hardware, which is why it is the safe default, though newer codecs like [AV1](/shared/glossary/#av1) shrink files further. Analogy: it is the JPEG of video — not the smallest or newest, but supported everywhere. Example: a 5-second 720p clip that is 333 MB as raw frames might be only a few megabytes as an H.264 `.mp4`.

### H2O {#h2o}
Short for *Heavy-Hitter Oracle*, a [KV cache](/shared/glossary/#kv-cache) eviction method that keeps only the handful of past tokens that have been getting most of the [attention](/shared/glossary/#attention) — the "heavy hitters" — and throws the rest away. Like skimming a long book and keeping only the few sentences you keep flipping back to: you save shelf space while barely losing the plot, which lets a model serve much longer sequences in the same memory. It always keeps the very first tokens too (the [attention sink](/shared/glossary/#attention-sink)), since those anchor the model no matter what they say.

### Half-rotation {#half-rotation}
An efficient way to apply [RoPE](/shared/glossary/#rope): rather than rotating each adjacent pair of vector components on its own, you split the vector into two halves and combine them in one shot (the `rotate_half` trick, `[x₁, x₂] → [−x₂, x₁]`). It turns many tiny 2-D rotations into a couple of whole-vector operations, so it runs fast on a GPU while giving the same result.

### Hallo {#hallo}
A specific [talking-head](/shared/glossary/#talking-head) model designed to generate high-quality portrait animations from a single image and audio. Like [EMO](/shared/glossary/#emo), it focuses on making the facial movements, including lip sync and expressions, look naturally tied to the audio without the stiffness of older, landmark-based methods.

### Hallucination {#hallucination}
When an [LLM](/shared/glossary/#llm) states something false with the same confident tone it uses for true things — invented citations, made-up people, fabricated facts. Like a student who didn't read the book but answers the essay question anyway in confident prose; the grammar is fine, the facts are not. Hallucination is built in to the [next-token prediction](/shared/glossary/#next-token-prediction) objective, which rewards fluent continuation rather than truth, and is mitigated (not solved) by [RAG](/shared/glossary/#rag), [verifiers](/shared/glossary/#verifier), and abstention training.

### Hard negatives {#hard-negatives}
In [contrastive](/shared/glossary/#infonce) training, the *wrong* candidates the model finds hard to reject because they look almost right — as opposed to *easy* negatives so obviously wrong they teach it nothing. For a photo of a husky, the caption "a wolf in deep snow" is a hard negative (close, but wrong), while "a slice of pizza" is an easy one. Training learns fastest from hard negatives because they sit right on the boundary the model is still getting wrong, so each one delivers a large, informative [gradient](/shared/glossary/#gradients); *mining* them means actively searching the data for these near-misses (e.g. the highest-[cosine-similarity](/shared/glossary/#cosine-similarity) mismatch) instead of hoping a random [batch](/shared/glossary/#batch) happens to contain some. Like a chess student who improves quickest by drilling against opponents just above their level, not by beating beginners over and over. Example: to mine hard negatives for a caption, retrieve the images it scores highly against but does *not* actually describe, and add those as negatives for the next training step.

### HBM {#hbm}
High-Bandwidth Memory — stacked DRAM on a modern GPU; usually the bandwidth bottleneck

### Headroom {#headroom}
The safety margin you have left before something breaks. In low-precision training it is the spare range of values a number format can still represent before it overflows or rounds down to zero and triggers [numerical issues](/shared/glossary/#numerical-issues) — like the gap between your head and the ceiling: the less you have, the easier it is to bump into trouble. [FP8](/shared/glossary/#fp8) packs numbers into far fewer bits than [bfloat16](/shared/glossary/#bfloat16), so it has much less headroom and is more likely to destabilize a run.

### Heads (attention) {#heads}
The independent, parallel [attention](/shared/glossary/#attention) sub-computations in multi-head attention. Each head operates on its own learned projections of queries, keys, and values, so different heads can latch onto different relationships — one might track which word is the grammatical subject while another tracks what rhymes — and the model attends to several representation subspaces at once. They are called *heads* by analogy to the read/write "heads" of a tape or disk drive: several separate readers scanning the same strip of data in parallel, each pulling out something different. "Multi-head" attention simply runs many such readers side by side and then joins their findings.

### Hessian {#hessian}
The matrix of all second partial derivatives of a function — it captures the *curvature* of a loss landscape, not just its slope. Where the [gradient](/shared/glossary/#gradients) tells you "which way is downhill," the Hessian tells you "and how sharply does it bend." Like the difference between knowing a road slopes down and knowing whether it banks into a tight curve or stretches out almost flat. For real LLMs the full Hessian is too big to store (rows × columns each equal to the parameter count), so methods like [GPTQ](/shared/glossary/#gptq) use cheap approximations of it — typically built from a small [batch](/shared/glossary/#batch) of [calibration](/shared/glossary/#calibration) activations — to decide which weights matter most when [quantizing](/shared/glossary/#quantization).

### Heun's method {#heuns-method}
A second-order ODE solver that improves on the [Euler method](/shared/glossary/#euler-method) with a predict-then-correct step: it takes a tentative Euler step, measures the slope at that new point too, then moves using the *average* of the start and end slopes. Averaging the two slopes cancels much of the error Euler makes, so Heun reaches the same accuracy in far fewer steps — which is why it is the default sampler in [EDM](/shared/glossary/#edm). Like checking the road both where you are and where you're about to be, then steering down the middle. Named after the German mathematician Karl Heun.

### Hierarchical generation {#hierarchical-generation}
Building a long or complex video in *stages from coarse to fine* rather than all at once: first decide the big structure — a [shot list](/shared/glossary/#shot-list) or a handful of [keyframes](/shared/glossary/#keyframe) spread across the timeline — then fill in the detailed frames between them. Working top-down keeps a long video coherent, because the overall plan is fixed before any single moment is rendered, the same way a director storyboards a film before shooting it. Contrast it with generating a clip straight through frame by frame, where the story has no plan to hold it together and tends to wander. It is one of the main strategies (alongside [sliding-window generation](/shared/glossary/#sliding-window-generation) and [streaming](/shared/glossary/#streaming-video-generation)) for getting past the few-second limit of a single model.

### Hierarchical VAE {#hierarchical-vae}
A [VAE](/shared/glossary/#vae) with several layers of latent variables stacked at different scales instead of just one. Higher levels capture the big picture (overall layout and shape) while lower levels fill in fine detail (texture and edges), much like an artist who first sketches rough shapes and then adds the small touches. Splitting the work across levels lets the model represent complex images far better than a single flat [latent space](/shared/glossary/#latent-space) can. [NVAE](/shared/glossary/#nvae) and [Very Deep VAE](/shared/glossary/#very-deep-vae) are well-known examples.

### Higher-order sampler {#higher-order-sampler}
In diffusion sampling, the solver takes a series of discrete steps along an [ODE](/shared/glossary/#ode) path from pure noise toward a finished image. A *higher-order* sampler estimates the shape of that path more accurately at each step by using extra slope measurements, instead of assuming the path is locally a straight line. A first-order method ([Euler](/shared/glossary/#euler-method)) just follows the slope where it currently stands; a second-order method like [Heun's method](/shared/glossary/#heuns-method) or [DPM-Solver++](/shared/glossary/#dpm-solver) also peeks ahead and averages the two slopes, cancelling most of the error. Because each step is more accurate, higher-order samplers reach good image quality in far fewer steps — often 15–25 instead of 50+. Picture driving toward a bend: a first-order driver steers only by where the road points right now and drifts wide, while a higher-order driver also notices how the road is curving ahead and corrects, staying on track with fewer adjustments.

### Holonomic {#holonomic}
A vehicle whose instantaneous motion can be any direction (mecanum, omni)

### Hopper {#hopper}
NVIDIA's 2022 GPU architecture (H100, H200) and the workhorse of LLM training and serving in 2023–2024. It was the first generation to ship dedicated [FP8](/shared/glossary/#fp8) [Tensor Cores](/shared/glossary/#tensor-core), which is what made FP8 inference a practical option. Named after Grace Hopper, the computer scientist who invented the compiler.

### Hue {#hue}
The "color name" part of a color — red, orange, green, blue, and so on — separate from how light or dark it is and how vivid it is. It is one axis of the way computers describe color (the *H* in the HSV color model), and it wraps around in a circle, so the two ends meet and both 0° and 360° are red. Analogy: hue is the label on a paint tube ("blue"), brightness is how much white or black you stirred in, and saturation is how strong the color is. In an [optical flow](/shared/glossary/#optical-flow) picture, hue is often used to show the *direction* each pixel moved — each compass direction gets its own color — while brightness shows how *fast* it moved.

### Hybrid retrieval {#hybrid-retrieval}
Retrieving with both dense [embedding](/shared/glossary/#embedding) search (matches meaning) and sparse keyword search ([BM25](/shared/glossary/#bm25)) (matches exact words) and merging the two result lists, so each method covers the other's weaknesses.

### I2V {#i2v}
Image-to-Video: the task of generating a short video clip *starting from a single still image*, where the model invents plausible motion while keeping the first frame's appearance fixed. It is easier than [text-to-video (T2V)](/shared/glossary/#t2v) because the image already settles *what the scene looks like*, leaving the model to handle only *how it moves* — and its training data is essentially free, since any video clip can be split into "first frame = input, the rest = target" with no text caption needed. [Stable Video Diffusion](/shared/glossary/#stable-video-diffusion-svd) is the canonical open I2V model.

### Identity function {#identity-function}
A function that returns its input unchanged: `f(x) = x`. In the context of straight-through estimators, gradients are passed through a non-differentiable operation as if it were the identity function.

### Ideogram {#ideogram}
A [text-to-image](/shared/glossary/#text-to-image) model (and product) built by a startup of the same name, especially praised for [text rendering](/shared/glossary/#text-rendering) — drawing legible, correctly-spelled words, logos, and typography inside images, which makes it a favorite for posters and graphic design. Like a sign painter you can trust to spell the shop name right, not just paint pretty letters. It competes with [DALL·E 3](/shared/glossary/#dalle-3), [Imagen 3](/shared/glossary/#imagen-3), and [Flux](/shared/glossary/#flux).

### Image embedding {#image-embedding}
The single dense vector a vision model boils a whole picture down to — a fixed-length list of numbers (say 512 of them) that captures *what is in the image* rather than its raw pixels. [CLIP](/shared/glossary/#clip)'s image encoder, for instance, reads the pixels and outputs one such vector, placing pictures with similar content near each other in the shared [embedding](/shared/glossary/#embedding) space. Analogy: distilling a whole meal down to a single flavor profile you can quickly compare against other dishes — you lose the individual ingredients but keep the essence needed to say "these two are alike." Because an image and a caption can then be compared just by the [cosine similarity](/shared/glossary/#cosine-similarity) of their vectors, image embeddings are what make [zero-shot](/shared/glossary/#zero-shot) classification and [cross-modal retrieval](/shared/glossary/#cross-modal-retrieval) work. Example: in CLIP the photo of a dog and the sentence "a photo of a dog" each become one vector, and the two land close together.

### Imagen 3 {#imagen-3}
Google's [text-to-image](/shared/glossary/#text-to-image) model, known for photorealistic detail and unusually good [text rendering](/shared/glossary/#text-rendering) — it can spell words inside the picture correctly, long a weak spot for generators. It leans on a strong [text encoder](/shared/glossary/#text-encoder) and carefully curated training data to follow prompts faithfully. Like a meticulous illustrator who not only paints the scene you describe but gets the lettering on the signs right. It is Google's competitor to [DALL·E 3](/shared/glossary/#dalle-3) and [Stable Diffusion](/shared/glossary/#stable-diffusion).

### ImageNet {#imagenet}
A large benchmark dataset of about 1.2 million photos hand-labeled into 1,000 everyday categories (breeds of dog, kinds of mushroom, vehicles, and so on). For over a decade it has been the standard yardstick for "how well does this model see," so a new image encoder is almost always reported by its ImageNet accuracy. Think of it as the standardized entrance exam of computer vision — not perfect, but common enough that everyone's scores can be compared on one scale. A larger, even more finely labeled version is called ImageNet-21k (≈21,000 categories); see also its much smaller cousin [CIFAR-10](/shared/glossary/#cifar-10).

### img2img {#img2img}
Generating a new image that is guided by an existing input image instead of starting from pure noise. You partially noise the input — controlled by a *denoising strength* (0 = keep the original, 1 = ignore it) — then let the [diffusion model](/shared/glossary/#diffusion-model) denoise from there, so the result keeps the rough layout and colors of the input while following the new prompt. Like tracing over a rough sketch: the more you erase first, the more freedom the model has to redraw.

### Impedance control {#impedance-control}
Command a virtual spring-damper between end-effector and reference

### IMU {#imu}
Inertial Measurement Unit — gyroscope + accelerometer (often + magnetometer)

### Inception network {#inception-network}
A famous image-classification [convolutional neural network](/shared/glossary/#cnn) (the "Inception" / GoogLeNet family) trained on millions of labeled photos. Along the way it learns to boil any image down to a compact *feature vector* — a list of numbers that captures *what is in the picture* (fur, wheels, sky) rather than the raw pixels. Because those features are such good summaries of image content, quality metrics like [FID](/shared/glossary/#fid) reuse a frozen, pretrained Inception network as a fixed yardstick instead of training anything new — like always using the same trusted scale to weigh two bags so the comparison is fair. (It was nicknamed "Inception" after the movie, for its "[network inside a network](/shared/glossary/#network-in-network)" design.)

### Indexing {#indexing}
Mapping a multidimensional index `[i, j, …]` to a flat storage position via `offset + Σ iₖ·strideₖ`

### Inference-time compute {#inference-time-compute}
The work a model does while answering a question (not while training) — for reasoning models, mostly the tokens it spends "thinking" before it replies. Giving a fixed model more inference-time compute, like giving a student more time on an exam, can raise its accuracy without changing the model at all.

### InfiniBand (IB) {#infiniband-ib}
High-speed network with RDMA; standard for AI clusters

### InfoNCE {#infonce}
The *contrastive* [loss](/shared/glossary/#loss-function) that [CLIP](/shared/glossary/#clip) and most dual encoders train with: for each item it pulls the one correct match closer and pushes every other candidate away. **How it is computed.** Take a batch of N image–caption pairs, [L2-normalize](/shared/glossary/#l2-normalization) every vector, and build the N×N grid of [cosine-similarity](/shared/glossary/#cosine-similarity) scores (one [matmul](/shared/glossary/#matmul)). Each row is one image scored against all N captions, and the *correct* caption sits on the diagonal. Apply [softmax](/shared/glossary/#softmax) across the row and ask that the diagonal entry get nearly all the probability — which is exactly [cross-entropy](/shared/glossary/#cross-entropy) with "the right answer is position i." Do this across rows and again across columns and average the two. The name is short for *Noise-Contrastive Estimation* of mutual *Info*rmation: the off-diagonal pairs are the "noise" the true pair must be told apart from. Analogy: a police lineup where the model must point to the one caption that truly goes with this photo while N−1 decoys stand beside it, scored on how confidently it picks the right one.

### Inpainting {#inpainting}
Filling in a masked-out region of an image so the patch blends seamlessly with the rest. You hand the model the surrounding pixels as fixed context and let it generate only the hole — like a restorer repainting a torn corner of a photo to match the surviving picture. With a [diffusion model](/shared/glossary/#diffusion-model) this is done by re-noising and denoising only inside the mask while pasting the known pixels back on every step.

### Instruction tuning {#instruction-tuning}
A second training stage that turns a model which merely *continues* text (or, for a [VLM](/shared/glossary/#vlm), describes an image) into one that *follows requests* — by [fine-tuning](/shared/glossary/#fine-tuning) it on many (instruction, response) examples instead of raw documents. For a VLM the examples are conversational (image, question, answer) triples, like the LLaVA-Instruct set whose dialogues a strong language model wrote from image annotations. Analogy: a fluent speaker who can ramble on any topic versus a helpful assistant who answers the exact question you asked — same vocabulary, very different behavior, and the gap is closed purely by showing thousands of question-and-answer demonstrations. Example: before tuning, shown a photo and "What is the dog doing?", the model might just caption "a dog on grass"; after tuning it answers "It is catching a frisbee." The key lesson is that this capability comes from *data, not architecture* — the network is unchanged; only what it trains on differs.

### InstructPix2Pix {#instructpix2pix}
An image-editing model that takes a photo and a plain-English instruction ("make it winter," "add sunglasses") and returns the edited photo in a single pass — no masks, no per-image optimization. Its real trick is the *training data*: since no one wants to hand-edit thousands of photos, the data is made synthetically — a [large language model](/shared/glossary/#llm) writes an instruction plus before/after captions, and a text-to-image model ([Stable Diffusion](/shared/glossary/#stable-diffusion)) with [Prompt-to-Prompt](/shared/glossary/#prompt-to-prompt) renders a matched image pair that differs *only* in the described change. The finished model is then [fine-tuned](/shared/glossary/#fine-tuning) on millions of these triples. Like teaching an editor by showing them countless "before, instruction, after" flashcards until they can follow any new instruction.

### int8 {#int8}
8-bit integer format; storing [weights](/shared/glossary/#weights) or [activations](/shared/glossary/#activations) as int8 uses a quarter of the memory of [float32](/shared/glossary/#float32) and can run faster, at some cost in precision.

### Inter-rater agreement {#inter-rater-agreement}
A measure of how often two or more graders give the *same* scores to the same items — the check you run before trusting one grader to stand in for another. If a cheap [LLM-as-judge](/shared/glossary/#llm-as-judge) and a human reviewer rate the same 100 answers and their scores line up, the automatic judge can replace expensive human review; if they disagree a lot, it cannot. It is computed with a statistic such as a *correlation* (how well two lists of numbers rise and fall together, on a −1-to-+1 scale) or *Cohen's kappa* (the fraction of agreement beyond what random guessing alone would produce, on a roughly 0-to-1 scale, named after the psychologist Jacob Cohen). Analogy: two teachers marking the same stack of essays — if their grades nearly match you can trust either one alone next time, but if they wildly differ then the rubric (or one of the teachers) is unreliable.

### IP-Adapter {#ip-adapter}
A lightweight add-on that lets a [diffusion model](/shared/glossary/#diffusion-model) take an *image* as a prompt alongside (or instead of) text — you hand it a reference picture and it copies that subject's appearance or style into what it generates. It works by encoding the reference image and feeding it through a small set of *extra* [cross-attention](/shared/glossary/#cross-attention) layers added next to the existing text ones (the "IP" stands for *image prompt*), so the base model's own [weights](/shared/glossary/#weights) stay frozen and untouched. Because it needs no per-subject training and accepts any new reference on the fly, it is a popular way to hold a character or style steady across shots (see [character consistency](/shared/glossary/#character-consistency)). Like handing an artist a photo and saying "draw new scenes, but keep this person looking exactly like this."

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

### Joint image-video training {#joint-image-video-training}
A training recipe that feeds a video model a mix of still images and video clips in the same run — treating each still image as a one-frame "video" — so the model keeps its sharp single-image skills while it learns motion. The problem it solves: training on video *alone* lets a model's still-image quality decay, because video datasets are smaller and more compressed than image datasets, so the rich appearance knowledge an inflated image model started with drifts away. Mixing in a large fraction of images (often the majority of each batch) keeps that knowledge fresh and makes the model far more data-efficient. It needs no architectural change because a still image is simply the `T=1` special case of a video — the same layers process both.

### Jailbreak {#jailbreak}
A prompt — sometimes plain English, sometimes a gradient-found suffix like in [GCG](/shared/glossary/#gcg), sometimes a long role-play setup or a translation into a low-resource language — that gets a safety-trained model to do what its alignment training was supposed to refuse. Like picking a hotel-room door lock instead of asking for the key. Modern defenses assume any single safety layer can be jailbroken and use *defense in depth* — input filtering, output filtering, monitoring, refusal classifiers — instead of trusting the model alone.

### Kernel {#kernel}
A single function that runs on the GPU (or CPU) to carry out one operation, such as a matrix multiply or an element-wise add.

### Kernel fusion {#kernel-fusion}
Combining several small operations into one [kernel](/shared/glossary/#kernel) so the hardware reads and writes memory fewer times and pays fewer launch costs.

### Keyframe {#keyframe}
A frame chosen to anchor a specific moment in a video — the picture you fix in place first and then build motion around. The term comes from hand-drawn animation, where a lead artist draws the important "key" poses and assistants fill in the frames between them. In long-form generation you create a few keyframes spread seconds apart to lock down how the scene looks at those points, then use an [image-to-video](/shared/glossary/#i2v) or [frame-interpolation](/shared/glossary/#frame-interpolation) model to invent the frames in between — a form of [hierarchical generation](/shared/glossary/#hierarchical-generation). Choosing keyframes well matters: two that are too different cannot be smoothly bridged.

### KF {#kf}
Kalman Filter — optimal linear-Gaussian Bayes filter

### KL divergence {#kl-divergence}
Short for **Kullback-Leibler divergence** — a number that measures how far one probability distribution has drifted from another, growing larger the more the two disagree. In [RLHF](/shared/glossary/#rlhf) it acts as a leash on the [policy](/shared/glossary/#policy) being trained: the further its word probabilities wander from the frozen [reference model](/shared/glossary/#reference-model), the bigger the penalty it pays. Like a tether that lets a climber explore but stops them straying somewhere dangerous, it lets the model chase reward without forgetting how to talk sensibly.

### KV cache {#kv-cache}
A [scratchpad](/shared/glossary/#scratchpad) that stores the [attention](/shared/glossary/#attention) keys and values already computed for every earlier token in the sequence, so generating the next token only has to compute keys and values for that *one* new token instead of redoing all the previous ones. Like writing out a long multiplication table once and then looking up products instead of recalculating them — it turns each decode step from "redo the whole prompt" into "do one more token," which is what makes long-context serving fast enough to be usable.

### L2 normalization {#l2-normalization}
Rescaling a vector so its length becomes exactly 1 while keeping the direction it points unchanged — done by dividing every element by the vector's own length. It is called *L2* because the length it uses is the **L2 norm** (also called the *Euclidean norm* — the ordinary straight-line distance you would measure with a ruler). The "2" comes from the *p* in the general *Lp* norm formula, which takes the *p*-th root of the sum of each element's *p*-th power; set *p* = 2 and that becomes the square root of the sum of squares — exactly the Pythagorean length √(x₁² + x₂² + …). **Worked example:** `[3, 4]` has length √(3² + 4²) = √25 = 5, so its L2-normalized form is `[3/5, 4/5] = [0.6, 0.8]` — same direction, but now length 1 and sitting on the unit sphere. Analogy: shrinking every arrow on a map to the same one-inch length so you can compare *which way* they point without the longer arrows drowning out the shorter ones. This is the step that turns a plain [dot product](/shared/glossary/#dot-product) into [cosine similarity](/shared/glossary/#cosine-similarity), which is why [CLIP](/shared/glossary/#clip) L2-normalizes every image and text [embedding](/shared/glossary/#embedding) before scoring matches — so only direction (meaning), not magnitude, decides the score.

### L2 regularization {#l2-regularization}
A regularization technique that adds a penalty proportional to the squared magnitude of model weights to the loss function, encouraging smaller weights and reducing overfitting. In standard adaptive optimizers such as [Adam](/shared/glossary/#adam), this penalty is folded into the gradient and scaled by the adaptive learning rate, which is why [AdamW](/shared/glossary/#adamw) uses [decoupled](/shared/glossary/#decoupled) weight decay instead.

### LAION {#laion}
A family of huge, openly released image-text datasets (LAION-400M, LAION-5B — the number counts the image-caption pairs) scraped from the public web by the non-profit **LAION** (short for *Large-scale Artificial Intelligence Open Network*). Each entry is just an image URL plus its [alt-text](/shared/glossary/#alt-text) caption, kept only if [CLIP](/shared/glossary/#clip) judged image and caption to roughly match. It is the public fuel that trained [Stable Diffusion](/shared/glossary/#stable-diffusion) and many other open models. Like a giant secondhand library assembled by photographing every captioned picture on the open internet — enormous and free, but riddled with mislabeled, duplicated, and low-quality entries, which is why every serious user re-filters and [deduplicates](/shared/glossary/#deduplication) it before training. Example: "LAION-2B-en" is the roughly 2-billion-pair English-caption subset.

### Langevin dynamics {#langevin-dynamics}
A way to draw samples from a distribution when you only know its [score](/shared/glossary/#score) — the [gradient](/shared/glossary/#gradients) of its log-density. You start from a random point and repeatedly take a small step in the score direction (uphill toward higher probability) while also adding a little random noise each step so you explore rather than collapse onto a single peak. The uphill pull plus the random shake settles the point into high-probability regions in the right proportions, like a ball jiggling around a bumpy bowl and spending most of its time in the deepest dips. It is the sampling method behind the original [score](/shared/glossary/#score)-based generative models. Named after the physicist Paul Langevin.

### Latency {#latency}
The time it takes to complete a single request, from input to output; distinct from [throughput](/shared/glossary/#throughput), which counts how many requests finish per second.

### Latent action model {#latent-action-model}
A model that *infers* the action taken between two consecutive video frames when no action was ever recorded, by learning a small [latent](/shared/glossary/#latent-space) code that best explains how the first frame turned into the second. Train it on mountains of unlabeled video and it discovers, on its own, a compact and reusable vocabulary of "moves" — step left, jump, pan the camera — which is precisely what lets a [world model](/shared/glossary/#world-model) like [Genie](/shared/glossary/#genie) become controllable without anyone hand-labeling a single action. Like watching thousands of silent chess games and deducing the set of legal moves purely from how the board changes between snapshots.

### Latent space {#latent-space}
The compressed set of numbers a model uses to represent its data internally, after stripping away the raw detail. Each point in this space stands for one possible output, and nearby points usually mean similar outputs — so you can smoothly "walk" from one to another and watch the result morph. Think of it as the model's private map of its world: instead of a full 28×28-pixel image, an [autoencoder](/shared/glossary/#autoencoder) might describe each digit with just 32 numbers, and that 32-number space is the latent space.

### Latent video {#latent-video}
The compressed form of a video that a [3D VAE](/shared/glossary/#3d-vae) produces: instead of the raw `(T, H, W, C)` pixel [tensor](/shared/glossary/#tensor), you get a much smaller `(T', H', W', C)` grid where time, height, and width have all been shrunk (often ~100× fewer numbers overall). Modern video diffusion runs in this [latent space](/shared/glossary/#latent-space) rather than on pixels, because denoising a 100×-smaller tensor is what makes high-resolution video generation affordable at all.

### LCM {#lcm}
Latent Consistency Model — a [consistency model](/shared/glossary/#consistency-model) distilled in the [latent space](/shared/glossary/#latent-space) of a [VAE](/shared/glossary/#vae), giving 1–4-step [Stable Diffusion](/shared/glossary/#stable-diffusion)-style sampling. It is the most practical few-step recipe for SD-style stacks, which is what makes near-interactive image generation possible.

### LDM {#ldm}
Latent Diffusion Model — a [diffusion model](/shared/glossary/#diffusion-model) that runs in the [latent space](/shared/glossary/#latent-space) of a [VAE](/shared/glossary/#vae) rather than on raw pixels. A VAE first compresses the image (or video) into a much smaller grid of numbers; the diffusion model learns to denoise *that* small grid, and the VAE decoder turns the finished latent back into pixels. Because the latent is often ~50–100× smaller than the image, every training and sampling step is dramatically cheaper, which is the whole reason high-resolution generation became affordable. [Stable Diffusion](/shared/glossary/#stable-diffusion) is the canonical image LDM; modern video models apply the same idea on top of a [3D VAE](/shared/glossary/#3d-vae).

### LFQ {#lfq}
Lookup-Free Quantization — a way to turn a continuous latent into a discrete [token](/shared/glossary/#token-visualaudio) *without* a learned [codebook](/shared/glossary/#codebook). Instead of comparing each latent vector against a trained table of code entries and picking the nearest (the [VQ-VAE](/shared/glossary/#vq-vae) way), LFQ squashes each latent dimension to a sign — roughly, "is this number positive or negative?" — so the pattern of signs across the dimensions *is* the integer code. With no table to look up, there is nothing that can go unused, which sidesteps [codebook collapse](/shared/glossary/#codebook-collapse) and lets the effective vocabulary grow huge cheaply. It is the quantizer behind [MagViT-v2](/shared/glossary/#magvit-v2) and a close cousin of [FSQ](/shared/glossary/#fsq), which snaps each dimension to a small grid of levels rather than just a sign.

### Leaderboard {#leaderboard}
A public ranking that lists models by their score on one or more [benchmarks](/shared/glossary/#benchmark), best at the top — like a sports league table for AI models. It makes progress easy to see at a glance, but a single number hides many hidden choices (prompt wording, answer parsing, image resolution), so two groups can report different scores for the *same* model; a high rank is also suspect if the test questions leaked into training (see [contamination](/shared/glossary/#contamination)). Example: the [MMMU](/shared/glossary/#mmmu) leaderboard ranks [VLMs](/shared/glossary/#vlm) by their accuracy on the MMMU exam, and a new model's headline claim is usually "we moved up this board."

### Learnable {#learnable}
Refers to parts of an AI model (like weights or parameters) that are not set in stone by the programmer, but are instead adjusted automatically during training to improve performance. Like the knobs on a radio that tune themselves until the station comes in perfectly clear, rather than being glued in place.

### Learning rate {#learning-rate}
The step size an [optimizer](/shared/glossary/#optimizer) takes when nudging the [weights](/shared/glossary/#weights) along the [gradient](/shared/glossary/#gradients). Too large and training overshoots and diverges; too small and it crawls — like choosing how big a step to take walking downhill in fog. It is usually ramped up during [warmup](/shared/glossary/#warmup) and then decayed over the run.

### LiDAR {#lidar}
Light Detection And Ranging — laser range scanner

### Linear probe {#linear-probe}
A small linear classifier trained on the frozen hidden [activations](/shared/glossary/#activations) of a layer of a neural network to test whether that layer has *already* encoded some property — for example, "is this sentence true?", "what is the capital of this country?", or "which language is this?" Like sticking a voltmeter into one wire of a circuit to see what signal is flowing past that point; you don't change the circuit, you just read what's already there. The standard first tool in [mechanistic interpretability](/shared/glossary/#mechanistic-interpretability).

### Lipschitz constraint {#lipschitz-constraint}
A limit on how fast a function's output can change as its input changes: a 1-Lipschitz function never changes its output by more than the distance you moved the input. Picture a road whose slope is capped so it can never get steeper than 45° — no cliffs allowed. (The name simply honors the 19th-century German mathematician Rudolf Lipschitz, who first wrote down this "bounded-steepness" condition; it is *not* a description of the rule itself, the way "Celsius" is just a person's name rather than a word about temperature.) [Wasserstein GANs](/shared/glossary/#wasserstein-gan-wgan) require their critic to obey this so the [Earth Mover's Distance](/shared/glossary/#earth-movers-distance) it estimates stays valid, which is what the [gradient penalty](/shared/glossary/#gradient-penalty) enforces.

### LLaVA {#llava}
Large Language and Vision Assistant — an open-source [vision-language model](/shared/glossary/#vlm) that shows how far the simplest possible design can go: take a [frozen](/shared/glossary/#frozen) [CLIP](/shared/glossary/#clip) image encoder, take a frozen [LLM](/shared/glossary/#llm), and connect them with nothing but a lightweight [projector](/shared/glossary/#projector) (a single linear layer or small [MLP](/shared/glossary/#mlp)) that translates each image patch's feature vector into the LLM's word-[embedding](/shared/glossary/#embedding) space. The LLM then "reads" the image as if it were a sequence of extra words. Think of a United Nations translator who listens to a speech in one language and re-phrases each sentence for a listener who only speaks another — the translator (projector) does not change the content, just the format. Despite having no [cross-attention](/shared/glossary/#cross-attention) or [Q-Former](/shared/glossary/#q-former), LLaVA matches or beats far more complex architectures on many visual-question-answering benchmarks, which is why its projector-only design became a widely-copied template. Compare with [Flamingo](/shared/glossary/#flamingo), which uses [gated](/shared/glossary/#gated) cross-attention instead.

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
Low-Rank Adaptation — a cheap way to fine-tune a huge model without rewriting it. Instead of changing the model's billions of frozen [weights](/shared/glossary/#weights), you leave them all untouched and bolt on a tiny *pair* of extra [low-rank](/shared/glossary/#low-rank) matrices that nudge the output. Why a *pair* and not a single matrix? A lone update matrix would have to be the same full size as the weights it is correcting — which defeats the whole point of saving space. The trick is to split that update into two skinny matrices in a row: the first squeezes the big input down to just a handful of numbers, and the second expands those few numbers back out to full size. Picture an hourglass — wide, pinched to a narrow waist, then wide again: it is the narrow waist in the middle (the [low rank](/shared/glossary/#low-rank)) that keeps the total number of stored values tiny, and you need both halves of the hourglass to get from one side to the other. Like leaving a printed textbook exactly as it is and slipping in a few sticky notes that change how you read it: the notes are small to store, quick to write, and you can keep a different set of notes for each task and swap them in and out.

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
A way of approximating a big matrix as the product of two much skinnier ones, capturing most of its information with far fewer numbers. A full 1000×1000 weight matrix holds a million entries, but if its real content is "low rank" you can rebuild it well from, say, two 1000×8 matrices — a few thousand numbers instead of a million. Like mixing any shade of paint from just a few primary colors instead of stocking thousands of separate tubes—you capture the full variety using only a handful of basic components. This is the trick behind [LoRA](/shared/glossary/#lora): freeze the giant base [weights](/shared/glossary/#weights) and learn only a small low-rank update on top.

### Low-resource language {#low-resource-language}
A language for which little digital training data exists — few transcribed recordings, books, or web pages — compared with high-resource languages like English or Mandarin that have billions of words online. Models trained mostly on the abundant languages do worst here, simply because they have not seen enough examples to learn the language's sounds and spellings. Like a cook who has made thousands of Italian dishes but tasted Ethiopian food only once — they will be shaky at Ethiopian cooking until they practice it specifically. Example: [Whisper](/shared/glossary/#whisper) transcribes English almost perfectly but makes far more errors on a language like Welsh or Amharic, which is exactly where a few hours of targeted [fine-tuning](/shared/glossary/#fine-tuning) data helps most.

### LQR {#lqr}
Linear-Quadratic Regulator — optimal linear feedback for quadratic cost

### LSTM {#lstm}
Long Short-Term Memory — a type of recurrent neural network (RNN) cell designed to remember things over long sequences without the information fading away. A plain RNN is like whispering a message down a long line of people — by the end, the message is garbled. An LSTM fixes this with three [gates](/shared/glossary/#gated): a **forget gate** that decides what old information to throw out, an **input gate** that decides what new information to store, and an **output gate** that decides what to actually hand to the next step. Together they maintain a "cell state" — a conveyor belt of memory that can carry important facts across hundreds of time steps with minimal loss. LSTMs were the go-to architecture for sequences (language, speech, time series) before [transformers](/shared/glossary/#transformer) took over, and they remain the classic example of gated memory in neural networks.

### MagViT-v2 {#magvit-v2}
The strongest open recipe for *discrete* video tokenization — turning a clip into a grid of integer [tokens](/shared/glossary/#token-visualaudio) that an [autoregressive](/shared/glossary/#autoregressive-model) or transformer model can generate the same way it generates language. It builds on the [VQ-VAE](/shared/glossary/#vq-vae) idea of a discrete latent but replaces the learned [codebook](/shared/glossary/#codebook) with [LFQ](/shared/glossary/#lfq) (lookup-free quantization), which sidesteps [codebook collapse](/shared/glossary/#codebook-collapse) and scales to a very large vocabulary cheaply. A single MagViT-v2 tokenizer handles both still images and video (it shares the [causal](/shared/glossary/#causal-3d-vae) trick of encoding the first frame on its own), and its reconstructions are sharp enough that token-based generators can finally rival [diffusion models](/shared/glossary/#diffusion-model) on quality — its headline claim is that a good enough tokenizer is what makes language-model-style video generation competitive.

### Manifold {#manifold}
The thin, curved surface inside a much larger space where real data actually lives. A 32×32 color image is a point in a space of 3,072 numbers, but almost every random point in that space looks like static — only a vanishingly small, smoothly connected sliver of it looks like a real photo, and that sliver is the manifold. A useful analogy: a sheet of paper is a 2D surface, but if you crumple it and drop it into a room it traces out a thin curved shape floating in 3D space; the paper is the manifold and the room is the full space. Learning to generate images is largely learning the shape of this surface so you only ever land on it.

### Manipulability {#manipulability}
Scalar measure of how "easy" motion is from a given configuration (e.g. `sqrt(det(JJᵀ))`)

### Mantissa {#mantissa}
The part of a [floating-point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number that holds the *precision digits* — the significant figures sitting in front of the scale factor. In `3.5 × 10¹²`, the `3.5` is the mantissa (also called the *significand*). More mantissa bits give finer resolution between nearby values; fewer mantissa bits leave larger gaps between representable numbers. [FP8](/shared/glossary/#fp8)'s `E4M3` format means 4 [exponent](/shared/glossary/#exponent) bits + 3 mantissa bits, so it can only distinguish about 8 distinct values between each consecutive power of two — coarse, but small enough to fit twice as many numbers in the same memory as [bfloat16](/shared/glossary/#bfloat16).

### Markov property {#markov-property}
The assumption that the current state already contains everything relevant about the past, so what happens next depends only on *where you are now*, not on how you got there. It is the "memoryless" condition that makes an [MDP](/shared/glossary/#mdp) tractable: when it holds, a [policy](/shared/glossary/#policy) can ignore history and look only at the present state. Named after the mathematician Andrey Markov. Example: in chess the current board position is Markov (the move history doesn't change which moves are legal or good), but a single still photo of a moving ball is *not* — you can't tell which way it is heading without the previous frame. When the property fails, you have a [POMDP](/shared/glossary/#pomdp).

### Marlin {#marlin}
A specialized GPU [kernel](/shared/glossary/#kernel) for mixed-precision [matmul](/shared/glossary/#matmul) — 4-bit [weights](/shared/glossary/#weights) multiplied by 16-bit [activations](/shared/glossary/#activations) — built to stay fast even on the skinny, small-batch shapes of [decode](/shared/glossary/#decode). It unpacks the 4-bit weights on the fly while keeping the [Tensor Cores](/shared/glossary/#tensor-core) busy, so a [quantized](/shared/glossary/#quantization) model runs nearly as fast as the math allows. (Named after the fast-swimming marlin fish.)

### MaskGIT {#maskgit}
A way to generate image [tokens](/shared/glossary/#token-visualaudio) in parallel instead of one at a time. Starting from a grid where almost every token is hidden ("masked"), a [transformer](/shared/glossary/#transformer) predicts them all at once, keeps only the predictions it is most confident about, and repeats over a handful of rounds until the grid is full. The analogy is filling in a crossword: lock in the answers you are sure of first, and the rest get easier. This makes it much faster than [raster-order](/shared/glossary/#raster-order) [autoregressive](/shared/glossary/#autoregressive-model) generation, which must fill the grid one token at a time.

### matmul {#matmul}
Matrix multiplication — the dominant compute operation in neural networks; written `A @ B` in PyTorch.

### Matrix inverse {#matrix-inverse}
For a square matrix `A`, its inverse `A⁻¹` is the matrix that undoes it: `A⁻¹A = I`, where `I` is the identity matrix (the matrix equivalent of the number 1, leaving anything it multiplies unchanged). Multiplying by `A⁻¹` is how you *divide* by a matrix, which is exactly what solving a system of linear equations `Ax = b` needs — the solution is `x = A⁻¹b`. In RL it gives the one-shot answer to [policy evaluation](/shared/glossary/#policy-evaluation): the [Bellman equation](/shared/glossary/#bellman-equation) for a fixed [policy](/shared/glossary/#policy) is the linear system `(I − γPπ) V = rπ`, so `V = (I − γPπ)⁻¹ rπ`. Analogy: if a matrix is a recipe that scrambles a list of numbers, its inverse is the recipe that unscrambles them. In practice you rarely form the inverse explicitly — solving the system directly (e.g. [`np.linalg.solve`](/shared/glossary/#nplinalgsolve)) is faster and more numerically stable — but the inverse is the cleanest way to *think* about the answer.

### MDP {#mdp}
Markov Decision Process — the standard mathematical description of a decision-making problem, written as the tuple `(S, A, P, R, γ)`: the set of **S**tates the world can be in, the **A**ctions the agent can take, the [transition probabilities](/shared/glossary/#transition-function) **P** saying where each action is likely to land you, the [**R**eward function](/shared/glossary/#reward-function) scoring what happens, and the [discount factor](/shared/glossary/#discount-factor) **γ** weighing future reward against present. "Decision Process" because the agent makes a sequence of choices over time; "[Markov](/shared/glossary/#markov-property)" because the next state depends only on the current state and action, not on the full history of how you got there. Nearly every RL method assumes the problem is (or can be treated as) an MDP. When the agent cannot fully observe the state, it becomes a [POMDP](/shared/glossary/#pomdp).

### Mechanistic interpretability {#mechanistic-interpretability}
The line of research that tries to reverse-engineer *what individual pieces of a neural network actually do* — which neurons or [attention heads](/shared/glossary/#heads) detect what, where a fact is stored, why a particular output came out. Like opening up a watch to see which gears turn the hands, instead of only timing how fast the watch runs. Main tools: [linear probes](/shared/glossary/#linear-probe), [sparse autoencoders](/shared/glossary/#sae), activation patching, and circuit analysis.

### Media container {#media-container}
The file format that *wraps* compressed video (plus audio, subtitles, and metadata) into one file — `.mp4`, `.mov`, `.webm`, and `.mkv` are containers. The container is the box; the [video codec](/shared/glossary/#video-codec) is how the picture inside was compressed, and the two are independent — the same H.264 video can sit in an `.mp4` or a `.mov`. Analogy: a container is like a shipping box labeled on the outside, while the codec is the packing method used for the fragile thing inside. Example: a `.webm` file is a container that usually holds [AV1](/shared/glossary/#av1)- or [VP9](/shared/glossary/#vp9)-compressed video, whereas `.mp4` most often holds [H.264](/shared/glossary/#h264).

### Medical-image segmentation {#medical-image-segmentation}
The task of labelling *every pixel* in a medical scan (MRI, CT, X-ray, microscopy) as belonging to a particular structure — outlining a tumour, an organ, or a cell boundary — rather than just classifying the whole image with one label. The output is a per-pixel mask, like a precise coloring-book page where each region is filled with its own color. It demands very fine spatial accuracy, since a few pixels can be the difference between the edge of a tumour and healthy tissue, which is exactly why the [U-Net](/shared/glossary/#u-net)'s skip connections — carrying fine detail straight across the network — were originally designed for it. Think of tracing the exact outline of each country on a map instead of just saying "this is a map of Europe."

### Mel bands {#mel-bands}
The output channels of a mel [filterbank](/shared/glossary/#filterbank) — the handful of frequency buckets (commonly 80) that a [mel spectrogram](/shared/glossary/#mel-spectrogram) keeps after squeezing the [STFT](/shared/glossary/#stft)'s hundreds of fine frequency rows onto the perceptual mel scale. Low-pitch bands are narrow and closely spaced while high-pitch bands are wide, mirroring how human hearing tells low notes apart easily but lumps high ones together. Like sorting a piano's 88 keys into a few labeled bins where each bass key gets its own bin but many treble keys share one. Example: an 80-band mel spectrogram describes each moment of sound with 80 numbers instead of 500+ raw frequency values, small enough for a [CNN](/shared/glossary/#cnn) or [transformer](/shared/glossary/#transformer) to process like an image.

### Mel spectrogram {#mel-spectrogram}
A picture of sound: a 2D map with time along one axis and pitch along the other, where brightness shows how much of each pitch is present at each moment. It is built by sliding a short window across the audio waveform and measuring its frequencies (a [Short-Time Fourier Transform](/shared/glossary/#stft)), then squashing the frequency axis onto the *mel scale* — a perceptual spacing that, like human hearing, gives lots of resolution to low pitches and lumps high ones together (the jump from 100 to 200 Hz sounds bigger than the jump from 5,000 to 5,100 Hz). The payoff is that audio becomes an image with, say, 80 frequency rows, so the same [CNN](/shared/glossary/#cnn) or [transformer](/shared/glossary/#transformer) machinery built for vision can process it. Like turning a song into sheet music — a flat diagram you can read at a glance instead of a wiggling waveform.

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
Multi-Modal Diffusion [Transformer](/shared/glossary/#transformer) — the [DiT](/shared/glossary/#dit) variant used in [SD3](/shared/glossary/#sd3) and [Flux](/shared/glossary/#flux) where text tokens and image tokens flow through the *same* [attention](/shared/glossary/#attention) layers ("joint attention") instead of having the image attend to the text through a separate [cross-attention](/shared/glossary/#cross-attention) step. Each [modality](/shared/glossary/#modality) (text vs image) keeps its own normalization and [MLP](/shared/glossary/#mlp) weights, but they see and influence each other inside one shared attention operation, which helps the model get compositional prompts ("a red cube on a blue sphere") right. Like seating writers and illustrators at one table where everyone hears the whole conversation, instead of passing notes between two separate rooms.

### MMBench {#mmbench}
A multiple-choice [benchmark](/shared/glossary/#benchmark) for [VLMs](/shared/glossary/#vlm) that probes many separate abilities — object recognition, spatial relationships, attribute comparison, and more — with each question offering a few labeled answer choices. To stop a model from scoring well by luck or by always favoring one letter, it asks the same question several times with the choices shuffled and counts it correct only if the model picks the right answer *every* time (a trick its authors call *CircularEval*). Like re-asking a quiz question with the options reordered to be sure the student actually knows the answer rather than having memorized "it's always C." It is one of the standard general-capability scores reported for any new VLM.

### MMLU {#mmlu}
Massive Multitask Language Understanding — a 57-subject multiple-choice [benchmark](/shared/glossary/#benchmark) (history, law, medicine, math, and more) that became the standard quick test of how much general knowledge a model has, like a giant trivia exam spanning many school subjects at once.

### MNIST {#mnist}
A classic dataset of 70,000 small 28×28 grayscale images of handwritten digits 0–9. It is the most common "hello world" for image models — tiny, clean, and quick to train on — so a brand-new idea is almost always tried on MNIST first, before anyone risks it on harder, fuller-color data like [CIFAR-10](/shared/glossary/#cifar-10).

### MMMU {#mmmu}
Massive Multi-discipline Multimodal Understanding — a hard [benchmark](/shared/glossary/#benchmark) of college-exam questions across many fields (medicine, engineering, art, business), where each question mixes text with an image such as a diagram, chart, or chemical structure. It is built to require real subject reasoning rather than just reading the picture, which is why even strong [VLMs](/shared/glossary/#vlm) still score far below human experts on it. Like a university final that hands you a figure and expects you to apply the course material to it, not merely describe what you see. It is the most-cited measure of frontier multimodal reasoning, and the harder *MMMU-Pro* variant adds more answer choices and trickier distractors to fight [contamination](/shared/glossary/#contamination).

### MoCoGAN {#mocogan}
MoCoGAN (Motion and Content GAN) is an early video [GAN](/shared/glossary/#gans) whose key idea is to split a video's [latent](/shared/glossary/#latent-space) code into two parts: a single *content* vector that stays fixed for the whole clip (the identity of the person or object) and a sequence of *motion* vectors that change frame to frame (how it moves). Because content is held fixed while motion varies, the same face can be made to perform different expressions, or one motion can be replayed on different faces. This separation — disentangling *what* from *how* — keeps a subject from morphing as it moves; the [generator](/shared/glossary/#generator) reads a new motion vector each frame (produced by a small [recurrent network](/shared/glossary/#recurrent-network)) on top of the one shared content vector. The same content/motion split keeps reappearing inside later diffusion-based systems, which is why the 2017 model is still worth studying.

### Modality {#modality}
One type or format of data — text, images, audio, video, a depth map, and so on. Each modality has its own structure (text is a sequence of tokens, an image is a grid of pixels, audio is a waveform), so a model usually needs a dedicated encoder for each one before their information can be combined. A model that handles more than one is called *multimodal*. Think of modalities as the different human senses — sight, hearing, touch — each carrying information about the same world but in a different form, which the brain then has to fuse into one understanding. [Cross-attention](/shared/glossary/#cross-attention) is one common way to let two modalities exchange information.

### Modality balancing {#modality-balancing}
In a single model trained on several [modalities](/shared/glossary/#modality) at once, the practice of adjusting how much each one contributes to the [loss](/shared/glossary/#loss-function) so that no single modality drowns out the others. The problem arises because modalities are rarely equal in size: if 99% of your [tokens](/shared/glossary/#token-visualaudio) are text and only 1% are image, the text [next-token-prediction](/shared/glossary/#next-token-prediction) loss dominates the gradient and the model barely learns to handle images. It is like a study schedule where, left alone, you would spend every hour on your strongest subject — you have to deliberately reweight so the weaker subjects get their share of attention. Concretely, you either oversample the under-represented modality's data or multiply its loss term by a larger coefficient, tuning until each modality's loss falls at a comparable rate.

### Modality gap {#modality-gap}
The repeated empirical finding that, even in a model like [CLIP](/shared/glossary/#clip) trained to put matching items in one shared space, the [embeddings](/shared/glossary/#embedding) of one [modality](/shared/glossary/#modality) (all the images) sit in a different region from those of another (all the captions) — two separate clusters rather than one blended cloud. The pairs are still correctly *aligned* (a photo is nearer its own caption than to a wrong one), but a constant offset separates the two modalities, a side effect of how [contrastive](/shared/glossary/#infonce) training and the random initial weights shape the geometry. Analogy: two choirs singing the same song in perfect harmony but standing on opposite sides of the stage — in tune with each other, yet never in the same spot. You can *see* it by encoding a batch of images and captions, reducing them with [PCA](/shared/glossary/#pca-principal-component-analysis), and watching the two colors land in separate blobs; it matters because it lowers the [cosine-similarity](/shared/glossary/#cosine-similarity) scores of true pairs and can be partly fixed by shifting one modality's vectors toward the other.

### Mode collapse {#mode-collapse}
A [GAN](/shared/glossary/#gans) failure where the [generator](/shared/glossary/#generator) discovers a few outputs that reliably fool the [discriminator](/shared/glossary/#discriminator) and just keeps making those, ignoring the variety in the real data — like a comedian who finds one joke that always lands and tells only that joke. Each sample may look fine on its own, but the model has stopped covering most of the data. It is the defining instability of GAN training; its discrete-latent cousin is [codebook collapse](/shared/glossary/#codebook-collapse).

### MoE {#moe}
Mixture-of-Experts — instead of one big [MLP](/shared/glossary/#mlp) per layer, the model holds many parallel "[expert](/shared/glossary/#expert)" MLPs and a small router sends each token to only the top few. Like a big company where every question goes to just the two or three relevant specialists rather than the whole staff, the model can hold a huge number of total [parameters](/shared/glossary/#weights) while doing only a fixed, small amount of compute per token. The serving catch: which experts get used shifts with the workload, so keeping them evenly busy across GPUs ([expert parallelism](/shared/glossary/#expert-parallelism-ep)) is the hard part.

### Momentum {#momentum}
A technique that accumulates a moving average of past gradients to dampen oscillations and accelerate gradient descent in consistent directions

### Monosemantic {#monosemantic}
A feature inside a neural network that fires for exactly *one* concept — for example, a direction in [activation](/shared/glossary/#activations) space that lights up only for "Golden Gate Bridge," or only for "negation in a clause." The opposite is *polysemantic*: one neuron that activates for several unrelated concepts at once. Like a single word that means just one thing versus a homonym that means several. Recovering monosemantic features is the main goal of [SAE](/shared/glossary/#sae)-based interpretability.

### Monte Carlo method {#monte-carlo-method}
A way to estimate a [value function](/shared/glossary/#value-function) from complete sampled episodes alone, with no model of the environment (meaning the agent does not know the rules or transition probabilities of the world beforehand and must learn purely by trial and error). To estimate values, the agent plays an episode to the end, computes the actual [return](/shared/glossary/#return) that followed each state, and averages those returns over many runs. Having "no model" is like learning to navigate a maze by actually walking through it, bumping into walls, and finding the exit, rather than studying a complete blueprint of the maze before entering. 

"Monte Carlo" (after the casino) is the general name for estimating a quantity by random sampling instead of exact calculation. Two variants differ only in bookkeeping: *first-visit* MC averages the return from the first time a state appears in each episode (keeping the samples independent), while *every-visit* MC averages from every occurrence. Monte Carlo estimates are [unbiased but high-variance](/shared/glossary/#bias-variance-tradeoff) — they wait for the true outcome but inherit the randomness of a whole trajectory — the opposite trade-off from [temporal-difference learning](/shared/glossary/#temporal-difference-learning).

### Motion module {#motion-module}
The plug-in component at the heart of [AnimateDiff](/shared/glossary/#animatediff): a stack of time-aware ([temporal](/shared/glossary/#temporal-inflation)) layers — mostly [attention](/shared/glossary/#attention) along the time axis — inserted between the blocks of a [frozen](/shared/glossary/#frozen) image [U-Net](/shared/glossary/#u-net). The frozen image model still produces each frame's appearance; the motion module's only job is to look across frames and nudge them so the sequence moves smoothly instead of flickering independently. Think of it as a "motion adapter" you clip onto a still-image model — trained once on video, then reused unchanged across many image [checkpoints](/shared/glossary/#checkpoint).

### Motion score {#motion-score}
A single number handed to a video model that says *how much motion* a clip should contain — low for a near-still "animated photo", high for vigorous movement. During training it is measured from each real clip (commonly from the average [optical-flow](/shared/glossary/#optical-flow) magnitude between frames — how far pixels travel), so the model learns to associate the number with an amount of movement; at inference you set it by hand to dial motion up or down. [Stable Video Diffusion](/shared/glossary/#stable-video-diffusion-svd) calls its version the *motion bucket id*, sorting clips into discrete buckets of increasing motion rather than using a continuous value. It is the simplest control surface for video: one knob that separates *how much it moves* from *what is in it*.

### MoveIt {#moveit}
ROS 2 manipulation-planning framework

### Moving MNIST {#moving-mnist}
A simple synthetic video dataset built by taking handwritten digits from [MNIST](/shared/glossary/#mnist) and bouncing two of them around inside a black 64×64 frame, where they drift in straight lines and ricochet off the edges. The motion is perfectly predictable (constant velocity plus bounces), but the digits overlap and pass in front of each other, which is just hard enough to test a [future frame prediction](/shared/glossary/#future-frame-prediction) model without the cost and decoding pain of real video. It became the standard first benchmark for video-prediction models such as the [ConvLSTM](/shared/glossary/#convlstm).

### MPC {#mpc}
Model Predictive Control — re-solved finite-horizon optimization at each step

### MPS {#mps}
Metal Performance Shaders — the GPU backend for Apple Silicon

### MQA {#mqa}
Multi-Query Attention — all query heads share a single key/value head; the most aggressive [KV-cache](/shared/glossary/#kv-cache) saver, at some quality cost

### MSE (mean squared error) {#mse-mean-squared-error}
The most basic way to score how wrong a prediction is: at each point take the difference between the predicted and true value, square it (so overshoots and undershoots both count as positive, and big misses are punished extra), then average over all points. For images it compares pixel by pixel, so a guess that is a little off everywhere still scores well — which is exactly why training on MSE alone tends to produce *blurry* results: when the model is unsure, the safest low-MSE answer is to predict the average of all the plausible pixels, and an average of sharp options looks like a smudge. This is the failure a [perceptual loss](/shared/glossary/#perceptual-loss-lpips) is designed to avoid.

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
A model trained from scratch on *all* modalities at once over a single shared [vocabulary](/shared/glossary/#vocabulary), instead of bolting a vision encoder onto a finished language model. Every modality is turned into [tokens](/shared/glossary/#token-visualaudio) — text tokens, image tokens from a [VQ-VAE](/shared/glossary/#vq-vae), audio tokens from a [neural codec](/shared/glossary/#neural-codec) — that all live in one alphabet, and one [transformer](/shared/glossary/#transformer) reads and writes any mix of them with a single next-token objective. This is the [early-fusion](/shared/glossary/#fusion-earlymiddlelate) extreme, used by models like [Chameleon](/shared/glossary/#chameleon) and GPT-4o. Analogy: rather than hiring separate translators for each language and patching their notes together, you raise one person bilingual from birth, so switching between "languages" (modalities) is effortless and mid-thought. The payoff is true any-to-any flexibility; the cost is far more data and compute, since nothing is reused from a pretrained backbone.

### Negative prompt {#negative-prompt}
A second text prompt describing what you do *not* want in the image (e.g. "blurry, extra fingers, watermark"). It works through [classifier-free guidance](/shared/glossary/#cfg-classifier-free-guidance): instead of pushing away from a blank unconditional prediction, the model pushes away from the negative prompt's prediction and toward your real prompt — so naming a flaw steers the result away from it. Like telling an artist "paint a beach, and whatever you do, no people."

### Network in Network {#network-in-network}
A design idea where a tiny neural network is tucked *inside* a single layer of a bigger one, so that layer can do more thinking than a plain filter could. A normal [convolution layer](/shared/glossary/#convolution-layers) slides a simple filter that just takes a weighted sum of each [patch](/shared/glossary/#patch); a network-in-network slides a small multi-step mini-network over each patch instead, letting it recognize more complicated local patterns on the spot. Picture a factory line where, instead of one worker stamping each part, every station hides a little expert team that inspects and shapes the part before passing it on. The idea (from the 2013 *Network In Network* paper) inspired the [Inception network](/shared/glossary/#inception-network)'s building blocks — which is why Inception was nicknamed after the movie about a dream inside a dream.

### Neural codec {#neural-codec}
A neural network that learns to *compress* a signal — audio, an image, or video — into a compact code and then rebuild it, a learned cousin of hand-designed formats like MP3 or JPEG. ("Codec" = **co**der + **dec**oder.) The encoder squeezes the signal down to a small set of numbers or [tokens](/shared/glossary/#token-visualaudio) and the decoder reconstructs it; because the whole thing is trained on real data instead of hand-tuned, it can often pack more quality into fewer bits. A [VQ-VAE](/shared/glossary/#vq-vae) is one example used for images; for audio, **EnCodec** and SoundStream are the best-known examples, squeezing a waveform into a short stream of discrete [tokens](/shared/glossary/#token-visualaudio) at a chosen *bitrate* (bits per second) — so a lower bitrate means fewer tokens and rougher sound, a higher one means more tokens and cleaner audio.

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

### Noise schedule {#noise-schedule}
The recipe a [diffusion model](/shared/glossary/#diffusion-model) follows for *how much* noise to add at each step of its forward (noising) process — and therefore how much the denoiser must remove at each reverse step. A *linear* schedule raises the noise level by equal amounts every step; a *cosine* schedule ramps up gently at the start and end, keeping recognizable image structure alive for more of the process, which usually trains better. Think of it as a dimmer switch for how quickly a picture fades to static: turn it down too fast (linear) and most steps see only static, leaving little to learn from. The choice mainly affects training quality and how many sampling steps you need, not the model architecture.

### non_blocking {#non_blocking}
The `non_blocking=True` flag on `.to()` / `.cuda()` that lets a host→device copy run asynchronously from pinned memory

### Normalization {#normalization}
Rescaling a layer's outputs so they keep a consistent size — typically zero mean and unit variance (LayerNorm) or unit root-mean-square ([RMSNorm](/shared/glossary/#rmsnorm)). Like adjusting every photo to the same brightness before comparing them, it stops numbers from ballooning or vanishing as they flow through a deep network, which is what keeps training stable.

### Normalizing flow {#normalizing-flow}
A generative model that starts from simple random noise (usually a plain Gaussian "bell curve") and pushes it through a chain of *reversible* steps to reshape it into realistic data — like kneading a smooth ball of dough into a detailed shape, where you can always un-knead it back. Why can you *always* un-knead it? Because every step is deliberately built to be undoable: it only ever stretches, shifts, or folds the dough in a way that has an exact opposite, and it never merges two blobs into one or throws any dough away. For example, if a step's rule is "double this number and add 3," its reverse is simply "subtract 3, then halve" — feed the output back through and you recover the original number exactly, with nothing lost. (An ordinary neural network is *not* like this: it mashes information together — like flattening the dough — so there is no way to run it backwards.) Because every step can be run backwards exactly, a flow can also report the precise probability of any data point, which most generative models cannot do. The price for that exactness is that each step must stay reversible, which heavily constrains the architecture; examples include [Real NVP](/shared/glossary/#real-nvp) and [Glow](/shared/glossary/#glow).

### np.linalg.solve {#nplinalgsolve}
A NumPy function that solves a system of linear equations `Ax = b` for the unknown vector `x`, given a square matrix `A` and a known vector `b`. It is the practical alternative to forming the [matrix inverse](/shared/glossary/#matrix-inverse) and multiplying by it: rather than first computing `A⁻¹` and then `A⁻¹b` (two costly steps), it finds `x` in a single pass, which is both faster and less prone to rounding error. Analogy: to undo "multiply by 3" you don't first write out "1 ÷ 3" and then multiply — you just divide by 3 directly; `np.linalg.solve` divides by a whole matrix at once. In RL it gives the closed-form answer to [policy evaluation](/shared/glossary/#policy-evaluation) in one line — `V = np.linalg.solve(I − γPπ, rπ)` — the exact [value function](/shared/glossary/#value-function) for a fixed [policy](/shared/glossary/#policy) without iterating the [Bellman equation](/shared/glossary/#bellman-equation).

### NVAE {#nvae}
Short for *Nouveau VAE* — a [hierarchical VAE](/shared/glossary/#hierarchical-vae) from NVIDIA (2020) that stacks many layers of latent variables through a deep network built from depthwise separable [convolutions](/shared/glossary/#convolution-layers) and [residual connections](/shared/glossary/#residual-connection), reaching then state-of-the-art image generation quality. Like a skyscraper where each floor refines the blueprint handed down from above — the top floors sketch the overall shape and the lower floors fill in the fine details. The name "Nouveau" is French for "new," positioning it as a modern reimagining of the classic VAE.

### NVLink {#nvlink}
NVIDIA's GPU-GPU interconnect; much faster than PCIe

### NVSwitch {#nvswitch}
NVLink switch chip; full-bandwidth all-to-all within a node

### Observability {#observability}
The practice of making a running system's inner state visible from the outside — through metrics, logs, and traces — so you can ask new questions about *why* it is misbehaving without adding new code. Like the dashboard and warning lights in a car: you can tell what is wrong while still driving, instead of pulling the engine apart. For a serving stack it is the difference between knowing "p99 [latency](/shared/glossary/#latency) tripled at 9 a.m." and finding out only when users complain.

### Object permanence {#object-permanence}
The basic fact that objects keep existing even when you cannot see them — a ball that rolls behind a couch is still there and should reappear on the other side. For a video generator this is a surprisingly hard test: a model that only makes each frame look locally plausible may let an object silently vanish, change color, or duplicate while it is briefly hidden (occluded) and then revealed. Infants learn object permanence in their first year; generative video models still routinely fail it, which is why it is one of the world-behavior criteria [Sora](/shared/glossary/#sora)'s report singles out. It is one facet of the broader [physical plausibility](/shared/glossary/#physical-plausibility) problem and closely tied to [world consistency](/shared/glossary/#world-consistency).

### OCR (Optical Character Recognition) {#ocr-optical-character-recognition}
Reading the text *inside* an image — turning pixels of letters into actual characters a computer can use — for example pulling the line items off a photographed receipt or the words out of a scanned page. It is the skill that separates a [VLM](/shared/glossary/#vlm) that "sees a document" from one that can answer "what is the total?", and it is hard precisely because the answer often hides in small print that survives only if the image is fed in at high enough resolution (one reason [AnyRes](/shared/glossary/#anyres) tiling helps). Analogy: the difference between glancing at a street sign and actually reading the words on it. Example: given a photo of a price tag, an OCR-capable model returns the string "$19.99" rather than just "a label"; benchmarks like DocVQA and OCRBench score exactly this ability.

### ODE (Ordinary Differential Equation) {#ode}
A mathematical equation describing how a system's current state determines its rate of change (its slope). Rather than giving a fixed value as an answer, solving an ODE yields a full continuous function (a path). In diffusion models, the "Probability Flow ODE" acts as the exact navigation route transitioning pure random noise into a structured image. If the current state is a car's position, the ODE defines its exact velocity at that spot.

### Off-policy {#off-policy}
An RL algorithm where the data used for learning comes from a different policy than the one being improved. The agent learns the value of the [optimal policy](/shared/glossary/#optimal-policy) while actually following a more exploratory policy (like [ε-greedy](/shared/glossary/#epsilon-greedy)) to gather data. [Q-learning](/shared/glossary/#q-learning) and [DQN](/shared/glossary/#dqn) are off-policy methods. Unlike on-policy methods, off-policy methods can learn from old experience gathered by past versions of the agent or even by a completely different agent. Think of learning to play a game by watching someone else play: you figure out the best moves by observing their mistakes and successes, without having to make all those moves yourself.

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
An RL algorithm that learns exclusively from data gathered by the exact same policy it is currently trying to improve. If the agent explores by acting randomly 10% of the time, the policy it learns will account for that 10% randomness. [PPO](/shared/glossary/#ppo), [REINFORCE](/shared/glossary/#reinforce), and [SARSA](/shared/glossary/#sarsa) are on-policy methods. Unlike off-policy methods, on-policy methods learn only from the policy currently being executed — they cannot reuse old experience. Think of learning to play a game entirely through your own trial and error: you can only learn from the moves you are making right now, not by watching old recordings of yourself or others.

### Open-ended {#open-ended}
A task where many different answers can all be reasonable and there is no single right one to check against — writing a poem, summarizing an article, replying helpfully in a chat. The opposite of a closed-ended task like a multiple-choice question (one correct letter) or arithmetic (one correct number). Like grading a creative-writing assignment versus grading a true/false quiz: with the quiz you just count matches, but with the essay you need a human reader — or an [LLM-as-judge](/shared/glossary/#llm-as-judge) — to weigh quality, which is why evaluating open-ended work is the hard part of LLM evals.

### Open model {#open-model}
A model whose [weights](/shared/glossary/#weights) you can download and run yourself — Meta's Llama, Mistral, Qwen, DeepSeek — as opposed to a *closed* model like GPT-4 or Claude where the weights stay on the provider's servers and you can only call them through an API. Like the difference between buying a recipe book (you have the actual instructions, can modify them, can bake offline) and ordering at a restaurant (you only see the finished dish). Open models are essential for any white-box research that needs the model's internals: methods like [GCG](/shared/glossary/#gcg) optimize against the model's own [gradients](/shared/glossary/#gradients), and interpretability tools like [SAEs](/shared/glossary/#sae) read its hidden [activations](/shared/glossary/#activations) — neither is possible through a closed API.

### OpenSora {#opensora}
An open-source, fully documented re-implementation of OpenAI's [Sora](/shared/glossary/#sora) recipe, built by HPC-AI Tech (a closely related project, *Open-Sora-Plan*, comes from Peking University). It is a [DiT](/shared/glossary/#dit) that denoises [3D VAE](/shared/glossary/#3d-vae) [latents](/shared/glossary/#latent-video) and is trained with [flow matching](/shared/glossary/#flow-matching), with the code, model [weights](/shared/glossary/#weights), and data pipeline all released publicly. Because nothing is hidden behind an API, it is the standard way to study a complete Sora-style pipeline hands-on — run it, swap a component such as the [VAE](/shared/glossary/#vae), and retrain. Think of it as a working blueprint of a frontier video model that you are free to open up and rewire, where the closed originals only publish a sketch.

### Optical flow {#optical-flow}
A per-pixel map of motion between two frames: for every pixel it gives an arrow saying which direction and how far that bit of the image moved. "Dense" optical flow computes an arrow for *every* pixel, versus "sparse" flow, which tracks only a few chosen points. It is the rawest form of the "motion signal" in video and shows up everywhere — data filtering, frame interpolation, and motion conditioning. Analogy: imagine laying a sheet of thin see-through paper (tracing paper, the kind you can see a drawing through to copy it) over two snapshots taken a moment apart, then drawing a tiny arrow from where each speck — a tiny spot of detail in the picture — sat in the first frame to where it ended up in the second. Example: between two frames of a car driving right, every pixel on the car gets a rightward arrow while the still background gets near-zero arrows. Common ways to compute it are the classical [Farnebäck](/shared/glossary/#farnebäck-optical-flow) algorithm and the neural [RAFT](/shared/glossary/#raft) model.

### Optimizer {#optimizer}
An algorithm that updates model parameters using computed gradients; in PyTorch, a subclass of `torch.optim.Optimizer` that holds parameter groups and per-parameter state

### Optimizer state {#optimizer-state}
The extra per-parameter values an [optimizer](/shared/glossary/#optimizer) stores between steps — for example, [Adam](/shared/glossary/#adam) keeps two (the first- and second-moment estimates) — which adds to training memory.

### Orbit {#orbit}
A camera move that circles around a subject while keeping it centered in frame — like walking in a ring around a statue, always looking inward at it. Because the viewpoint travels around the object, you see its different sides in turn, which makes the orbit a demanding test of whether a video model keeps an object's 3D shape consistent as the angle changes. It is one of the paths a model can follow under [camera control](/shared/glossary/#camera-control).

### Optimal policy {#optimal-policy}
The [policy](/shared/glossary/#policy) (rule for choosing actions) that earns the highest expected [return](/shared/glossary/#value-function) from every state — no other policy does better anywhere. Its [value function](/shared/glossary/#value-function) is written `V*`, the one the Bellman *optimality* version of the [Bellman equation](/shared/glossary/#bellman-equation) describes. A key fact: which policy is optimal depends on the [discount factor](/shared/glossary/#discount-factor) — make the agent more short-sighted or more patient and the best action in a state can change. Solving an [MDP](/shared/glossary/#mdp) means finding this policy.

### Outcome reward model {#outcome-reward-model}
A scorer that judges only a solution's final answer as right or wrong, ignoring the steps in between — simpler than a [process reward model](/shared/glossary/#process-reward-model), which grades each step, but blind to *where* a wrong answer first went off track.

### Outlines {#outlines}
An open-source Python library for [constrained generation](/shared/glossary/#constrained-generation): you hand it a regular expression, a JSON schema, or a [Pydantic](/shared/glossary/#pydantic) model and it patches the [LLM](/shared/glossary/#llm)'s decoder to mask out any next-token choices that would break the structure. Like putting guardrails on a road so the car physically cannot drive off the edge no matter how the driver steers, it makes the model's output structurally valid by construction rather than by hope.

### Outpainting {#outpainting}
[Inpainting](/shared/glossary/#inpainting) applied to the *outside* of an image: you place the original on a larger blank canvas, mark the new border area as the region to fill, and let the model extend the scene outward so it continues naturally past the original frame. Like a painter adding more landscape beyond the edges of an existing painting.

### Overfitting {#overfitting}
When a model learns its training examples *too* literally — memorizing their specific details and noise instead of the general pattern — so it does great on the training set but poorly on anything new. A personalization [LoRA](/shared/glossary/#lora) trained for too many steps overfits: asked for "the subject on the moon," it just spits back one of its training photos. The classic analogy is a student who memorizes the exact answers to the practice exam and then fails the real test because the questions are worded differently. You spot it when training accuracy keeps improving while held-out performance gets worse, and you fight it with more data, fewer training steps, or regularization. Its opposite — doing well on unseen inputs — is [generalization](/shared/glossary/#generalization).

### Padding {#padding}
Filling shorter sequences with a placeholder value so that every [sample](/shared/glossary/#sample) in a batch has the same length.

### Pan {#pan}
A camera move where the camera stays in one spot but rotates left or right — like standing still and turning only your head to sweep your gaze across a room. The viewpoint's position never changes, only the direction it faces, so near and far objects slide across the frame together. It is one of the basic moves a video model learns to follow under [camera control](/shared/glossary/#camera-control).

### Parameters {#parameters}
The numbers a model *learns* during training — its adjustable internal settings. Picture thousands of tiny knobs on a giant mixing board: training nudges each knob a little at a time until the whole board produces good output, and the final knob positions *are* what the model "knows." They come in two kinds — [weights](/shared/glossary/#weights) and [biases](/shared/glossary/#biases) — are stored as [tensors](/shared/glossary/#tensor), and are adjusted by the [optimizer](/shared/glossary/#optimizer) during training. (When people say a "7B model," they mean 7 billion of these knobs.) In PyTorch they are `nn.Parameter` objects, registered automatically when assigned to an [`nn.Module`](/shared/glossary/#nnmodule).

### Partial derivative {#partial-derivative}
How much a function changes when you nudge just one of its inputs and hold all the others still — the [derivative](/shared/glossary/#derivative) taken one input at a time. If a recipe's tastiness depends on both salt and sugar, the partial derivative with respect to salt tells you the effect of adding a pinch more salt while keeping the sugar fixed. A [gradient](/shared/glossary/#gradients) is simply the full list of these one-at-a-time slopes, one per [parameter](/shared/glossary/#parameters).

### PagedAttention {#pagedattention}
A way of storing the [KV cache](/shared/glossary/#kv-cache) for many concurrent requests by splitting each request's cache into small fixed-size "pages" that the engine can scatter freely around GPU memory and look up through a per-request page table — the same idea operating systems use for virtual memory. It removes the wasted space and fragmentation you get when each request needs its own contiguous chunk, which is why [vLLM](/shared/glossary/#vllm) made it the default scheme.

### Patch {#patch}
A small rectangular section of an image. Instead of looking at an entire image at once, models often break it down into a grid of these smaller blocks to process them one by one. Like cutting a jigsaw puzzle into individual pieces and examining each piece separately before seeing how they fit together.

### Patchification {#patchification}
Splitting a (latent) tensor into a sequence of small square [patches](/shared/glossary/#patch) and turning each one into a single token, so a [transformer](/shared/glossary/#transformer) can treat an image like a sentence of words. For example, a 32×32 latent cut into 2×2 patches becomes a sequence of 256 tokens (a 16×16 grid), each a little block projected to the model's hidden width. The patch size is the key knob: smaller patches make more tokens (finer detail but more compute), bigger patches make fewer tokens (cheaper but coarser) — a suffix like "/2" in [DiT](/shared/glossary/#dit)-S/2 means patch size 2. Like slicing a photo into postage-stamp squares and reading them left-to-right, top-to-bottom. The same idea extends to video by cutting [spatiotemporal patches](/shared/glossary/#spatiotemporal-patches) — little 3D boxes that also span a few frames in time, so one sequence of tokens carries both motion and appearance.

### PCA (principal component analysis) {#pca-principal-component-analysis}
A technique that finds the few directions along which data varies the most and uses them to compress many numbers down to a handful, so high-dimensional data can be drawn on a 2D plot. Imagine photographing a 3D object from the angle that reveals its shape best — PCA picks that most-informative "camera angle" automatically. It is a quick, standard first step for *seeing* the structure in data, such as checking whether real images cluster together while random noise scatters apart.

### PCIe {#pcie}
The standard CPU-GPU connection (and slower GPU-GPU when no NVLink)

### Perceiver IO {#perceiver-io}
DeepMind's modality-agnostic architecture that handles inputs of any size or type — pixels, audio samples, point clouds — without the cost normally blowing up. Plain [attention](/shared/glossary/#attention) compares every input element with every other, so a million-pixel image would need a million-by-million grid; Perceiver instead keeps a *small fixed set of learned latent vectors* (say 256 of them) and lets only those latents [cross-attend](/shared/glossary/#cross-attention) to the giant input, squeezing it into the small set once, then doing all the heavy processing among just the 256. The "IO" version adds a matching trick on the output side: a set of learned *query* vectors cross-attends to the processed latents to read out an answer of whatever shape you need. Like a small committee (the latents) that skims a huge pile of documents, takes compact notes, deliberates among themselves, and then answers any question put to them — the committee's workload depends on its own size, not on how tall the pile was. Because nothing in it assumes a grid or a sequence, the same architecture works across [modalities](/shared/glossary/#modality) with almost no changes, which is its headline selling point. It is closely related to the [Q-Former](/shared/glossary/#q-former), which uses the same small-set-of-learned-queries idea to distill an image for a language model.

### Percentile {#percentile}
A way to describe where a value ranks in a sorted list: the p99 [latency](/shared/glossary/#latency) is the time that 99% of requests beat, with only the slowest 1% taking longer. Unlike an average, which a single huge outlier can hide, percentiles expose the slow tail that users actually feel — like reporting "even the slowest of the top 99% of diners was served within 20 minutes" instead of a misleading table-wide average. Serving teams quote p50, p95, and p99 rather than the mean for exactly this reason.

### Perceptual loss (LPIPS) {#perceptual-loss-lpips}
A loss that compares two images by the features a pretrained network sees in them, rather than by their raw pixels. Two photos shifted by a single pixel are nearly identical to a human eye but very different under pixel-by-pixel error; a perceptual loss judges them the way an eye does, rewarding matching textures and shapes. Training with it (LPIPS — Learned Perceptual Image Patch Similarity — is the popular version) gives much sharper results than plain pixel [MSE](/shared/glossary/#mse-mean-squared-error), which tends to blur. It is widely used inside [VQ-GAN](/shared/glossary/#vq-gan) and VAE training.

### permute {#permute}
Reorders all of a tensor's dimensions by rewriting strides — never copies

### Perplexity {#perplexity}
A score for how *surprised* a language model is by a piece of text — roughly, how many words it was effectively choosing between at each step. Lower is better: a perplexity of 1 means the model knew exactly what came next, while a high number means it was guessing wildly. Because it is cheap to compute and rises the moment a model gets worse, it is a common first [tripwire](/shared/glossary/#tripwire) in a [quality gate](/shared/glossary/#quality-gate) after [quantization](/shared/glossary/#quantization).

### Physical plausibility {#physical-plausibility}
Whether the motion and interactions in a generated video obey the everyday rules of the physical world — gravity pulls things down, water flows downhill, solid objects do not pass through each other, dropped things fall instead of hovering. Also called *physical realism* or *physical correctness*. The catch is that a clip can score well on appearance metrics like [FVD](/shared/glossary/#fvd) — every frame looks sharp and real — while still getting the physics badly wrong, because looking right and behaving right are different things. It stays largely unmeasured: there is no clean benchmark for it yet, so it is usually probed by hand with trick prompts (water flowing uphill, a glass that should shatter). [Object permanence](/shared/glossary/#object-permanence) and [world consistency](/shared/glossary/#world-consistency) are specific facets of it.

### PID {#pid}
Proportional-Integral-Derivative — the workhorse linear controller

### Pinned memory {#pinned-memory}
Page-locked CPU memory that enables faster, asynchronous transfers to the GPU; enabled with `pin_memory=True` on a [DataLoader](/shared/glossary/#dataloader).

### Pinocchio {#pinocchio}
Fast rigid-body dynamics library (CRBA, RNEA, ABA)

### PixelCNN {#pixelcnn}
An [autoregressive](/shared/glossary/#autoregressive-model) image model — a [CNN (Convolutional Neural Network)](/shared/glossary/#cnn) repurposed for generation — that draws a picture one pixel at a time, predicting each pixel from the pixels already drawn above it and to its left — like filling in a coloring grid square by square, always glancing back at what you have already colored to decide the next color. The image quality is strong and it can report an exact [probability](/shared/glossary/#probability-density) for any picture, but generating one is slow because the pixels must come out strictly in order, each waiting on the one before it.

### Plücker coordinates {#plücker-coordinates}
A way to describe a single straight line (here, the ray of sight through one pixel) using six numbers instead of a point-plus-direction. The six split into the ray's direction and its *moment* (a [cross product](/shared/glossary/#cross-product) that pins down which parallel line it is), so a line floating anywhere in 3D space gets one compact, position-independent code. Video models use them for [camera control](/shared/glossary/#camera-control): give every pixel of every frame its Plücker ray and the model knows exactly which way the camera is looking, which lets learned camera moves generalize to angles never seen in training — far better than feeding raw camera-position numbers. Named after the 19th-century mathematician Julius Plücker, who introduced this line geometry.

### PoC {#poc}
Proof of Concept — a small, rough build whose only job is to show that an idea *can* work, before anyone invests in a polished version. Like frying one test pancake to check the batter before making the whole stack: you are not trying to serve it, just to learn whether the approach is sound.

### Point cloud {#point-cloud}
A loose scatter of dots in space, where each dot is one data item placed by its numbers. Turn every image in a [batch](/shared/glossary/#batch) into a [feature vector](/shared/glossary/#inception-network) — a single point — and the whole batch becomes a cloud of such points. Comparing two clouds (say, real images vs. generated ones) is how a metric like [FID](/shared/glossary/#fid) measures similarity: it is like comparing two swarms of bees and asking whether they are hovering in the same spot and spread out in the same shape.

### Policy {#policy}
In reinforcement learning, the model being trained to choose what to do next — for an [LLM](/shared/glossary/#llm), the network that picks the next token. "Improving the policy" just means making those choices earn more reward.

### Policy evaluation {#policy-evaluation}
The task of computing the [value function](/shared/glossary/#value-function) of a *given, fixed* [policy](/shared/glossary/#policy) — answering "if I always act this way, how much reward should I expect from each state?" — without yet trying to improve the policy. Because the policy is fixed, the [Bellman equation](/shared/glossary/#bellman-equation) becomes a set of *linear* equations with one unknown per state, so it can be solved two ways: in one shot with a [matrix inverse](/shared/glossary/#matrix-inverse), or by repeatedly applying the [Bellman operator](/shared/glossary/#bellman-operator) until the numbers stop changing. It is the evaluation step that, alternated with a policy-improvement step, builds up to the [optimal policy](/shared/glossary/#optimal-policy).

### Policy iteration {#policy-iteration}
A [dynamic-programming](/shared/glossary/#dynamic-programming) algorithm that solves a known [MDP](/shared/glossary/#mdp) by strictly alternating two phases until neither changes: [policy evaluation](/shared/glossary/#policy-evaluation) — compute the [value function](/shared/glossary/#value-function) of the current [policy](/shared/glossary/#policy) exactly — and *policy improvement* — replace the policy with the [greedy](/shared/glossary/#greedy-policy) one with respect to those fresh values. Each improvement is guaranteed to give a policy at least as good, and because a finite MDP has only finitely many policies, it reaches the [optimal policy](/shared/glossary/#optimal-policy) in surprisingly few rounds — usually far fewer than [value iteration](/shared/glossary/#value-iteration), though each round costs more because the evaluation phase is solved to completion. Unlike value iteration, which takes a tiny step of improvement after every single evaluation sweep, policy iteration fully evaluates the policy before improving it. Think of finding the best route to work: policy iteration is like driving one specific route every day for a month until you perfectly know its average time, then choosing a new route to test; value iteration is like driving a route once and immediately updating your guess for the best path. It is the original instance of [generalized policy iteration](/shared/glossary/#generalized-policy-iteration).

### POMDP {#pomdp}
Partially Observable Markov Decision Process — an [MDP](/shared/glossary/#mdp) in which the agent does **not** see the true state, only a partial or noisy *observation* of it. Because that observation alone breaks the [Markov property](/shared/glossary/#markov-property) (it no longer holds everything needed to act well), a [policy](/shared/glossary/#policy) that reacts only to the current observation is provably suboptimal — to act well the agent generally has to *remember* past observations, e.g. with a [recurrent network](/shared/glossary/#recurrent-network) or an explicit belief over which state it is probably in. Example: a robot with only a forward camera that cannot see what is behind it, or a poker player who cannot see opponents' cards. Many real problems that look like MDPs are secretly POMDPs because the "state" the designer chose leaves something important out.

### Position bias {#position-bias}
A judge's tendency to pick an answer based on *where* it sits rather than *what* it says — for example, an [LLM-as-judge](/shared/glossary/#llm-as-judge) that quietly prefers whichever response appears first (or last) when shown two side-by-side. Like a job interviewer who can't help favoring the candidate they meet right after lunch, regardless of qualifications. The standard fix is to ask the judge twice with the two answers swapped and accept the verdict only if both runs name the same winner.

### Position interpolation {#position-interpolation}
Extending a model's context length by linearly rescaling [RoPE](/shared/glossary/#rope) position indices so longer sequences fall within the trained range

### Position vector {#position-vector}
A vector that represents the exact location of a specific point in space. You make one by drawing a straight geometric arrow from the origin — the `(0, 0, 0)` center of the coordinate system — directly to your target point. If a point lives at coordinates `(x, y, z)`, its position vector is simply the vector `[x, y, z]`. 

Why do we need this? In geometry, a *point* is just a fixed location, while a general *vector* is just a movement (a direction and a length, like "walk 5 steps North") that can float anywhere in space. A position vector bridges the two: by permanently anchoring the tail of the arrow to the origin, the vector perfectly describes that specific location. This is the mathematical trick that lets you plug a fixed point into vector operations like the [cross product](/shared/glossary/#cross-product).

### Posterior collapse {#posterior-collapse}
A [VAE](/shared/glossary/#vae) failure where the decoder grows strong enough to reconstruct inputs on its own and simply ignores the [latent space](/shared/glossary/#latent-space). The encoder then stops bothering to encode anything and just outputs the default prior, so the latent variables carry no information about the input — like a student who has memorized the answer key and no longer reads the question. When this happens the [KL divergence](/shared/glossary/#kl-divergence) term drops toward zero and the latent code becomes useless for generation.

### Postmortem {#postmortem}
A written review done after an incident — an outage, a slowdown — that lays out what happened, how it was detected and fixed, and what will stop it recurring. A good one is *blameless*: it focuses on the system and the process, not on punishing a person, like an air-crash investigation whose goal is safer future flights rather than someone to fire.

### PPO {#ppo}
Proximal Policy Optimization — the [workhorse](/shared/glossary/#workhorse) [on-policy](/shared/glossary/#on-policy) RL algorithm, used in classic RLHF

### Precision and recall {#precision-and-recall}
Two numbers that, used together, describe how a yes/no detector is doing — far more honest than a single accuracy figure. *Precision* asks "when the model says yes, how often is it right?" — of all the times it shouted "dog!", what fraction really had a dog. *Recall* asks "of all the real yes-cases, how many did it catch?" — of all the images that truly had a dog, how many it found. You compute each as a simple fraction: precision = true positives / (true positives + false positives); recall = true positives / (true positives + false negatives). They trade off against each other — a model that says "yes" to everything has perfect recall but terrible precision — which is exactly why a [hallucination](/shared/glossary/#hallucination) probe must report both, not just accuracy. Analogy: a fisherman's net — precision is how much of the catch is the fish you actually wanted (not boots and weeds), and recall is how many of the lake's fish you managed to net at all.

### Prefill {#prefill}
The first stage of LLM inference: reading the *entire* prompt at once to fill the [KV cache](/shared/glossary/#kv-cache), before any new tokens are generated. Because all the prompt's tokens can be processed together in a single [forward pass](/shared/glossary/#forward-pass), prefill is compute-heavy and fast per token — like a reader skimming a whole page at a glance to grasp it before starting to write a reply. It is the opposite of [decode](/shared/glossary/#decode), which then produces the answer one token at a time, and prefill time is what sets the [time to first token](/shared/glossary/#ttft).

### Prefix cache {#prefix-cache}
Sharing KV cache across requests that begin with the same tokens (e.g., system prompts)

### Pretraining {#pretraining}
Self-supervised training on a large unlabeled corpus to predict the next token

### Prior-preservation loss {#prior-preservation-loss}
An extra training term used by [DreamBooth](/shared/glossary/#dreambooth) to stop a model from forgetting a whole *class* while learning one specific member of it. When you fine-tune on five photos of *your* dog, the model risks deciding every "dog" now looks like yours — a form of [catastrophic forgetting](/shared/glossary/#catastrophic-forgetting). Prior preservation counters this by mixing in the model's *own* generic "a photo of a dog" images during training and asking it to keep reproducing them, so the broad concept of "dog" is preserved while the narrow concept of *your* dog is added on top. Like teaching someone your cousin's face without making them forget what faces in general look like.

### PRM {#prm}
Probabilistic Roadmap — multi-query sampling-based planner

### PRM800K {#prm800k}
A public dataset of about 800,000 human labels that mark each step of a math solution as right or wrong, released by OpenAI to train [process reward models](/shared/glossary/#process-reward-model). Rather than only checking whether the final answer was correct, human graders read each worked solution line by line — like a math teacher putting a check or an X next to every step of a student's proof, not just the boxed answer at the bottom. Because the feedback is step-level, a model trained on it learns to spot exactly where the reasoning went off the rails instead of whether the ending happened to be lucky. It is the standard training set for the step-by-step scorers used in [Best-of-N](/shared/glossary/#best-of-n) re-ranking.

### Probability density {#probability-density}
A function that says how *likely* each possible value is — high where real data points pile up, low in the empty regions where they rarely fall. For a 2D dataset you can picture it as a heatmap: bright ridges over the crowded spots, dark valleys over the bare ones. It must stay non-negative everywhere, and all of it added up (the total volume under the surface) equals exactly 1, since *some* value always occurs. Most generative models can only *draw* new samples; a [normalizing flow](/shared/glossary/#normalizing-flow) is special because it can also report the exact probability density of any point you hand it.

### Probability flow ODE {#probability-flow-ode}
The deterministic twin of a diffusion model's reverse-time [SDE](/shared/glossary/#sde-stochastic-differential-equation): an [ODE](/shared/glossary/#ode) with no injected randomness that produces the *same* distribution of images at every noise level. Determinism buys two things the stochastic sampler can't: the same starting noise always maps to the same image (so you can interpolate between samples and invert a real image back to its noise), and the model's exact log-likelihood of any image — how probable it thinks that image is — becomes computable via the ODE's change-of-variables. It is the basis of fast deterministic samplers like [DDIM](/shared/glossary/#ddim).

### Process reward model {#process-reward-model}
A scorer that grades each individual step of a model's reasoning rather than just the final answer — like a teacher marking every line of a proof, not only the last one — so a mistake can be caught at the exact step it happens. Contrast with an [outcome reward model](/shared/glossary/#outcome-reward-model).

### Profiler {#profiler}
A tool (`torch.profiler`) that records how long each operation in a training step takes, used to locate performance [bottlenecks](/shared/glossary/#bottleneck).

### Projection discriminator {#projection-discriminator}
A way to feed a class label into a [conditional GAN](/shared/glossary/#conditional-gan-cgan)'s [discriminator](/shared/glossary/#discriminator) by taking a dot product between the image's features and a learned vector for that class, then adding it to the score — rather than just gluing the label on as an extra input. This matches how the math of conditioning actually factorizes, so it conditions more strongly for almost no extra cost, and it became the standard trick for class-conditional GANs such as [BigGAN](/shared/glossary/#biggan).

### Projector {#projector}
The small network — often a single linear layer or a two-layer [MLP](/shared/glossary/#mlp) — that maps one [modality](/shared/glossary/#modality)'s feature vectors into the space another model expects. It initially acts as a physical adapter cable that reshapes one plug into another (e.g., resizing a 1024-dimensional image vector into a 4096-dimensional word vector). Crucially, simply matching dimensions is not enough; the projector must undergo *alignment training* (like installing a software driver for the adapter) to learn the exact mathematical transformation that routes the visual semantics into the LLM's native coordinate space. This is the entire fusion mechanism in [LLaVA](/shared/glossary/#llava): freeze the vision encoder, freeze the LLM, and train only this projector to perfectly align the two spaces. The catch is that all the image information must squeeze through this one thin bridge, so it can become a bottleneck on detail-heavy tasks.

### Prompt injection {#prompt-injection}
An attack in which adversarial text smuggled into something the model reads — a retrieved document, a tool's output, an email, even text inside an image — overrides the original system instructions. Like a customer slipping a fake "manager-approved" note into a server's order pile: the server can't easily tell the planted note from a real one. The hardest unsolved security problem in deployed [LLMs](/shared/glossary/#llm), because the model has no built-in way to separate "instructions" from "data" in its input.

### Prompt-to-Prompt {#prompt-to-prompt}
A diffusion editing technique that changes *what* an image shows while keeping its layout intact, by reusing the [cross-attention](/shared/glossary/#cross-attention) maps from the original generation. Those attention maps act like a set of stencils, recording exactly *which word controls which region* of the image (for example, the map for the word "cat" literally points to the pixels where the cat is drawn). If you change the prompt from "cat" to "dog" but force the new run to reuse the old attention maps—meaning you tell the model, "draw the dog exactly inside the stencil you previously used for the cat"—the dog lands in exactly the same pose and place as the cat. Picture keeping a painting's pencil under-drawing fixed and only changing the colors you fill in. It is one of the tools used to build paired before/after data for [InstructPix2Pix](/shared/glossary/#instructpix2pix).

### PTQ / QAT {#ptq--qat}
Post-Training Quantization / Quantization-Aware Training

### Pydantic {#pydantic}
A popular Python library for declaring the *shape* of your data as a class — you write a class with typed fields (e.g. `name: str`, `age: int`) and Pydantic validates that any data you load actually matches, raising a clear error if a value is the wrong type or a required field is missing. Like a customs form for data: anything that does not match the listed fields gets stopped at the border. In LLM work it is the standard way to describe the JSON object you want the model to produce, which tools like [Outlines](/shared/glossary/#outlines) or OpenAI's structured-output mode can then enforce during decoding.

### Q-Former {#q-former}
The fusion module from BLIP-2 (a 2023 [vision-language model](/shared/glossary/#vlm)) that shrinks a whole image down to a *fixed* small number of tokens — typically 32 — that a language model can read. It holds a set of learned *query* vectors that [cross-attend](/shared/glossary/#cross-attention) to the [frozen](/shared/glossary/#frozen) image encoder's many patch features, each query pulling out one summary of what it cares about; the 32 outputs are then [projected](/shared/glossary/#projector) and fed to the LLM as if they were 32 word [tokens](/shared/glossary/#token-visualaudio). Like 32 interviewers who each question a sprawling exhibit and walk away with one concise note, so the language model reads 32 notes instead of touring the whole gallery. The point of the fixed count is cost control: an image becomes a constant, small number of tokens no matter its resolution, instead of hundreds. It shares the small-set-of-learned-queries idea with the [Perceiver IO](/shared/glossary/#perceiver-io); later VLMs like LLaVA showed a plain [projector](/shared/glossary/#projector) often matches it with less complexity.

### Q-learning {#q-learning}
The foundational [off-policy](/shared/glossary/#off-policy) [temporal-difference](/shared/glossary/#temporal-difference-learning) control algorithm. It keeps a table of [action-values](/shared/glossary/#value-function) `Q(s, a)` and, after each `(state, action, reward, next-state)` transition, nudges `Q(s, a)` toward `r + γ · maxₐ′ Q(s′, a′)` — "reward now plus the [discounted](/shared/glossary/#discount-factor) value of the *best* action available next." Using that **max** in the target — rather than the action the agent actually goes on to take — is what makes it off-policy: it learns the value of the [optimal policy](/shared/glossary/#optimal-policy) even while exploring with a softer rule like [ε-greedy](/shared/glossary/#epsilon-greedy). Its on-policy cousin [SARSA](/shared/glossary/#sarsa) instead plugs in the action actually taken next, learning more cautious behavior; swapping the lookup table for a neural network is the leap to deep Q-networks in [Phase 3](/guides/reinforcement-learning/#phase-3-function-approximation-and-dqn).

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

### RAFT {#raft}
RAFT (Recurrent All-Pairs Field Transforms) is a neural network for computing dense [optical flow](/shared/glossary/#optical-flow), and on release one of the most accurate. Its core idea is to compare *all pairs* of pixels between the two frames to build a similarity volume, then iteratively refine a flow estimate with a recurrent update — repeatedly nudging the guess until it stops improving (which is the "recurrent" in its name). Analogy: it is a careful editor who, instead of guessing the motion once, keeps revising the answer over many small passes. Example: feeding RAFT two adjacent video frames returns a `(H, W, 2)` flow field that is far cleaner on fast motion than the classical [Farnebäck](/shared/glossary/#farnebäck-optical-flow) method.

### RAG {#rag}
Retrieval-Augmented Generation — give the model an "open-book exam" instead of asking it to answer from memory alone. First a search step fetches the documents most relevant to the question (from a company wiki, a manual, the web), then those documents are pasted into the prompt, and only then does the model write its answer using them as notes. This lets it use fresh or private facts it was never trained on, and makes it easy to check where an answer came from.

### rank {#rank}
The unique integer ID of a process in a distributed job. `RANK` is the global ID across all machines; `LOCAL_RANK` is the ID within one machine; `WORLD_SIZE` is the total number of processes.

### Raster order {#raster-order}
Walking through a 2D grid of pixels (or image [tokens](/shared/glossary/#token-visualaudio)) one row at a time, left to right and top to bottom — the exact path your eyes take reading a page. The name comes from how old CRT TVs and monitors painted the screen: an electron beam swept across in horizontal lines called *raster* lines (from the Latin *rastrum*, "rake," because the lines look raked across the glass). An [autoregressive](/shared/glossary/#autoregressive-model) image model that generates in raster order produces the top-left pixel first and the bottom-right pixel last.

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
A [flow-matching](/shared/glossary/#flow-matching) parameterization whose training paths are *straight lines* from noise to data, popular in 2024+ models like [SD3](/shared/glossary/#sd3) and [Flux](/shared/glossary/#flux). Straight trajectories are easy to follow in a few big steps, so sampling needs fewer steps than the curvy paths of older diffusion. You can also "re-flow": after training once, use the model to generate (noise, image) pairs and retrain on those straight pairs, which straightens the paths even further and lets you sample in as few as one or two steps. Like replacing a winding mountain road between two towns with a straight highway — same destination, far fewer turns to take.

### Recurrent network {#recurrent-network}
A recurrent neural network (RNN) is a network built for *sequences* — text, audio, a stream of video frames — that reads one item at a time and carries a running summary, its *hidden state*, forward from step to step. That looping of the hidden state back into the next step is the "recurrent" part: the same small network is applied again and again, and its memory of everything seen so far is what lets it act on context rather than on the current item alone. Analogy: reading a sentence word by word while keeping a mental note of what came before, so "it" near the end still refers to the right thing. Plain RNNs struggle to hold information over long sequences (the running note fades), which is why gated variants like the [LSTM](/shared/glossary/#lstm) were invented. In RL, a recurrent network is the standard way to give an agent memory in a [POMDP](/shared/glossary/#pomdp), where it must summarize past observations to recover the missing state.

### Reference model {#reference-model}
A frozen copy of the starting model that [RLHF](/shared/glossary/#rlhf) and [DPO](/shared/glossary/#dpo) measure against (through a [KL](/shared/glossary/#kl-divergence) term) so the model being trained does not drift too far from sensible behavior — a "before" photo to compare every change against.

### REINFORCE {#reinforce}
The foundational [on-policy](/shared/glossary/#on-policy) policy-gradient algorithm. Instead of learning a [value function](/shared/glossary/#value-function) to guess how good actions are, REINFORCE directly updates the policy's [weights](/shared/glossary/#weights) using the [returns](/shared/glossary/#return) from complete [rollouts](/shared/glossary/#rollout). If an episode went well, it nudges the weights to make all the actions taken during that episode more likely; if it went poorly, it makes them less likely. Analogy: training a pet with treats—you do not explain the exact mechanics of a trick (a value function), you just reward a successful attempt (the return) so the pet is more likely to repeat that exact sequence of movements in the future.

### Rejection sampling {#rejection-sampling}
A way to draw samples from a target distribution by proposing easy guesses and keeping or throwing away each one with just the right probability, so the survivors are distributed exactly as if they came from the hard distribution directly. The "right probability" of keeping a guess is `min(1, p ÷ q)`, where `p` is how likely the [target model](/shared/glossary/#target-model) thinks that token is and `q` is how likely the [draft model](/shared/glossary/#draft-model) thought it was. The rule is intuitive: if the target wants the token at least as much as the draft did (`p ≥ q`), always keep it; if the target wants it only half as much (`p` is half of `q`), keep it half the time and otherwise draw a replacement. For example, the draft proposes "cat" with `q = 0.6` but the target only gives it `p = 0.3`, so you keep "cat" with probability `0.3 ÷ 0.6 = 0.5` — a coin flip — which exactly cancels the draft's over-eagerness for that word. In [speculative decoding](/shared/glossary/#speculative-decoding) this is the step that lets a draft model's guesses be reused for random [sampling](/shared/glossary/#sampling) without changing the target model's true output distribution.

### ReLU {#relu}
Rectified Linear Unit — the most common and simplest [activation function](/shared/glossary/#activations): it keeps positive numbers unchanged and turns every negative number into 0 (`max(0, x)`). Like a one-way valve that lets water through in one direction and blocks it in the other. That single sharp bend is enough to give a network its non-linear power, and because it is so cheap to compute it was the default for years; newer models often swap it for smoother curves like [Swish](/shared/glossary/#swish) or [GELU](/shared/glossary/#gelu).

### Reparameterization trick {#reparameterization-trick}
A method to keep the training signal flowing through a random sampling step, enabling models like [VAEs](/shared/glossary/#vae) to be trained with ordinary backpropagation. 
* **The Problem:** Drawing the latent variable `z` directly from the encoder's distribution introduces randomness that blocks the flow of [gradients](/shared/glossary/#gradients).
* **The Solution:** The trick separates the randomness by drawing plain noise `ε` from a fixed standard normal distribution (a bell curve). You then compute `z = μ + σ · ε`.
* **Why it Works:** The randomness is now isolated in `ε` (which has no learnable parts). As a result, the network's `μ` and `σ` remain on a clean, differentiable path. 
* **Analogy:** It is like rolling one shared die outside the machine and then scaling the result, rather than building the dice into the machine itself.

### Reranker {#reranker}
A second-stage model that re-scores the top candidates from a fast first-stage retriever and reorders them by true relevance — usually a [cross-encoder](/shared/glossary/#cross-encoder). The "retrieve then rerank" two-stage pattern is standard in search and [RAG](/shared/glossary/#rag).

### reshape {#reshape}
Returns a tensor with a new shape, copying only when a no-copy view isn't possible

### Residual connection {#residual-connection}
A shortcut that adds a block's *input* straight onto its *output* — written `output = x + f(x)`, where `x` is what went in and `f(x)` is what the block computed. Instead of each block having to rebuild the whole signal from scratch, the original `x` flows past it on an express lane and the block only contributes a small `f(x)` correction on top. Think of editing a draft: rather than rewriting the entire essay at every pass, each editor keeps the existing text and just marks up the few changes that improve it.

What does "adds a block's input to its output" actually buy you? Two big things:

- **An easy "do nothing" default.** If a block has nothing useful to add, it can simply output near-zero, and `x + 0 = x` passes the input through unchanged. So adding more layers can never make things *worse* than the layers already learned — a new block starts from "leave it alone" and only departs from that when it finds something helpful. (This is exactly why [AdaLN-Zero](/shared/glossary/#adaln-zero) zero-initializes its gate: each block begins as a clean pass-through.)
- **A gradient highway.** On the [backward pass](/shared/glossary/#backward-pass), the `+ x` term hands every layer a direct path back to the earlier layers, so [gradients](/shared/glossary/#gradients) don't shrink toward zero as they travel through many layers (the [vanishing-gradients](/shared/glossary/#vanishing-gradients) problem). That direct path is what makes very [deep networks](/shared/glossary/#deep-network) trainable at all — before residual connections, stacking 50+ layers usually trained *worse*, not better. It is the same [skip-and-add logic](/shared/glossary/#skip-and-add-logic) found in convolutional nets, and it is what carries the [residual stream](/shared/glossary/#residual-stream) through a [transformer](/shared/glossary/#transformer).

### Residual parameterization {#residual-parameterization}
A modeling trick used in deep [hierarchical VAEs](/shared/glossary/#hierarchical-vae) where each layer of latent variables is expressed as a small *correction* to what the previous layer already predicted, rather than as a full absolute value. Like a GPS giving "turn left in 200 m" instead of stating exact coordinates — each step describes only the gap from where you already are, so no single step has to carry the whole story. Because each latent group only needs to represent a tiny residual change, gradients flow smoothly through many stacked layers and very deep hierarchies become trainable. The idea borrows from [residual connections](/shared/glossary/#residual-connection) in standard networks, applying the same [skip-and-add logic](/shared/glossary/#skip-and-add-logic) to the latent variable structure itself.

### Residual stream {#residual-stream}
In a [transformer](/shared/glossary/#transformer), the running activation vector that flows through every layer via [residual connections](/shared/glossary/#residual-connection) — each [attention](/shared/glossary/#attention) block and [MLP](/shared/glossary/#mlp) block reads from this stream and adds its update back to it, without erasing what came before. Like a shared bulletin board that every department reads and pins notes to as it passes through the office: by the end of the building, the board carries the combined contribution of every team. Because every layer reads and writes the same vector space, the residual stream is the most natural place to look for interpretable features, which is why [sparse autoencoders (SAEs)](/shared/glossary/#sae) are usually trained on residual-stream [activations](/shared/glossary/#activations).

### ResNet {#resnet}
Residual Network — a deep [CNN](/shared/glossary/#cnn) whose layers each learn a small *change* to add to their input rather than a brand-new output, thanks to [residual (skip) connections](/shared/glossary/#residual-connection) that route the input straight past each block. The name is short for "residual," the leftover the layer adds on top. Before ResNet, stacking many layers made networks *harder* to train because the signal degraded on its way through; letting each block default to "pass the input through unchanged, plus a tweak" means adding depth can only help. Like a relay of editors who each suggest small edits to a draft instead of rewriting it from scratch — the original text is never lost. ResNet-50 (50 layers) is still a common, sturdy baseline image encoder.

### Return {#return}
The total reward an agent collects from a moment onward, discounted so that sooner rewards count for more: `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`, with each step shrunk by the [discount factor](/shared/glossary/#discount-factor) γ (0 ≤ γ < 1). The geometric shrinking keeps the sum finite even over an endless task and bakes in "a reward now is worth more than the same reward later." The [value function](/shared/glossary/#value-function) is just the *expected* return, so the return is the raw quantity that every value estimate — and every [Monte Carlo](/shared/glossary/#monte-carlo-method) average — is ultimately trying to predict.

### reverse-mode {#reverse-mode}
The order [autograd](/shared/glossary/#autograd) walks the computation graph when differentiating: the forward pass first, then a single [backward pass](/shared/glossary/#backward-pass) that propagates [gradients](/shared/glossary/#gradients) from the scalar output back to every input. It is the efficient choice when a model has many parameters but only one [loss value](/shared/glossary/#loss-value).

### Reward function {#reward-function}
The part of an [MDP](/shared/glossary/#mdp) that scores what happens, `R(s, a)`: the immediate number the agent receives for taking action `a` in state `s` (sometimes also depending on the next state). It defines *what the agent is trying to achieve* — the whole goal of RL is to collect as much total reward as possible over time, weighted by the [discount factor](/shared/glossary/#discount-factor). Designing it is deceptively hard: an agent optimizes the reward you *wrote*, not the behavior you *meant*, which is the source of [reward hacking](/shared/glossary/#reward-hacking). Example: in a [gridworld](/shared/glossary/#gridworld) you might give +1 for reaching the goal, −1 for stepping in a hazard, and a small −0.01 each step to encourage short paths.

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
Rotary Position [Embedding](/shared/glossary/#embedding) — a way to tell a [transformer](/shared/glossary/#transformer) *where* each token sits by physically rotating its query and key vectors by an angle proportional to the position, so the [attention](/shared/glossary/#attention) [dot product](/shared/glossary/#dot-product) between two tokens depends only on how far apart they are. Because the encoding lives in the rotation rather than an added vector, it [extrapolates](/shared/glossary/#extrapolation) to longer sequences than the model trained on. **2D RoPE** extends the trick to images: a [patch](/shared/glossary/#patch) token is rotated by its row *and* its column, encoding 2D spatial position. **3D RoPE** adds a third axis — time — so a video token is rotated by its row, column, *and* frame index; this is the standard position encoding in [DiT](/shared/glossary/#dit)-based video models, and because the rotation extrapolates, it is what lets a model trained on short clips generate longer ones at [variable resolution](/shared/glossary/#variable-resolution). Like giving every seat in a theater a precise angle on a [dial](/shared/glossary/#dial), so the model can always work out the spacing between any two seats — and, for 3D RoPE, every seat across every showtime.

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

### SARSA {#sarsa}
An [on-policy](/shared/glossary/#on-policy) [temporal-difference](/shared/glossary/#temporal-difference-learning) control algorithm, named for the five pieces of experience it uses in one update — **S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction `(s, a, r, s′, a′)`. It nudges `Q(s, a)` toward `r + γ Q(s′, a′)`, where `a′` is the action the agent *actually* takes next under its current (usually [ε-greedy](/shared/glossary/#epsilon-greedy)) policy. That one choice — using the real next action rather than the [greedy](/shared/glossary/#greedy-policy) one — is what separates it from [Q-learning](/shared/glossary/#q-learning) and makes it on-policy: because it accounts for its own exploration, it learns safer, more conservative behavior in risky environments like [Cliff Walking](/shared/glossary/#cliff-walking).

### Scale-and-shift {#scale-and-shift}
A two-step tweak applied to a layer's [activations](/shared/glossary/#activations): multiply every value by a learned *scale* and then add a learned *shift* — the operation `y = scale × x + shift`. It is exactly like the brightness and contrast sliders on a photo editor: scale stretches or squashes the range (contrast), and shift nudges everything up or down (brightness). The two numbers are usually the [weights](/shared/glossary/#weights) and [biases](/shared/glossary/#biases) a [normalization](/shared/glossary/#normalization) layer learns; when they are instead predicted from a condition such as a class label, you get conditioning schemes like [AdaGN](/shared/glossary/#adagn-adaptive-group-normalization), [AdaIN](/shared/glossary/#adaptive-instance-normalization-adain), and [AdaLN](/shared/glossary/#adaln).

### Scaling laws {#scaling-laws}
The empirical finding that a model's [loss](/shared/glossary/#loss-function) drops in a smooth, predictable curve as you add [parameters](/shared/glossary/#parameters), training data, and compute — like a growth chart that lets you forecast a bigger model's quality from smaller ones before you ever build it.

### Scene detection {#scene-detection}
Automatically finding the "cuts" in a video — the hard jumps where the footage switches from one shot to another — so a long video can be split into clean single-shot clips. It works by watching for a sudden, large change between two adjacent frames, measured by something like the difference in their color histograms (a tally of how many pixels fall into each color bucket) or in deep features. Analogy: flipping through a photo album and starting a new pile every time the picture suddenly looks completely different. Example: a 90-minute movie might be split into roughly 1,500 single-shot clips, each safe to use as a training example because the motion inside it is continuous rather than spanning an editing splice.

### Scheduler {#scheduler}
The part of an inference server that decides, at every step, which requests to start, which to keep generating, and which to pause when memory runs low — like an air-traffic controller choosing which planes take off, keep flying, or circle, so the runway (the GPU) is always busy but never overloaded. A good scheduler is often worth more real-world [throughput](/shared/glossary/#throughput) than any single clever kernel.

### Score {#score}
The [gradient](/shared/glossary/#gradients) of the log-probability of the data with respect to the input, written `∇_x log p(x)`. It points in the direction that makes an image *more likely* under the data distribution — in plain terms, "which way should I nudge these pixels to make this look more like a real image?" Diffusion models implicitly learn this at every noise level, so generation becomes a matter of repeatedly stepping in the score's direction, from noise toward a realistic sample.

### Score matching {#score-matching}
A way to train a generative model by teaching it the [score](/shared/glossary/#score) — the gradient of log-density, "which way makes this more likely" — instead of the density itself, which avoids ever computing an intractable normalizing constant. The practical version, *denoising score matching*, sidesteps needing the true score: add a known amount of Gaussian noise to each training example and have the network predict the direction back to the clean point, which provably equals the score of the noised data. (A relative, *sliced* score matching, estimates it instead by checking random one-dimensional projections.) Once the score is learned, you generate by following it with [Langevin dynamics](/shared/glossary/#langevin-dynamics). This is the lens that reveals [diffusion models](/shared/glossary/#diffusion-model) as score estimators trained at many noise levels.

### Scratchpad {#scratchpad}
A temporary, fast-access workspace where intermediate results are stashed so they don't have to be recomputed later. Like a math student's scratch paper next to an exam: jot the partial sums, look them up later, move on much faster than redoing each calculation. In serving, the [KV cache](/shared/glossary/#kv-cache) is the model's scratchpad — every key and value it has already computed sits there ready to be reused on the next decode step.

### SD3 {#sd3}
Stable Diffusion 3 — the 2024 release of [Stable Diffusion](/shared/glossary/#stable-diffusion) from Stability AI that switched the architecture to an [MMDiT](/shared/glossary/#mmdit) [transformer](/shared/glossary/#transformer) and trained it with [rectified flow](/shared/glossary/#rectified-flow) instead of the older U-Net-plus-[DDPM](/shared/glossary/#ddpm) recipe of earlier versions. Letting text and image tokens share the same [attention](/shared/glossary/#attention) layers, and feeding prompts through both [CLIP](/shared/glossary/#clip) and a large [T5](/shared/glossary/#t5) text encoder, gave it noticeably better prompt-following and spelling than SD1.x/SDXL. Think of it as the bridge release that moved Stable Diffusion from the U-Net era into the modern transformer-and-flow era that [Flux](/shared/glossary/#flux) then built on.

### SDE (stochastic differential equation) {#sde-stochastic-differential-equation}
An equation describing how something evolves over time under both a predictable push (the "drift") and continuous random jitter (the "diffusion") — like the path of a pollen grain carried by a current while being constantly buffeted by water molecules. A [diffusion model](/shared/glossary/#diffusion-model) can be written as an SDE that gradually turns an image into noise; reversing that SDE turns noise back into an image. The reverse SDE has a deterministic twin with identical statistics, the [probability flow ODE](/shared/glossary/#probability-flow-ode), and the two standard noising conventions are the [VP and VE SDE](/shared/glossary/#vp--ve-sde) families.

### SDF {#sdf}
Signed Distance Field — scalar field giving distance to nearest obstacle (negative inside)

### SDXL Turbo {#sdxl-turbo}
A speed-tuned version of Stable Diffusion XL that produces a usable image in a single step (or just a few), instead of the usual 20–50. It was created with [Adversarial Diffusion Distillation (ADD)](/shared/glossary/#add-adversarial-diffusion-distillation), which trains a fast "student" model under the eye of a [GAN](/shared/glossary/#gans)-style [discriminator](/shared/glossary/#discriminator) that rejects any quick output which doesn't look real — keeping the picture sharp despite the shortcut. Like a chef who learns to plate a dish in seconds because a tough critic tastes every rushed attempt. The trade-off: near-instant generation, with slightly less fine detail and variety than the slow original.

### SE(3) / SO(3) {#se3--so3}
Special Euclidean / Orthogonal group — rigid-body motions / rotations in 3D

### Seed {#seed}
A fixed starting number for a random-number generator; setting the same seed makes random operations (shuffling, initialization, dropout) produce the identical sequence every run.

### Segmentation map {#segmentation-map}
A picture that has been divided up so that every pixel is painted a flat color standing for *what kind of thing* it belongs to — all the "sky" pixels one color, all the "road" pixels another, all the "person" pixels a third. It is like a color-by-numbers outline of a scene: it throws away the photographic detail and keeps only a labeled map of which region is which. (Splitting an image into these labeled regions is called *segmentation*; the result is the segmentation map, sometimes a *segmentation mask*.) [ControlNet](/shared/glossary/#controlnet) can take one as a conditioning signal so a generated image places each kind of object exactly where its colored region sits — the prompt decides what a "building" looks like, but the map decides where the building goes.

### Sentence embedding {#sentence-embedding}
A single dense vector that captures the meaning of an entire sentence (or short passage), so two sentences about the same topic end up close together in vector space even if they use completely different words. Think of it as a GPS coordinate for meaning — two sentences that "mean the same thing" land near the same point on the map. Sentence [embeddings](/shared/glossary/#embedding) are the backbone of semantic search in [RAG](/shared/glossary/#rag): you embed the user's question and every stored passage, then find the passages whose coordinates are closest.

### Self-consistency {#self-consistency}
Sampling many independent [chain-of-thought](/shared/glossary/#cot) solutions to the same problem and taking a majority vote on the final answer — like asking several people to solve a puzzle on their own and trusting the answer most of them land on.

### Self-distillation {#self-distillation}
A twist on [distillation](/shared/glossary/#distillation) where the "teacher" and the "student" are the *same* model instead of a big teacher and a smaller student — the network learns by trying to match its *own* output on a slightly different view of the same input. Like checking your work by solving a problem a second way and forcing the two answers to agree: there is no answer key and no smarter tutor, so the network teaches itself just by staying consistent. Concrete example: in [DINOv2](/shared/glossary/#dinov2), a "teacher" copy (which is just a slowly-updated running average of the "student") looks at one crop of a photo while the student looks at a *different* crop, and the student is trained to reproduce the description the teacher gave — so the model learns [features](/shared/glossary/#embedding) that stay the same when an object is moved or cropped, all with no human labels. Because the teacher is only a smoothed copy of the student, this is a form of [self-supervised](/shared/glossary/#self-supervised) learning, and the slow averaging is what stops the network from cheating by collapsing to one constant answer for every image.

### Self-Forcing {#self-forcing}
A training method for [autoregressive](/shared/glossary/#autoregressive-model) video models that closes the gap between how they are trained and how they actually run. Normally such a model is trained to predict the next frame given *real* past frames, but at generation time it must feed on its *own* earlier outputs, whose small errors compound into drift — a mismatch called *exposure bias*. Self-Forcing fixes this by making the model generate from its own predictions *during training* too ("forcing" it to rely on itself), so it learns to cope with its own mistakes. Combined with [distillation](/shared/glossary/#distillation), it is a current recipe (alongside [CausVid](/shared/glossary/#causvid)) for real-time, long, [streaming](/shared/glossary/#streaming-video-generation) video.

### Self-supervised {#self-supervised}
Learning from raw, unlabeled data by inventing the labels from the data itself — for example hiding part of an input and asking the model to predict the missing piece, or asking whether two altered views came from the same original. No human annotation is needed, so the model can train on billions of images or sentences nobody had to tag. Like learning a language by covering up words in books you already own and guessing them, instead of paying a tutor to quiz you. This is how [DINOv2](/shared/glossary/#dinov2) learns vision features and how the masked- and next-token objectives behind most [LLMs](/shared/glossary/#llm) work; contrast it with supervised training, which needs an answer key.

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

### Shot list {#shot-list}
A structured plan that breaks a story into an ordered sequence of individual shots, each with its own short description of what happens in it — exactly the storyboard a film director writes before shooting. In video generation a [large language model](/shared/glossary/#llm) can act as the "director," expanding a one-line prompt into such a list (often as JSON), after which each shot is generated separately and the clips are stitched together. This is the planning step of [hierarchical generation](/shared/glossary/#hierarchical-generation): deciding *what happens in what order* before any frames are made is what lets a long video follow a sensible narrative instead of drifting aimlessly. Systems like VideoTetris and MovieDreamer build on this idea.

### SigLIP {#siglip}
[Sigmoid-loss](/shared/glossary/#sigmoid-loss) CLIP — a [CLIP](/shared/glossary/#clip) variant that swaps CLIP's batch-wide [softmax](/shared/glossary/#softmax) [contrastive loss](/shared/glossary/#infonce) for a [sigmoid loss](/shared/glossary/#sigmoid-loss), which scores each image–text pair on its own as an independent yes/no match. Judging pairs one at a time means it trains well even with small [batches](/shared/glossary/#batch), where CLIP needs very large ones to gather enough negatives to compare against. SigLIP 2 (2025) extends it with better data and multilingual training.

### Sigmoid loss {#sigmoid-loss}
A training [loss](/shared/glossary/#loss-function) that scores each example with one simple, independent yes/no question — "should these two things match?" — instead of making examples compete against each other. It runs the model's raw match score through the *sigmoid* function, an S-shaped curve that squashes any number into a probability between 0 and 1 (very negative → near 0, very positive → near 1, zero → 0.5), then rewards the model when a true pair lands near 1 and a mismatched pair lands near 0. Like grading each true/false exam question on its own merits, instead of ranking every student in the room against one another — which is what [softmax](/shared/glossary/#softmax)-based losses such as CLIP's [InfoNCE](/shared/glossary/#infonce) do. **How it is computed:** for each pair take the label y (1 if they truly match, else 0) and the predicted probability p = sigmoid(score), then add up the [cross-entropy](/shared/glossary/#cross-entropy) of that single decision, `−[y·log p + (1−y)·log(1−p)]`, independently over every pair. Because each pair is judged alone rather than against a whole [batch](/shared/glossary/#batch), training still works with small batches, unlike softmax losses that need many examples per batch to compare against. This is the loss behind [SigLIP](/shared/glossary/#siglip).

### SiLU {#silu}
Sigmoid Linear Unit — just another name for [Swish](/shared/glossary/#swish), the activation `x · σ(x)`. The two words mean the exact same function: you will see "SiLU" in code (PyTorch's `nn.SiLU`) and "Swish" in papers.

### SIMT {#simt}
Single Instruction Multiple Threads; NVIDIA's execution model

### Skip-and-add logic {#skip-and-add-logic}
A design pattern in neural networks where a signal bypasses a layer unchanged and is then added back to the layer's output. Think of it like a chef tasting a soup that already has a good base flavor (the "skip" part, where the main base is kept), and deciding to just stir in a pinch of salt (the "add" part) to improve it, rather than throwing the soup out and cooking a new one from scratch. Because the main signal flows straight through, the layer only has to figure out the small correction (the residual) needed to make it better. This keeps information flowing easily in very deep networks.

### SLAM {#slam}
Simultaneous Localization and Mapping

### SLI {#sli}
Service Level Indicator — the actual measured number for how well a service is doing, such as the real percentage of requests that succeeded or answered within 500 ms. The [SLO](/shared/glossary/#slo) is the *target*; the SLI is the *measurement* you compare against it — like the speedometer reading (SLI) versus the posted speed limit (SLO).

### Sliding-window generation {#sliding-window-generation}
A way to make a video longer than a model's trained clip length without retraining it: generate a series of short clips that *overlap* by a few frames, then blend the overlapping frames so the joins are seamless. Sliding a fixed-size window forward a little at a time is where the name comes from. Its weakness is long-range coherence — the only information tying distant parts together is whatever the small overlap can carry forward, so the scene slowly *drifts*, with colors and details creeping away from the opening. It is the cheapest long-form trick because it wraps any existing [text-to-video](/shared/glossary/#t2v) model; **FreeNoise** and **Gen-L-Video** are named methods that refine how the overlaps are blended (for example, reusing a shared noise pattern so neighboring windows stay consistent).

### SLO {#slo}
Service Level Objective — a specific, measurable promise about how a service should perform, such as "p95 [TTFT](/shared/glossary/#ttft) under 500 ms" or "99.9% of requests succeed." It is the target you design toward and get alerted on, like a delivery company promising most parcels arrive within two days. The measured reality you check it against is the [SLI](/shared/glossary/#sli), and the slack it allows for failure is the [error budget](/shared/glossary/#error-budget).

### SM {#sm}
Streaming Multiprocessor; the GPU's "core"

### Soft gate {#soft-gate}
A multiplier that scales each value by some amount *between* fully off (0) and fully on (1), instead of the hard either/or of a switch that is only ever 0 or 1. Picture a dimmer knob rather than a light switch: a hard gate can only block a signal or let it through untouched, but a soft gate can pass 0.3 of it, or 0.8, dialing each value partly up or down. In a [SwiGLU](/shared/glossary/#swiglu) layer the gate amounts come from squeezing one projection of the input through a smooth [activation function](/shared/glossary/#activations) like [Swish](/shared/glossary/#swish), whose output slides continuously rather than snapping between two settings — and that smoothness is what lets the network learn the right gate values from clean [gradients](/shared/glossary/#gradients).

### softmax {#softmax}
The function that turns a vector of scores into a probability distribution — each value squeezed into 0–1, and all of them summing to 1; the core of [attention](/shared/glossary/#attention) and classification heads. The name means a *soft* version of `max`: instead of the hard "winner takes all" of [argmax](/shared/glossary/#argmax), which hands the single biggest score 100% and the rest nothing, softmax gives most of the weight to the biggest score while still leaving a little for the others. That smoothness — a dimmer switch rather than an on/off toggle — is what lets the model be trained by gradients.

### Sora {#sora}
OpenAI's [text-to-video (T2V)](/shared/glossary/#t2v) model (2024), the release that made "[DiT](/shared/glossary/#dit) over [3D VAE](/shared/glossary/#3d-vae) latents, trained with [flow matching](/shared/glossary/#flow-matching)" the default blueprint for a frontier video generator. Its two headline ideas were treating video as a long sequence of [spatiotemporal patches](/shared/glossary/#spatiotemporal-patches) (so the same model handles images and clips of different lengths) and [variable resolution](/shared/glossary/#variable-resolution) (generating many sizes and aspect ratios from one model). Sora 2 (2025) added synchronized audio and stronger physical consistency. OpenAI never released weights, so open replicas like [OpenSora](/shared/glossary/#opensora) reconstruct the recipe from its technical report. Think of Sora as the proof-of-concept that reset everyone's expectations for what a single video model could do.

### Spatiotemporal attention {#spatiotemporal-attention}
An [attention](/shared/glossary/#attention) pattern for video in which every token attends to every other token across *both* space and time at once — all positions in all frames mixed in a single shared attention operation. This is the most expressive way to model motion, because it can directly relate any pixel in any frame to any other, but its cost grows quadratically with the total number of tokens `T×H×W`, so it becomes very expensive as clips get longer or larger. Contrast [(2+1)D](/shared/glossary/#21d), which splits spatial and temporal attention into two cheaper separate steps, and [windowed attention](/shared/glossary/#windowed-attention), which restricts the joint attention to small local 3D windows. Sora-class models can afford full spatiotemporal attention only because a [3D VAE](/shared/glossary/#3d-vae) first shrinks `T×H×W` aggressively before attention ever runs — moving the expense out of attention and into the compressor.

### Spatiotemporal patches {#spatiotemporal-patches}
The 3D version of image [patches](/shared/glossary/#patch): instead of cutting one frame into flat 2D squares, a video is cut into little boxes that span a small image region *and* a few consecutive frames, so each box (also called a *tubelet*) captures appearance and motion at once. Each box becomes one [token](/shared/glossary/#token-visualaudio) for a [transformer](/shared/glossary/#transformer), so movement is baked into the input from the start rather than reconstructed later from separate frames; TubeViT is a model built this way. Like cutting a flip-book into small columns that each go *down through* several pages — one cut shows how that corner of the picture changes over time. Example: a 16-frame clip cut into 2×16×16 patches (2 frames deep, 16×16 pixels wide) becomes a sequence of motion-aware tokens. The trade-off is that 3D boxes multiply the token count fast, raising compute — contrast plain [patchification](/shared/glossary/#patchification), which slices a single still image.

### Special tokens {#special-tokens}
Reserved [vocabulary](/shared/glossary/#vocabulary) entries that mark structure rather than text — e.g. `<bos>`, `<eos>`, `<pad>`, and chat-boundary tokens like `<|im_start|>`

### Speculative decoding {#speculative-decoding}
A trick to make [decode](/shared/glossary/#decode) faster for free: a small, fast "draft" model guesses the next few tokens, and the big "target" model checks all of them in a single parallel pass, keeping every guess that matches what it would have produced and discarding the rest. Like an editor who reads a sentence a junior writer drafted and approves the part that is already correct rather than writing every word from scratch — the answer is identical to what the target alone would say, just reached in fewer slow steps. It works because decode is starved for memory bandwidth, so the GPU has spare compute to verify several guesses at once.

### Stable Diffusion {#stable-diffusion}
The best-known open-source [diffusion model](/shared/glossary/#diffusion-model) for turning a text prompt into an image (first released by Stability AI in 2022). Its key trick is to do the slow denoising work in a small compressed space (the [latent](/shared/glossary/#ldm) space of a [VAE](/shared/glossary/#vae)) rather than on full-size pixels — like sketching a scene as a rough thumbnail first and only blowing it up to full resolution at the very end — which makes it light enough to run on a single consumer GPU. Because the weights were released publicly, it sparked a huge ecosystem of fine-tunes and add-ons such as [LoRA](/shared/glossary/#lora), [ControlNet](/shared/glossary/#controlnet), and [DreamBooth](/shared/glossary/#dreambooth).

### Stable Video Diffusion (SVD) {#stable-video-diffusion-svd}
Stability AI's open-weights [image-to-video (I2V)](/shared/glossary/#i2v) model (2023), the canonical baseline for turning a single still image into a short clip. It is built by [temporal inflation](/shared/glossary/#temporal-inflation): it [freezes](/shared/glossary/#frozen) a pretrained [Stable Diffusion](/shared/glossary/#stable-diffusion) image model and adds new time-aware layers that learn motion, so it keeps Stable Diffusion's strong sense of appearance and only has to learn how things move. Released in two variants — one tuned to generate 14 frames, one for 25 — it conditions on the input image (not text), which makes it the easiest strong model to run for hands-on I2V experiments. It also exposes a [motion score](/shared/glossary/#motion-score) input to control how much movement the clip contains.

### State dict {#state-dict}
A Python `OrderedDict` that maps every parameter and buffer name to its tensor value; the standard format for saving, loading, and transplanting PyTorch model weights

### Static quantization (PTQ) {#static-quantization-ptq}
A [quantization](/shared/glossary/#quantization) method that converts both [weights](/shared/glossary/#weights) and [activations](/shared/glossary/#activations) to [int8](/shared/glossary/#int8) before serving, using a [calibration](/shared/glossary/#calibration) pass to fix the activation scales in advance.

### STFT {#stft}
Short-Time Fourier Transform — a way to find *which frequencies are present and when* in a signal by chopping it into many short, overlapping windows (say 25 ms each) and running a [Fourier transform](/shared/glossary/#fourier-transform) on each one separately. A plain [Fourier transform](/shared/glossary/#fourier-transform) tells you the frequencies in a whole clip but loses all sense of *when* they happened; the STFT trades a little frequency precision for time precision by asking the question over and over on tiny slices. The output is a grid of (time × frequency) magnitudes — the raw material a [mel spectrogram](/shared/glossary/#mel-spectrogram) then refines. Like tapping out a song's rhythm window by window instead of blending the whole piece into one average chord.

### Stop-string {#stop-string}
A user-supplied substring that tells the server "as soon as the generated text contains this, stop." Matched on the *decoded* text, not the raw token IDs, because the same letters can land in different [BPE](/shared/glossary/#bpe) tokens depending on what came before — so the matcher has to keep a small rolling window of recent output and check for the string at every step.

### Storage {#storage}
The 1-D buffer that a tensor is a view into

### Straight-through estimator {#straight-through-estimator}
A trick for training through a step that has no usable gradient — such as the nearest-[codebook](/shared/glossary/#codebook)-entry lookup in a [VQ-VAE](/shared/glossary/#vq-vae). On the [forward pass](/shared/glossary/#forward-pass) the hard, non-differentiable operation runs as usual; on the [backward pass](/shared/glossary/#backward-pass) the model simply *pretends* that step was the identity and passes the gradient straight through unchanged. It is like sketching along a ruler and then erasing the ruler's marks: the rough step shapes the result, but learning flows as if it were never there.

### Streaming {#streaming}
Sending the model's reply to the client one piece at a time as it is generated, instead of waiting for the whole answer and then returning it in a single response. Over HTTP this is usually done with Server-Sent Events (SSE) or chunked transfer encoding; the connection stays open and the server flushes each new token as soon as it is sampled. Like a waiter who brings each course out as it leaves the kitchen rather than holding the whole meal until dessert is ready — the user sees [TTFT](/shared/glossary/#ttft) drop dramatically even though total generation time is the same.

### Streaming video generation {#streaming-video-generation}
Producing a video frame-by-frame (or chunk-by-chunk) and emitting each piece *as soon as it is ready*, conditioning every new chunk on the ones already made — instead of computing the whole clip before showing anything. This is what makes real-time and open-ended (potentially infinite) video possible: you see frames immediately and generation can run as long as you keep asking. It typically reuses a [KV cache](/shared/glossary/#kv-cache) so each new chunk does not recompute the [attention](/shared/glossary/#attention) over all earlier frames, and pairs with [distillation](/shared/glossary/#distillation) into few-step models for speed. [CausVid](/shared/glossary/#causvid), [Self-Forcing](/shared/glossary/#self-forcing), and StreamingT2V are examples. It is the video analog of how a chatbot [streams](/shared/glossary/#streaming) words out one at a time rather than waiting for the full answer.

### Stride {#stride}
The number of storage elements to step over for each dimension of a tensor

### StyleGAN {#stylegan}
A family of [GANs](/shared/glossary/#gans) (StyleGAN, [StyleGAN2](/shared/glossary/#stylegan2), [StyleGAN3](/shared/glossary/#stylegan3)) famous for photorealistic faces — the models behind sites like `thispersondoesnotexist.com`. Instead of forcing random noise directly into a rigid spherical shape (which tangles attributes together), it first passes the noise through a mapping network to "iron out" the warped space into an intermediate [W latent space](/shared/glossary/#w-and-w-latent-spaces). It then injects this unwarped style code into every generation layer through [adaptive instance normalization](/shared/glossary/#adaptive-instance-normalization-adain). This design "disentangles" the latent space, so moving in one direction smoothly changes a single attribute (hair, age, lighting) while leaving the rest completely untouched.

### StyleGAN2 {#stylegan2}
An improved version of [StyleGAN](/shared/glossary/#stylegan) that fixes visual artifacts like waterdroplet-like blobs. It does this by redesigning how the [adaptive instance normalization (AdaIN)](/shared/glossary/#adaptive-instance-normalization-adain) is applied, moving it outside the convolutions. Think of it as upgrading from a good camera that sometimes leaves dust spots on the lens to a professional one that takes perfectly clean photos every time.

### StyleGAN3 {#stylegan3}
The third generation of the [StyleGAN](/shared/glossary/#stylegan) family, which focuses on fixing "texture sticking" — a problem where textures like hair or wrinkles would stay glued to the screen coordinates even as the face moved. It achieved this by making the entire network "alias-free," ensuring that when the underlying features move, the generated pixels move perfectly with them, like a seamless video rather than a sequence of loosely connected frames.

### Super-resolution {#super-resolution}
Turning a low-resolution image or video into a higher-resolution one by *inventing* the missing fine detail — not merely stretching the pixels (which only blurs them) but hallucinating plausible texture and sharp edges that were never in the small version. A diffusion-based super-resolution model is trained by taking sharp images, shrinking them, and learning to reconstruct the originals while conditioned on the small input. Like an artist handed a thumbnail and asked to repaint it at poster size, filling in detail consistent with what the thumbnail implies. It is the upscaling stage in a [cascaded diffusion](/shared/glossary/#cascaded-diffusion) pipeline, and "super" simply means resolving detail finer than the input's resolution seemed to allow.

### SWE (Software Engineering) {#swe}
Short for **Software Engineering** — the discipline of building, testing, and maintaining software systems. In the AI/LLM context, "SWE" usually appears in compound terms like [SWE-bench](/shared/glossary/#swe-bench) or "SWE-style agent," meaning an [agent](/shared/glossary/#agent) that does the kind of work a human software engineer does: reading code, diagnosing bugs, writing fixes, and running tests.

### SWE-bench {#swe-bench}
Short for **Software Engineering Benchmark** — a benchmark of real GitHub issues paired with the code changes that fixed them; an [agent](/shared/glossary/#agent) is judged by whether its edits make the project's test suite pass, which makes it the standard test of coding agents.

### Sweep {#sweep}
Training the same model many times while changing one setting across a range of values, then comparing results to pick the best — for example trying ten different [learning rates](/shared/glossary/#learning-rate) and keeping the winner. Like tasting a sauce as you add salt in small steps to find the amount you like, rather than guessing the whole spoonful at once. A sweep is how you turn a hyperparameter hunch into a measured choice.

### SwiGLU {#swiglu}
The activation used in most modern transformer [MLPs](/shared/glossary/#mlp): a [GLU](/shared/glossary/#glu) gate whose non-linearity is [Swish](/shared/glossary/#swish), written `(xW) · Swish(xV)`. In plain terms, the input is projected two ways — one path is the content, the other is squeezed through Swish to become a [soft gate](/shared/glossary/#soft-gate) — and the two are multiplied so the gate dials each value up or down. It replaced plain [ReLU](/shared/glossary/#relu) feed-forward layers because, for the same size, it tends to learn a little better; it is the default [FFN](/shared/glossary/#ffn) in Llama-style models.

### Swish {#swish}
A smooth [activation function](/shared/glossary/#activations), `x · σ(x)`, also called [SiLU](/shared/glossary/#silu). It does roughly the same job as [ReLU](/shared/glossary/#relu) — squashing large negatives toward 0 and passing positives through — but with a gentle curve instead of a sharp corner (and it even dips a little below 0 for small negatives before recovering). Think of a soft-closing drawer that eases shut instead of slamming at exactly zero: that smoothness gives the network cleaner [gradients](/shared/glossary/#gradients) to learn from. It is the non-linearity used as the gate inside [SwiGLU](/shared/glossary/#swiglu).

### Synthetic captions {#synthetic-captions}
Replacing an image's original web [alt-text](/shared/glossary/#alt-text) — which is often missing, keyword-spammed, or unrelated to the picture — with a fresh, detailed caption written by a [VLM](/shared/glossary/#vlm) that actually looks at the image and describes it ("a golden retriever catching a frisbee on a beach at sunset"). Also called *recaptioning*. Training a [text-to-image](/shared/glossary/#stable-diffusion) model on these cleaner descriptions dramatically improves how faithfully it follows prompts — it is the single biggest reason DALL·E 3 became so good at composition. Like re-cataloguing a library where half the books were shelved under the wrong title: once every spine is relabeled to match its contents, readers (here, the model) finally learn which words map to which pictures. Example: an image whose alt-text was "IMG_2025.jpg" gets a full descriptive sentence before it is used for training.

### SynthID {#synthid}
Google DeepMind's [watermarking](/shared/glossary/#watermarking) tool that hides an invisible, detectable signal inside AI-generated content — images first, later audio, text, and video — so it can be identified as machine-made without any visible change to the picture. Rather than stamping the pixels afterward, it can weave the mark into the generation process itself, which helps it survive cropping, resizing, and JPEG compression. Like a secret ink woven into the paper of a banknote: you can't see it, but the right detector lights it up instantly. It is one practical answer to the safety problem of telling real photos from synthetic ones as AI images flood the web.

### System prompt {#system-prompt}
A message placed at the very start of a chat conversation that tells the model how to behave — its role, tone, rules, and the tools it can call — before the user's first turn ever arrives. Like a stage director's note to an actor before the curtain rises: *"You're a polite customer-support agent who answers only refund questions."* System prompts are usually long and shared across many requests, which is why caching their KV state (see [prefix cache](/shared/glossary/#prefix-cache)) saves so much repeated work.

### Systolic array {#systolic-array}
Data-flow matmul fabric used in TPUs

### T2V {#t2v}
Text-to-Video: generating a video clip from a text prompt alone, with no image to start from — the model must invent both *what the scene contains* and *how it moves*. This makes it harder than [image-to-video (I2V)](/shared/glossary/#i2v), where the first frame is given, and it needs paired text–video training data, which is scarce. Sora, Veo, and Kling are well-known T2V systems.

### T5 {#t5}
A text [transformer](/shared/glossary/#transformer) (Google's "Text-to-Text Transfer Transformer") that reads a sentence and produces rich embeddings of its meaning. Unlike [CLIP](/shared/glossary/#clip)'s text encoder, which was trained only to match images to short captions, T5 was trained on general language tasks, so it captures long, detailed prompts and word order more faithfully — which is why models like Imagen, SD3, and Flux feed it (often the large "T5-XXL" variant) into [cross-attention](/shared/glossary/#cross-attention) for better prompt adherence.

### Tail latency {#tail-latency}
The [latency](/shared/glossary/#latency) of the slowest requests (for example the p95 or p99 percentiles) rather than the median (p50); it is what users notice most.

### Talking head {#talking-head}
A model that takes a single portrait photo plus an audio clip and generates a video of that person *speaking* the audio — lips, jaw, and head moving in sync with the sound. The audio drives the motion while the photo fixes the identity, so the same face can be made to say anything. The hardest part is **lip sync**: the mouth has to form the right shape for each speech sound at the right instant, which is why these systems extract audio features (often with a speech encoder such as wav2vec) and align them to facial motion frame by frame, then enforce [temporal consistency](/shared/glossary/#temporal-consistency) so the face does not flicker or drift. Named systems include **EMO**, **Hallo**, **SadTalker**, and **V-Express**. Like a ventriloquist's puppet driven by a recording instead of a hand.

### Target model {#target-model}
In [speculative decoding](/shared/glossary/#speculative-decoding), the big, accurate model whose output you actually want — it checks the small [draft model](/shared/glossary/#draft-model)'s guesses and has the final say on every token. Like the senior editor who must approve the assistant's draft: slow and expensive to consult, so the trick is to bother it as rarely as possible while still letting it decide the real answer.

### TCP {#tcp}
Tool Center Point — the configurable point on a tool whose pose tracking controls

### TD error {#td-error}
`δ_t = r_t + γV(s_{t+1}) − V(s_t)` — the gap between a [bootstrapped](/shared/glossary/#bootstrapping) one-step estimate of a state's value and the value you currently hold for it. It is the core learning signal of [temporal-difference learning](/shared/glossary/#temporal-difference-learning): a positive `δ` means things turned out better than expected, so nudge `V(s_t)` up; a negative `δ` means worse, so nudge it down. Every TD method — [SARSA](/shared/glossary/#sarsa), [Q-learning](/shared/glossary/#q-learning), and TD(λ) with [eligibility traces](/shared/glossary/#eligibility-traces) — is some rule for which states to apply this error to.

### TD3 {#td3}
Twin Delayed DDPG — DDPG plus three stability fixes

### Temperature {#temperature}
A [sampling](/shared/glossary/#sampling) knob that scales the model's scores before [softmax](/shared/glossary/#softmax): low temperature (e.g. 0.2) sharpens the distribution so the model plays it safe and repeats the likeliest words, while high temperature (e.g. 1.5) flattens it so rarer, more surprising words can win. Think of it as a creativity dial — turn it down for factual answers, up for brainstorming. The same knob appears in [contrastive learning](/shared/glossary/#infonce) (written τ): there the similarity scores are *divided* by τ before the softmax, so a small τ (CLIP learns one starting around 0.07) sharpens the contest and forces the model to focus on its [hardest negatives](/shared/glossary/#hard-negatives), while a large τ softens it — too small destabilizes training, too large and even the true pair is barely preferred.

### Temporal attention {#temporal-attention}
[Attention](/shared/glossary/#attention) applied only along the time axis of a video: each spatial position (say, the pixel at row 10, column 20) looks at *that same position* across all the frames and decides how its value should change from frame to frame. It is the half of a [(2+1)D](/shared/glossary/#21d) block that handles motion, added on top of ordinary spatial attention (which works within each frame separately), and in [temporal inflation](/shared/glossary/#temporal-inflation) it is exactly the new layer dropped into a pretrained image model to teach it movement. Like tracking one fixed spot on a flip-book through every page to see how it animates, while ignoring the rest of each page. It is far cheaper than [spatiotemporal attention](/shared/glossary/#spatiotemporal-attention) because each position only compares itself across the `T` frames, not against every other position as well.

### Temporal consistency {#temporal-consistency}
The property of a generated or edited video where each object keeps a stable appearance, position, and identity *across frames* instead of changing slightly every frame. It is what separates real video from a flip-book of independently produced images: get it wrong and you see [temporal flicker](/shared/glossary/#temporal-flicker), crawling textures, or a character whose face morphs shot to shot. Because a model that processes frames one at a time has no built-in reason to keep them aligned, video methods deliberately share information across time — through [temporal attention](/shared/glossary/#temporal-attention), a [3D VAE](/shared/glossary/#3d-vae) that compresses several frames together, or by reusing features and noise between frames during editing. It is the single recurring obstacle in [video-to-video](/shared/glossary/#v2v), control, and long-form generation. Like a team of animators agreeing on exactly how a character looks so it does not subtly change every drawing.

### Temporal flicker {#temporal-flicker}
The shimmering, pulsing, or boiling look you get when a video is processed one frame at a time with no coordination across frames. Each frame is reconstructed (or generated) slightly differently from its neighbors — tiny independent errors in texture, color, or brightness — and because the real scene barely changed, your eye reads those frame-to-frame differences as unwanted motion. It is the classic failure of running an image [VAE](/shared/glossary/#vae) per frame, and the main reason video needs compressors and models that span the time axis (a [3D VAE](/shared/glossary/#3d-vae) or [temporal attention](/shared/glossary/#temporal-attention)) rather than treating a clip as a stack of unrelated stills.

### Temporal inflation {#temporal-inflation}
The dominant trick for making a video model out of an existing *image* model: take a pretrained 2D network, insert new layers that operate along the time axis (temporal convolutions or temporal [attention](/shared/glossary/#attention)), and usually initialize them as an *identity* (pass-through) so that, at the start of training, the inflated model behaves exactly like the original image model run frame by frame. You then [fine-tune](/shared/glossary/#fine-tuning) so the new layers gradually learn motion while the spatial layers keep everything they already knew about appearance. "Inflation" captures the picture of taking a flat 2D model and puffing it out into the third (time) dimension. [Stable Video Diffusion](/shared/glossary/#stable-video-diffusion-svd), [AnimateDiff](/shared/glossary/#animatediff), and Make-A-Video all use variants of this; the 2024+ frontier (Sora-class models) instead trains spatiotemporal models from scratch.

### Temporal-difference learning {#temporal-difference-learning}
The central learning idea of RL: update a [value](/shared/glossary/#value-function) estimate from a *single step* of experience by [bootstrapping](/shared/glossary/#bootstrapping) — using your own current estimate of the next state's value as a stand-in for the rest of the future. The one-step version, TD(0), moves `V(s)` toward the target `r + γ V(s′)`; the gap between that target and the old estimate is the [TD error](/shared/glossary/#td-error). Unlike a [Monte Carlo](/shared/glossary/#monte-carlo-method) update, TD does not wait for the episode to finish, so it learns online, step by step — trading a little [bias](/shared/glossary/#bias-variance-tradeoff) (the target leans on a still-imperfect estimate) for much lower variance. [Q-learning](/shared/glossary/#q-learning), [SARSA](/shared/glossary/#sarsa), and the [eligibility-trace](/shared/glossary/#eligibility-traces) method TD(λ) are all temporal-difference algorithms; the name refers to learning from the *difference between two estimates made at different times*.

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

### Text encoder {#text-encoder}
The part of a model that turns a piece of text into numbers — a list of [embeddings](/shared/glossary/#embedding) that capture what the words *mean* — so the rest of the system can work with language as math. Think of it as a translator that reads your prompt and rewrites it in the only language a neural network understands: vectors. In a text-to-image model the text encoder reads the prompt once and the [diffusion model](/shared/glossary/#diffusion-model) then keeps glancing at those vectors (via [cross-attention](/shared/glossary/#cross-attention)) to decide what to draw. It is one half of a two-part model like [CLIP](/shared/glossary/#clip) — the side that reads words, paired with an image encoder that reads pictures — but the term is more general: any model that maps text to embeddings (CLIP's text tower, [T5](/shared/glossary/#t5), a BERT-style encoder) is a text encoder, which is why it deserves its own name rather than being conflated with CLIP as a whole.

### Text rendering {#text-rendering}
The ability of an image generator to draw *legible, correctly-spelled* words inside the picture — a shop sign that actually reads "OPEN," not "OPNE" or wavy gibberish. It was for years the field's most visible failure, because a model trained only to match overall image statistics has no spelling checker: it learns the *shape* of letters but not that "the exact order of letters matters." Modern models ([Imagen 3](/shared/glossary/#imagen-3), [Flux](/shared/glossary/#flux), [Ideogram](/shared/glossary/#ideogram)) largely fixed it with dedicated training data and stronger [text encoders](/shared/glossary/#text-encoder). Like a painter who can flawlessly reproduce the *look* of handwriting in a language they cannot read — beautiful strokes, but misspelled words until they are taught the alphabet itself. Example test: prompt "a neon sign that says 'DIFFUSION'" and check whether all nine letters appear, in order, spelled right.

### Text-to-image {#text-to-image}
A model that turns a written prompt into a brand-new picture — you type "a corgi astronaut floating in space" and it paints one from scratch. Under the hood a [text encoder](/shared/glossary/#text-encoder) reads your words into numbers, and a generator (usually a [diffusion model](/shared/glossary/#diffusion-model)) uses them to decide what to draw, glancing back at the prompt the whole time via [cross-attention](/shared/glossary/#cross-attention). Like a sketch artist who never sees the scene and draws purely from your verbal description — the richer and clearer your words, the closer the result. Famous examples include [Stable Diffusion](/shared/glossary/#stable-diffusion), [DALL·E 3](/shared/glossary/#dalle-3), [Imagen 3](/shared/glossary/#imagen-3), and [Ideogram](/shared/glossary/#ideogram).

### Textual Inversion {#textual-inversion}
A personalization method that teaches a frozen [diffusion model](/shared/glossary/#diffusion-model) a new subject by learning a *single new word* for it — nothing in the model itself changes. Concretely it optimizes one fresh vector added to the [text encoder](/shared/glossary/#text-encoder)'s [embedding matrix](/shared/glossary/#embedding-matrix) (the lookup table of word [embeddings](/shared/glossary/#embedding)) so that this invented "word" makes the model draw your subject. Because only that one vector is trained, the saved file is a few kilobytes — the smallest personalization artifact there is — but its capacity is limited: one vector can pin down a recognizable look yet cannot match the fidelity of [LoRA](/shared/glossary/#lora) or [DreamBooth](/shared/glossary/#dreambooth), since the frozen model can only render what it already knows how to draw. Like adding one new entry to a shared dictionary: you define the word just once, and from then on that single word stands in for your whole subject — but, just like a dictionary, it can only ever be explained using words and ideas the model already understands.

### Throughput {#throughput}
How much work is completed per unit of time — for training, the number of examples processed per second.

### Tiling {#tiling}
Splitting a large computation into small blocks ("tiles") that fit in fast on-chip memory, so a [kernel](/shared/glossary/#kernel) reads slow memory fewer times.

### Token (visual/audio) {#token-visualaudio}
A discrete code that stands for a small piece of an image or sound, produced by a [VQ-VAE](/shared/glossary/#vq-vae) or [neural codec](/shared/glossary/#neural-codec). Just as a [tokenizer](/shared/glossary/#tokenizer) chops text into word-pieces, an image tokenizer turns a picture into a grid of these codes drawn from a fixed vocabulary — so a [transformer](/shared/glossary/#transformer) can model images (or audio) with the same machinery it uses for language.

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

### Transition function {#transition-function}
The part of an [MDP](/shared/glossary/#mdp) that describes how the world moves, `P(s' | s, a)`: the probability of landing in next-state `s'` given that you took action `a` in state `s`. It captures the environment's dynamics, including randomness — like a slippery floor where "go right" actually moves you right only 80% of the time. Also called the transition probabilities, the dynamics, or the model. When these probabilities are *known*, you can plan exactly with [value iteration](/shared/glossary/#value-iteration); when they are *unknown*, the agent must learn from sampled experience, which is what most of RL is about. Example: in a deterministic [gridworld](/shared/glossary/#gridworld), `P` simply maps "in cell 5, move up" to "now in cell 2" with probability 1.

### transpose {#transpose}
Swaps two dimensions by rewriting strides — never copies; the result is usually non-contiguous

### Tree-of-Thoughts {#tree-of-thoughts}
A reasoning method that explores several partial solutions at once as branches of a tree, scores them, and expands only the promising ones — like working through a maze by trying multiple paths and backing out of dead ends instead of committing to the first turn.

### Triage {#triage}
Sorting cases by what each one needs, borrowed from emergency-room medicine where a nurse classifies arriving patients by severity before any doctor sees them. In LLM evaluation, *hallucination triage* means sorting model answers into useful buckets — *correctly answered*, *correctly abstained ("I don't know")*, *confidently wrong* ([hallucination](/shared/glossary/#hallucination)) — so each rate can be measured separately, instead of collapsing everything into one "accuracy" number that hides which failures are dangerous.

### Triangular weights {#triangular-weights}
The triangle-shaped set of multipliers each filter in a mel [filterbank](/shared/glossary/#filterbank) uses to blend nearby [STFT](/shared/glossary/#stft) frequencies into one band. A filter's weight rises linearly from 0 up to 1 at its center frequency and falls back to 0 at its edges, so frequencies near the center count fully and those at the edges barely count — and neighboring triangles overlap so no frequency is dropped. Like a dimmer switch that is brightest in the middle of each band and fades to off at the borders, smoothly handing off to the next band. Example: a triangle centered at 500 Hz might weight 480 Hz at 0.8, 500 Hz at 1.0, and 520 Hz at 0.8, while 400 Hz and 600 Hz get 0; the band's value is the weighted sum of those frequencies' energies.

### Tripwire {#tripwire}
A cheap, fast check whose only job is to sound the alarm the instant something goes wrong — named after the thin wire that, when stepped on, sets off a trap or flare. In model deployment a quick metric like [perplexity](/shared/glossary/#perplexity) is used as a tripwire: it won't tell you *what* broke, but it spikes the moment quality drops, so it catches a bad build before the slower, fuller tests even run.

### Triton {#triton}
A Python-flavored language for writing GPU kernels, developed by OpenAI

### Triton Inference Server {#triton-inference-server}
NVIDIA's production server for hosting models behind an HTTP/gRPC API, with [batching](/shared/glossary/#batching) and multi-model support; unrelated to the [Triton](/shared/glossary/#triton) kernel language despite the shared name.

### TTFT {#ttft}
Time to *produce* the first token — the elapsed time from when a request arrives at the server until the model returns its first output token, dominated by [prefill](/shared/glossary/#prefill) plus any queue wait. Like a restaurant's "time until your drink arrives" — felt separately from the rest of the meal, and the first thing the user actually notices.

### U-Net {#u-net}
An encoder-decoder network whose name comes from its U shape: the left arm shrinks the image down to a small, abstract summary while the right arm builds it back up to full size, with *skip connections* that hand each down-sampling layer's detail straight across to its matching up-sampling layer. Those skips are what let it keep fine pixel detail while still reasoning about the whole image, which is why it became the standard backbone for [diffusion models](/shared/glossary/#diffusion-model) (before [DiT](/shared/glossary/#dit) brought in transformers). It was originally invented for [medical-image segmentation](/shared/glossary/#medical-image-segmentation).

### UCF-101 {#ucf-101}
A widely used action-recognition video dataset from the University of Central Florida of about 13,000 short YouTube clips spanning 101 human-action categories (playing guitar, applying makeup, bench-pressing, and so on); the "101" is simply the number of action classes. It is small and low-resolution by modern standards, which is exactly why it became a common testbed for early video generation — you can train on it without a data-center. It shows up constantly in pre-diffusion video-GAN papers as the dataset everyone reported numbers on.

### Underflow {#underflow}
Condition where a floating-point value is too small to be represented and rounds to zero; common with `float16` when accumulating very small gradients

### URDF / MJCF / USD {#urdf--mjcf--usd}
Robot description formats (ROS, MuJoCo, NVIDIA respectively)

### User turn {#user-turn}
One message a user sends in a chat conversation, paired with the model's reply (the *assistant turn*). A back-and-forth between user and assistant is a sequence of alternating turns, all under the same opening [system prompt](/shared/glossary/#system-prompt). In typical traffic, the system prompt is long and fixed while each user turn is short and varies — which is exactly the pattern a [prefix cache](/shared/glossary/#prefix-cache) exploits.

### V2V {#v2v}
Video-to-Video: transforming an existing video into a new one while keeping its motion and timing — for example restyling it into a cartoon, or re-rendering it conditioned on per-frame depth or pose. The hard part is *temporal consistency*: editing each frame independently makes the result flicker, so V2V methods share information across frames. Contrast with [image-to-video](/shared/glossary/#i2v) (one image in) and [text-to-video](/shared/glossary/#t2v) (text only).

### Vanishing gradients {#vanishing-gradients}
A problem during training where [gradients](/shared/glossary/#gradients) become extremely small, effectively preventing the weights from changing their value and stalling the learning process.

### VAE {#vae}
Variational Autoencoder — an [autoencoder](/shared/glossary/#autoencoder) whose encoder outputs not a single point but a small *cloud* of possibility (a mean and a spread) for each input, and whose decoder samples from that cloud to rebuild the image. Training on the [ELBO](/shared/glossary/#elbo) presses those clouds to fit neatly under one standard bell-curve shape, so afterwards you can draw a brand-new point from that shape and decode it into a fresh image the model has never seen. That sampling ability is what makes a VAE a *generative* model rather than just a compressor.

### Validation loss {#validation-loss}
The [loss](/shared/glossary/#loss-function) measured on held-out data the model was not trained on; the honest signal of how well training is generalizing.

### Value function {#value-function}
How much total future reward you should expect, used as a score for situations. The *return* it averages is the discounted sum of all rewards from now on (`r₀ + γr₁ + γ²r₂ + …`, each step shrunk by the [discount factor](/shared/glossary/#discount-factor) γ). Two flavors: the **state-value** `V(s)` is the expected return starting from state `s` and following your [policy](/shared/glossary/#policy); the **action-value** `Q(s, a)` is the expected return if you first take action `a` and follow the policy afterward. The difference `Q(s, a) − V(s)` is the [advantage](/shared/glossary/#advantage) — how much better that one action is than the policy's average. The [Bellman equation](/shared/glossary/#bellman-equation) is the recursive rule these values must satisfy.

### Value iteration {#value-iteration}
A [dynamic-programming](/shared/glossary/#dynamic-programming) algorithm for solving an [MDP](/shared/glossary/#mdp) when the [transition function](/shared/glossary/#transition-function) and [reward function](/shared/glossary/#reward-function) are known: start with any guess of the [value function](/shared/glossary/#value-function), then repeatedly apply the optimality [Bellman backup](/shared/glossary/#bellman-operator) — set each state's value to the *best* action's "reward now plus [discounted](/shared/glossary/#discount-factor) next-state value" — until the values stop changing. Because that backup is a [contraction mapping](/shared/glossary/#contraction-mapping), the values are guaranteed to converge to the optimum `V*`; reading off the best action in each state then gives the [optimal policy](/shared/glossary/#optimal-policy). It differs from [policy iteration](/shared/glossary/#policy-iteration) by never evaluating a policy to completion — it folds improvement into every sweep. Unlike policy iteration, which perfectly measures a policy's value before changing it, value iteration takes a tiny step of improvement after every single evaluation sweep. Think of finding the best route to work: policy iteration is like driving one specific route every day for a month to know its exact time before trying a new one, while value iteration is like driving a route once and immediately updating your guess for the best path.

### Value network {#value-network}
The helper network (the "critic") in some RL algorithms that estimates the [value function](/shared/glossary/#value-function) — its best guess of how much future reward a situation is worth — so the [policy](/shared/glossary/#policy) can tell whether an action turned out better or worse than expected. [PPO](/shared/glossary/#ppo) trains one alongside the policy, which roughly doubles the networks held in memory; [GRPO](/shared/glossary/#grpo) skips it entirely by comparing each sampled answer to the group's average instead, which is what makes it cheaper.

### Vanilla {#vanilla}
The plain, unmodified, baseline version of a model or algorithm — no special improvements or extra tricks, just the original idea as first described. Like ordering plain vanilla ice cream with no toppings: it is the default flavor before anyone adds anything extra. In machine learning, "vanilla VAE" means the original [VAE](/shared/glossary/#vae) from the 2013 Kingma & Welling paper, before later work added hierarchical latents, β controls, or other refinements. Comparing the vanilla version to improved variants is the clearest way to measure what each addition actually buys.

### Variable resolution {#variable-resolution}
The ability of one trained model to generate clips at many different sizes, durations, and aspect ratios, instead of being locked to the single resolution it trained on — the headline claim of [Sora](/shared/glossary/#sora). It is possible because a [DiT](/shared/glossary/#dit) processes a *sequence* of [spatiotemporal patches](/shared/glossary/#spatiotemporal-patches) rather than a fixed-size grid, so feeding it more or fewer tokens naturally yields a taller, wider, or longer video; [3D RoPE](/shared/glossary/#rope) makes this work because it encodes each token's position as a rotation that [extrapolates](/shared/glossary/#extrapolation) to lengths and shapes never seen in training, instead of a fixed lookup table that would have no entry for a new position. In practice the model is trained across several [aspect-ratio buckets](/shared/glossary/#aspect-ratio-bucketing) so it has seen a range of shapes. Like a printer that can lay the same document out on A4, letter, or a wide banner without re-typesetting it.

### VBench {#vbench}
A comprehensive [open](/shared/glossary/#open-model) benchmark suite for [text-to-video](/shared/glossary/#t2v) models that, instead of boiling quality down to one number, scores generated clips along many separate dimensions — subject consistency, motion smoothness, [aesthetic quality](/shared/glossary/#aesthetic-score), text–video alignment, and a dozen more — and reports each one. The idea is that "is this video good?" has several independent answers, so a single score hides whether a model is, say, beautiful but jittery; splitting the score apart tells you exactly what to fix. To produce each number it runs many generated clips through purpose-built detectors (for example, a tracker to check an object stays the same shape, an [optical-flow](/shared/glossary/#optical-flow) measure for smooth motion, a [CLIP](/shared/glossary/#clip)-style match for text alignment) and averages the results into a per-dimension percentage. It is the closest thing the field has to a standard video-generation [leaderboard](/shared/glossary/#leaderboard).

### Verifier {#verifier}
A program that automatically checks whether an answer is correct — running unit tests, or comparing to a known math result — giving the exact, unhackable reward that [RLVR](/shared/glossary/#rlvr) trains on.

### Very Deep VAE {#very-deep-vae}
A [hierarchical VAE](/shared/glossary/#hierarchical-vae) (Child, 2021) that scales to dozens of stacked latent variable groups — far more layers than earlier models. Each group only handles a thin slice of the work, with [residual-like parameterizations](/shared/glossary/#residual-parameterization) keeping gradients flowing through the depth. Like adding so many floors to a building that no single floor needs to bear much weight, it achieved strong image generation quality, showing that deeper hierarchies can capture richer structure than shallow ones.

### Video-CFG {#video-cfg}
Applying [classifier-free guidance (CFG)](/shared/glossary/#cfg-classifier-free-guidance) to a video model that has more than one condition — typically a text prompt *and* a conditioning image — by giving each condition its own guidance scale instead of one shared dial. You can then push text adherence and image faithfulness independently: strong text guidance to match the prompt, separate image guidance to stay locked to the conditioning frame. The catch unique to video is that turning either scale too high amplifies per-frame detail at the cost of smooth change *between* frames, so the clip's motion begins to flicker or its colors over-saturate — guidance strength trades against temporal smoothness. This is why production video models expose several guidance knobs rather than the single one image models use.

### Video codec {#video-codec}
The set of rules for compressing video into a small file and decompressing it back into frames — "codec" is short for **co**der–**dec**oder, which is literally what it does. Raw video is enormous (a few seconds can be hundreds of megabytes), so almost all real video is stored compressed; codecs exploit the fact that neighboring frames barely change. Analogy: a codec is like shorthand for a movie — instead of writing every frame in full, it writes "same as the last frame, but this corner moved." Examples include [H.264](/shared/glossary/#h264) (the universal default) and [AV1](/shared/glossary/#av1) (smaller files, slower to decode); the codec lives *inside* a [media container](/shared/glossary/#media-container) like `.mp4`.

### view {#view}
A no-copy alias that shares storage with its source; requires a contiguous-compatible layout

### VIO {#vio}
Visual-Inertial Odometry — fuse camera and IMU for high-rate ego-motion

### Video GAN {#video-gan}
A [GAN](/shared/glossary/#gans) adapted to produce short video clips instead of single images: the [generator](/shared/glossary/#generator) outputs a whole stack of frames at once and the [discriminator](/shared/glossary/#discriminator) judges whether the *motion*, not just each individual frame, looks real. The early family — VGAN, TGAN, [MoCoGAN](/shared/glossary/#mocogan), DVD-GAN, and StyleGAN-V — produced only short, low-resolution clips and suffered badly from [mode collapse](/shared/glossary/#mode-collapse) (the generator falling back on a few safe outputs). Each pushed one idea: MoCoGAN separated content from motion, DVD-GAN was the first to reach plausible quality, and StyleGAN-V applied [StyleGAN](/shared/glossary/#stylegan)'s latent-space tricks to video. The whole approach was largely abandoned around 2023 once [diffusion](/shared/glossary/#diffusion-model) proved both sharper and far more stable to train at scale.

### ViT {#vit}
Vision Transformer — a [transformer](/shared/glossary/#transformer) that classifies or encodes images by first chopping them into a grid of small square [patches](/shared/glossary/#patch) ([patchification](/shared/glossary/#patchification)), turning each patch into one token, and then treating the picture exactly like a sentence of words. Because a plain transformer has no built-in notion of "next to" the way a [CNN](/shared/glossary/#cnn) does, a ViT adds a learned *positional embedding* to each patch (a small vector that says "I am the patch at row 3, column 5") and usually prepends a [CLS token](/shared/glossary/#cls-token) whose output becomes the whole-image summary. Like reading a mosaic tile by tile, left to right, instead of taking in the whole wall at once — and, given enough data, this beats CNNs because the model can relate any tile to any other from the very first layer instead of only neighboring pixels. The "B/16" in a name like ViT-B/16 means a Base-size model with 16×16-pixel patches.

### VLA {#vla}
Vision-Language-Action model — transformer mapping image + instruction → action

### vLLM {#vllm}
The reference open-source inference engine with PagedAttention and continuous [batching](/shared/glossary/#batching)

### VLM {#vlm}
Vision-Language Model — a model that takes an image (usually plus a text question) in and produces text out, such as a caption or an answer. The standard build is [middle fusion](/shared/glossary/#fusion-earlymiddlelate): a pretrained image encoder turns the picture into feature vectors, a small *[projector](/shared/glossary/#projector)* maps those into the [token](/shared/glossary/#token-visualaudio) space of a pretrained language model, and the language model then "reads" the image alongside the words. [LLaVA](/shared/glossary/#llava) is the canonical open example; Qwen2-VL and Gemini are larger ones. Analogy: a sighted assistant describing a photo to a brilliant writer who cannot see it — the encoder does the looking, the language model does the talking. Unlike a [native multimodal](/shared/glossary/#native-multimodal) model, a plain VLM only *outputs* text; it cannot generate images.

### Vocabulary {#vocabulary}
The fixed set of tokens a [tokenizer](/shared/glossary/#tokenizer) can produce, each with an integer ID; its size trades tokens-per-document against [embedding matrix](/shared/glossary/#embedding-matrix) size

### Volta {#volta}
NVIDIA's 2017 GPU architecture (V100) and the first generation to ship [Tensor Cores](/shared/glossary/#tensor-core), the dedicated matmul units that made deep-learning training dramatically faster. Subsequent generations — Turing, Ampere, [Hopper](/shared/glossary/#hopper), [Blackwell](/shared/glossary/#blackwell) — kept Tensor Cores and added support for ever-lower-precision formats. Named after the Italian physicist Alessandro Volta.

### VP / VE SDE {#vp--ve-sde}
The two standard ways to define the forward noising process of a [diffusion model](/shared/glossary/#diffusion-model), each written as an [SDE](/shared/glossary/#sde-stochastic-differential-equation). Variance-Preserving (VP) — the family [DDPM](/shared/glossary/#ddpm) uses — shrinks the original signal as it adds noise so the total variance stays around 1 the whole way. Variance-Exploding (VE) — used by the early [score](/shared/glossary/#score)-based models — leaves the signal untouched and simply piles on ever-larger noise, so the variance grows without bound. They are mathematically interconvertible and reach similar quality, but differ in numerical conditioning and in which samplers behave well.

### VP9 {#vp9}
A royalty-free [video codec](/shared/glossary/#video-codec) built by Google as a free alternative to the patent-licensed [H.264](/shared/glossary/#h264). It compresses noticeably better than H.264 — smaller files at the same quality — and is the codec behind most YouTube streams and many `.webm` files, though it has since been largely overtaken by the newer, even-smaller [AV1](/shared/glossary/#av1). Analogy: VP9 is to H.264 what a tighter, license-free ZIP format is to an older paid one — it squeezes the video smaller with no license fee, at the cost of more work to decode it back into frames. Example: a clip saved as a VP9 `.webm` is usually a good bit smaller than the same clip as an [H.264](/shared/glossary/#h264) `.mp4`, but slower to unpack into frames during training.

### VQA (Visual Question Answering) {#vqa-visual-question-answering}
The task of answering a natural-language question about an image — "How many people are in this photo?", "What color is the car?" — where the model must read the picture *and* the words together to respond. It is the classic benchmark for [multimodal understanding](/shared/glossary/#modality): unlike captioning, which can lean on generic descriptions, a question pins the model to one specific detail it cannot fake. Think of an open-book exam where the "book" is a photograph and each question forces you to actually look. Most [VLMs](/shared/glossary/#vlm) are evaluated on VQA datasets, and it is the natural small task on which to compare fusion methods like [concatenation](/shared/glossary/#concatenation) versus [cross-attention](/shared/glossary/#cross-attention).

### VQ-GAN {#vq-gan}
A [VQ-VAE](/shared/glossary/#vq-vae) trained with two extra signals so its reconstructions look sharp instead of blurry: a [perceptual loss](/shared/glossary/#perceptual-loss-lpips) that compares images by their high-level features rather than exact pixels, and a patch discriminator — a small critic from the [GAN](/shared/glossary/#gans) world that scores whether each local region of an image looks real. The combination pushes the decoder to commit to crisp, specific details. This is the recipe [Stable Diffusion](/shared/glossary/#stable-diffusion)'s [VAE](/shared/glossary/#vae) descends from.

### VQ-VAE {#vq-vae}
Vector-Quantized [VAE](/shared/glossary/#vae) — an [autoencoder](/shared/glossary/#autoencoder) whose latent code is forced to be *discrete*. Instead of letting the encoder output any continuous numbers, each patch of the image must be described using an entry chosen from a small fixed [codebook](/shared/glossary/#codebook), like painting only with the colors in a numbered paint set. Turning an image into a grid of these code indices lets you treat it as a sequence of [tokens](/shared/glossary/#token-visualaudio) and generate it with the same tools used for language. It is trained with a [straight-through estimator](/shared/glossary/#straight-through-estimator) so gradients can flow through the non-differentiable lookup.

### W and W+ latent spaces {#w-and-w-latent-spaces}
The editable latent spaces inside [StyleGAN](/shared/glossary/#stylegan) that dictate how images are generated and controlled.
**W (The Master Remote):** The intermediate space the input noise is first mapped into. Because StyleGAN's training thoroughly disentangles it, it acts like an intuitive master remote control—turning a single "dial" in W smoothly changes one specific attribute (like age) without altering the rest, making it perfect for **editing**.
**W+ (The Individual Room Panels):** A relaxed version of W where *each* layer gets its own independent W code instead of sharing just one. Like abandoning the master remote for highly detailed control panels in every single room, it is harder to tweak one simple trait, but it can represent and reconstruct a specific, complex image much more precisely. This is the space [GAN inversion](/shared/glossary/#gan-inversion) usually targets when trying to match a real-world photo.

### Warmup {#warmup}
The opening phase of training where the learning rate ramps up from near zero to its peak, stabilizing the first noisy updates

### Warp {#warp}
32 threads scheduled in lockstep on NVIDIA GPUs

### Wasserstein GAN (WGAN) {#wasserstein-gan-wgan}
A [GAN](/shared/glossary/#gans) variant that replaces the original loss with the [Earth Mover's Distance](/shared/glossary/#earth-movers-distance) between the real and generated image distributions. The original loss gives almost no [gradient](/shared/glossary/#gradients) once the [discriminator](/shared/glossary/#discriminator) wins, stalling training; the Earth Mover's Distance stays informative even when the two distributions barely overlap, so the [generator](/shared/glossary/#generator) keeps learning. It requires the critic to obey a [Lipschitz constraint](/shared/glossary/#lipschitz-constraint), enforced in the popular [WGAN-GP](/shared/glossary/#wgan-gp) version by a [gradient penalty](/shared/glossary/#gradient-penalty).

### Watermarking {#watermarking}
Hiding an invisible, machine-detectable signal inside a generated image so software can later confirm "this was made by AI" without changing how the picture looks to a human. The signal can be stamped into the pixels after generation (a faint patterned perturbation) or baked into the model's own sampling — Google's SynthID nudges pixel values in a learned pattern, and Tree-Ring plants a ring-shaped mark in the *initial noise* that survives [diffusion](/shared/glossary/#diffusion-model) and is recovered by inverting the generation process. A matching detector then reads the mark back out and reports a confidence score. Like the watermark pressed into a banknote: invisible in normal use, obvious under the right lamp, and hard to forge or scrub off. The built-in tension is robustness vs invisibility — a mark strong enough to survive cropping and JPEG compression is harder to keep imperceptible. Example: generate 1,000 images, run the detector, and it should flag nearly all of them while leaving real photos unflagged.

### WBC {#wbc}
Whole-Body Control — fast QP solving for joint torques from task-space goals

### WebDataset {#webdataset}
A library that streams training data directly from sharded `.tar` archives, avoiding the need to unpack millions of individual files.

### Weight decay {#weight-decay}
A regularization technique that shrinks model parameters toward zero at each update step, discouraging large weights and improving generalization

### Weights {#weights}
The main, larger group of learned [parameters](/shared/glossary/#parameters) in a layer — the `W` in `y = xW + b` — that decide how strongly each input affects each output. Think of the volume sliders on a soundboard: a big weight turns an input way up, a near-zero weight mutes it, and a negative weight flips it. During training the [optimizer](/shared/glossary/#optimizer) keeps nudging these sliders to lower the [loss](/shared/glossary/#loss-function), and they make up the bulk of a model's size.

### WGAN-GP {#wgan-gp}
Short for **Wasserstein GAN with Gradient Penalty** — the most popular and reliable recipe for training a [Wasserstein GAN](/shared/glossary/#wasserstein-gan-wgan). A Wasserstein GAN only works if its critic obeys a [Lipschitz constraint](/shared/glossary/#lipschitz-constraint) (its output can't change too fast). The original WGAN enforced that bluntly, by clipping every critic [weight](/shared/glossary/#weights) back into a fixed range after each step — a heavy-handed move that often crippled the model's quality. WGAN-GP replaces the clipping with a gentle [gradient penalty](/shared/glossary/#gradient-penalty) that simply nudges the size of the critic's [gradient](/shared/glossary/#gradients) toward 1, which keeps training far more stable. Like keeping a car at the speed limit with a smooth governor that eases off the gas, instead of a hard rev-cut that jerks the whole engine every time you nudge past it.

### Whisper {#whisper}
OpenAI's open speech-recognition model — an encoder-decoder [transformer](/shared/glossary/#transformer) that turns a [mel spectrogram](/shared/glossary/#mel-spectrogram) of speech into text, trained on 680,000 hours of multilingual audio scraped from the web. Its encoder digests the audio into rich [embeddings](/shared/glossary/#embedding) and its decoder writes out the words, so one model handles transcription, translation, and language identification. Because that encoder learned such general audio representations, people often reuse just the encoder — freezing it and training a small head on top — as a ready-made audio feature extractor (much like a vision [linear probe](/shared/glossary/#linear-probe)). The name evokes catching even quiet, whispered speech.

### Windowed attention {#windowed-attention}
A cheaper form of [attention](/shared/glossary/#attention) that lets each token attend only to others inside a small local neighborhood (a "window") rather than to the whole sequence. In video, *windowed spatiotemporal attention* applies full joint space-and-time attention but only within small 3D boxes of nearby frames and pixels, so cost grows with the window size instead of the full `T×H×W`. It is the middle ground between cheap [(2+1)D](/shared/glossary/#21d) attention and expensive full [spatiotemporal attention](/shared/glossary/#spatiotemporal-attention): you keep some direct space-time interaction but give up reach across the whole clip. Like reading a document through a small sliding window that shows only a few lines at a time — fast, but you cannot see the whole page at once.

### Worker processes {#worker-processes}
Background subprocesses that a [DataLoader](/shared/glossary/#dataloader) spawns to load and preprocess data in parallel with GPU computation.

### Workhorse {#workhorse}
The dependable, go-to method that does the bulk of the everyday work in a field — not the flashiest, but the one practitioners reach for by default because it reliably gets the job done. Just as a workhorse on a farm pulls the heavy loads day in and day out, [PPO](/shared/glossary/#ppo) earned the title in [RLHF](/shared/glossary/#rlhf) and the [PID](/shared/glossary/#pid) controller earned it in robotics.

### World consistency {#world-consistency}
Whether a generated video keeps a single, self-consistent world as it plays — the same room keeps the same layout, a character keeps the same clothes and face, and the lighting and geography do not silently contradict themselves from one moment (or shot) to the next. It is a step beyond per-frame quality: each frame can look great while the *world* drifts, which is the same [drift](/shared/glossary/#drift) problem that makes long videos fall apart. Together with [object permanence](/shared/glossary/#object-permanence) it is one of the world-behavior criteria [Sora](/shared/glossary/#sora)'s report names as still unsolved, and it is a facet of the broader [physical plausibility](/shared/glossary/#physical-plausibility) challenge.

### World Model {#world-model}
A generative model that predicts the *next* state of an environment given the current state and an [action](/shared/glossary/#action-conditioning) — in plain terms, a video generator that also takes an action input. Run it in a loop, feeding each predicted frame back in as the new current state, and it becomes a learned simulator you can act inside: a human can play it like a game, a [policy](/shared/glossary/#policy) can train inside it by imagining [rollouts](/shared/glossary/#rollout) (as [DreamerV3](/shared/glossary/#dreamerv3) does), or a planner can search through it. A plain [text-to-video](/shared/glossary/#t2v) model is just the special case where the action is empty. Real examples include [Genie](/shared/glossary/#genie) (playable worlds from web video) and [GameNGen](/shared/glossary/#gamengen) (a neural DOOM).

### WSD {#wsd}
Warmup-Stable-Decay — a learning-rate schedule that warms up, holds the rate constant for most of training, then decays sharply at the end.

### XLA {#xla}
Accelerated Linear Algebra — a compiler backend (e.g. for TPUs) used via `torch_xla`

### YaRN {#yarn}
Yet another [RoPE](/shared/glossary/#rope) extensioN method — a context-extension scheme that rescales rotation frequencies unevenly across dimensions to reach long contexts with minimal fine-tuning

### ZeRO {#zero}
[DeepSpeed](/shared/glossary/#deepspeed)'s parameter/gradient/state sharding scheme — comparable to FSDP

### Zero-conv {#zero-conv}
A 1×1 [convolution](/shared/glossary/#convolution-layers) whose weights and bias all start at exactly zero, used by [ControlNet](/shared/glossary/#controlnet) to bolt a new branch onto a pretrained model without disturbing it. At initialization a zero-conv outputs nothing, so the new branch adds *zero* to the original network and the model behaves exactly as before — yet because the layer still receives [gradients](/shared/glossary/#gradients), it can gradually learn how much signal to pass through. Like wiring in a new tap that is turned fully off at first, then opened slowly as training discovers how much to let flow. This is what lets ControlNet train a fresh control signal without damaging the base model's existing quality.

### Zero-shot {#zero-shot}
Doing a task the model was never explicitly trained for, with zero task-specific examples shown at test time. The classic case is [CLIP](/shared/glossary/#clip) zero-shot image classification: instead of training a classifier head on labelled images, you write each candidate label as a short sentence — a *prompt template* such as "a photo of a {label}" — encode every sentence with the [text encoder](/shared/glossary/#text-encoder), and assign the image whichever label sentence has the highest [cosine similarity](/shared/glossary/#cosine-similarity) to its [image embedding](/shared/glossary/#image-embedding). The prompt wording matters: phrasing the label as a natural caption matches the style CLIP saw during training, and averaging several templates (*prompt ensembling*) lifts accuracy a little more. Like a quiz contestant who never studied your specific exam but has read so widely that, handed the answer choices written out in full, they can pick the best match anyway. Example: deciding whether a photo is a cat or a dog by checking whether "a photo of a cat" or "a photo of a dog" sits closer to the image in CLIP's shared space.

### ZMP {#zmp}
Zero-Moment Point — classical biped balance criterion

### Zoom {#zoom}
A camera move that magnifies or shrinks the view without the camera physically moving — like using binoculars to pull a faraway sign closer while your feet stay planted. It narrows or widens the lens's field of view, so the whole frame scales in or out at once. Contrast it with a [dolly](/shared/glossary/#dolly), which changes the picture by actually rolling the camera nearer or farther. It is one of the moves a video model can be steered through with [camera control](/shared/glossary/#camera-control).

### β-VAE {#β-vae}
A [VAE](/shared/glossary/#vae) variant that multiplies the [KL divergence](/shared/glossary/#kl-divergence) part of the [ELBO](/shared/glossary/#elbo) by an adjustable knob called β. Turning β up past 1 pressures the model to use its [latent space](/shared/glossary/#latent-space) more tidily, often making individual latent dimensions line up with meaningful features (like rotation or thickness) — but push it too far and the model stops reconstructing the input well. It is the simplest way to trade reconstruction quality against a cleaner, more interpretable latent.

### σ-schedule (Karras) {#σ-schedule-karras}
The [EDM](/shared/glossary/#edm) convention of describing each noise level by its standard deviation σ (a real number) rather than by a discrete timestep `t` (an index from 0 to ~1000). Because σ directly measures "how much noise is on the image right now," the math for training and sampling becomes cleaner and sampler step sizes are easier to choose. Like labeling oven settings by their actual temperature instead of an arbitrary dial number from 1 to 10.

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
