# Glossary

Terms from all guides in this repository, sorted alphabetically. Each guide's own concepts are included here; see individual guides for deeper context.

---

### (2+1)D {#21d}
Factorized spatiotemporal architecture: separate spatial and temporal layers

### 3D VAE {#3d-vae}
Variational autoencoder that compresses video in time as well as space

### ABA {#aba}
Articulated-Body Algorithm — `O(n)` forward dynamics for rigid-body chains

### Activations {#activations}
The intermediate tensor outputs produced by the layers of a neural network during the forward pass.

### AdaLN {#adaln}
Adaptive layer normalization; the conditioning mechanism in DiT

### AdaLN-Zero {#adaln-zero}
DiT's conditioning mechanism: layer norm modulated by shift/scale/gate, all initialized to zero

### ADD (Adversarial Diffusion Distillation) {#add-adversarial-diffusion-distillation}
SDXL Turbo's recipe: distill a multi-step diffusion model into a 1–4-step student using a discriminator loss

### Admission control {#admission-control}
Refusing requests early when capacity is saturated, to protect SLOs for accepted requests

### Advantage {#advantage}
`A(s, a) = Q(s, a) − V(s)` — how much better than the baseline this action is

### AI (arithmetic intensity) {#ai-arithmetic-intensity}
FLOPs per byte of memory accessed; determines roofline position

### Alignment (multimodal) {#alignment-multimodal}
Making embeddings from different modalities comparable in a shared space

### AllReduce {#allreduce}
A collective op that sums tensors across all ranks and gives every rank the result

### AnyRes {#anyres}
Dynamic-resolution input handling (tile images at native aspect ratio)

### AprilTag {#apriltag}
Square fiducial marker with a known code; widely used for pose ground truth

### ATen {#aten}
The C++ tensor library underneath PyTorch's Python frontend

### Attention {#attention}
The operation `softmax(QKᵀ/√d) V` — content-addressable token mixing; the core of every transformer

### autograd {#autograd}
The reverse-mode automatic differentiation engine

### AWQ {#awq}
Activation-aware Weight Quantization — preserve weights important to large activations

### Backward pass {#backward-pass}
The process of traversing the computation graph in reverse to compute gradients using the chain rule.

### BC {#bc}
Behavior Cloning — supervised imitation of demonstrator actions

### Behavior policy {#behavior-policy}
The policy that generated the data, in off-policy or offline RL

### Bellman equation {#bellman-equation}
The recursive consistency condition `V(s) = E[r + γV(s')]`

### bfloat16 {#bfloat16}
16-bit float with fp32's exponent range — the modern default for training (also written bf16, BF16)

### Bootstrapping {#bootstrapping}
Using a current estimate (e.g., `V(s')`) in the target instead of a full return

### bpd (bits per dimension) {#bpd-bits-per-dimension}
Standard likelihood metric for image models; `-log₂ p(x) / D`

### BPE {#bpe}
Byte-Pair Encoding — subword tokenization by greedy frequent-pair merges

### C-space {#c-space}
Configuration space — the abstract space of joint configurations

### c10 {#c10}
PyTorch's core C++ library (the "core ten[sor]" library)

### CBF {#cbf}
Control Barrier Function — runtime safety filter via a constraint on `ḣ`

### CDNA / RDNA {#cdna--rdna}
AMD's datacenter / consumer GPU architectures

### CFG (classifier-free guidance) {#cfg-classifier-free-guidance}
Inference trick: combine conditional and unconditional model outputs to amplify conditioning

### Chain rule {#chain-rule}
A calculus principle used to compute the derivative of a composite function by multiplying the derivatives of its parts.

### Chat template {#chat-template}
The structured format (system/user/assistant) the model is fine-tuned on

### Chinchilla {#chinchilla}
The scaling law showing compute-optimal training uses ~20 tokens per parameter

### Chunked prefill {#chunked-prefill}
Splitting long prompts across multiple iterations to interleave with decode steps

### CLIP {#clip}
Contrastive Language-Image Pretraining — paired text-image dual encoder

### Collision mesh {#collision-mesh}
Simplified geometry used for collision tests, distinct from visual mesh

### Consistency model {#consistency-model}
A diffusion-derived model that samples in 1–4 steps via consistency distillation

### Context window {#context-window}
The maximum number of tokens the model can attend over in one forward pass

### ControlNet {#controlnet}
Architecture that adds an auxiliary conditioning branch (depth, pose, edges, …) to a frozen diffusion model

### copy {#copy}
A tensor that owns its own storage, independent of any source tensor; created by `.clone()`, or automatically by operations like `.contiguous()` and `reshape` when a view is not possible

### CoT {#cot}
Chain of Thought — prompting / training the model to produce intermediate reasoning

### CQL {#cql}
Conservative Q-Learning — offline RL with a pessimistic Q penalty

### Cross-attention {#cross-attention}
Attention where queries come from one modality and keys/values from another

### CUDA {#cuda}
NVIDIA's GPU compute backend; tensors on the `cuda` device run their kernels here

### CUTLASS {#cutlass}
NVIDIA's open template library for matmul kernels

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

### Detached tensor {#detached-tensor}
A tensor that has been removed from the [dynamic computation graph](/shared/glossary/#dynamic-computation-graph) via the `.detach()` method, meaning operations performed on it will not be tracked for [autograd](/shared/glossary/#autograd).

### Derivative {#derivative}
The instantaneous rate of change of a function with respect to its input. In deep learning, derivatives are computed via the chain rule during backpropagation to produce gradients used to update model parameters.

### DH parameters {#dh-parameters}
Denavit-Hartenberg parameters — textbook arm-geometry description

### Disaggregated serving {#disaggregated-serving}
Running prefill and decode on separate GPU pools with KV cache transfer between them

### Dispatcher {#dispatcher}
The PyTorch component that routes `torch.foo(...)` calls to the right backend/dtype kernel

### DiT {#dit}
Diffusion Transformer — Peebles & Xie's transformer-based diffusion backbone

### Double backward {#double-backward}
Computing the gradient of a gradient by tracking the backward pass operations in a new computation graph.

### DPO {#dpo}
Direct Preference Optimization — closed-form RLHF without a reward model or PPO

### DQN {#dqn}
Deep Q-Network — Q-learning with neural-net function approximation + experience replay + target network

### DreamBooth {#dreambooth}
Fine-tuning recipe for subject personalization; updates the whole model on a few subject images

### dtype {#dtype}
A tensor's element data type — e.g. `float32`, `float16`, `bfloat16`, `int8`, `bool`

### Dynamic computation graph {#dynamic-computation-graph}
A graph of operations built on-the-fly as code executes, representing the forward pass used for autograd.

### EAGLE / Medusa {#eagle--medusa}
Self-speculation: extra heads on the target model propose tokens, no separate draft model

### EDM {#edm}
Karras et al. 2022 — a reformulation of diffusion in σ-space with clean preconditioning

### EKF {#ekf}
Extended Kalman Filter — Kalman filter linearized about the current estimate

### ELBO {#elbo}
Evidence Lower Bound — the variational objective trained by VAEs

### EMA weights {#ema-weights}
Exponential moving average of model weights; samples better than the live weights

### Expert parallelism (EP) {#expert-parallelism-ep}
For MoE models, distributing experts across GPUs with all-to-all token routing

### F/T sensor {#ft-sensor}
Force/Torque sensor — six-axis force and moment at a wrist or fingertip

### FID {#fid}
Fréchet Inception Distance — the standard sample-quality metric for image generation

### FK / IK {#fk--ik}
Forward / Inverse Kinematics — compute end-effector pose from joints or vice versa

### FlashAttention {#flashattention}
IO-aware attention kernel that avoids materializing the T×T score matrix in HBM

### float16 {#float16}
16-bit floating-point format (`fp16`); saves memory and can be fast on GPUs, but has a limited range (max ~65,504) that can cause [underflow](/shared/glossary/#underflow) when accumulating very small values

### float32 {#float32}
32-bit floating-point format (`fp32`); the standard default precision for PyTorch tensors — wide enough range and enough precision for most training and inference tasks

### Flow matching {#flow-matching}
Training a velocity field that transports noise to data via an ODE; modern alternative to DDPM

### FP8 {#fp8}
8-bit floating point (E4M3 / E5M2 on Hopper+); the modern default serving precision

### FSDP {#fsdp}
Fully Sharded Data Parallel — shard params, grads, and optimizer state across ranks

### FSQ {#fsq}
Finite Scalar Quantization — codebook-free discrete tokenization

### Fusion (early/middle/late) {#fusion-earlymiddlelate}
When in the network different modalities are combined

### FVD {#fvd}
Fréchet Video Distance — the standard (and flawed) automatic eval metric for video generation

### GAE {#gae}
Generalized Advantage Estimation — TD(λ) for advantages

### GANs (Generative Adversarial Networks) {#gans}
A class of generative models in which a generator network and a discriminator network are trained adversarially. The generator learns to produce realistic samples to fool the discriminator, which learns to distinguish real from generated data.

### GPTQ {#gptq}
Hessian-based per-row PTQ minimizing layer-wise reconstruction error

### GQA {#gqa}
Grouped-Query Attention — sharing K/V heads across query heads; primary KV-cache saver at serving time

### Gradients {#gradients}
The vector of partial derivatives of a function with respect to its inputs. In neural networks, gradients represent the direction and magnitude of the change required to minimize the [loss function](/shared/glossary/#loss-function).

### Gradient checkpointing {#gradient-checkpointing}
A memory-saving technique that discards intermediate activations during the forward pass and recomputes them during the backward pass.

### Grounding {#grounding}
Producing spatial outputs (boxes, points) referring to image regions

### GRPO {#grpo}
Group Relative Policy Optimization — value-function-free PPO variant; DeepSeek lineage

### GTSAM {#gtsam}
Factor-graph SLAM library; the standard back-end for many modern systems

### HBM {#hbm}
High-Bandwidth Memory — stacked DRAM on a modern GPU; usually the bandwidth bottleneck

### Holonomic {#holonomic}
A vehicle whose instantaneous motion can be any direction (mecanum, omni)

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

### InfiniBand (IB) {#infiniband-ib}
High-speed network with RDMA; standard for AI clusters

### InfoNCE {#infonce}
The contrastive loss used by CLIP; softmax over a similarity matrix

### IQL {#iql}
Implicit Q-Learning — offline RL that never queries `Q` at OOD actions

### Isaac Lab {#isaac-lab}
NVIDIA GPU-parallel robotics simulation platform

### ITL / TPOT {#itl--tpot}
Inter-token latency / time per output token — steady-state per-token decode time

### Jacobian {#jacobian}
Linear map from joint velocities to end-effector spatial velocity

### KF {#kf}
Kalman Filter — optimal linear-Gaussian Bayes filter

### KL divergence {#kl-divergence}
The regularizer that keeps RLHF policies near the reference model

### KV cache {#kv-cache}
Cached keys and values per past token per layer; the working set of the decoder

### Latent video {#latent-video}
Compressed (T', H', W', C) tensor produced by a 3D VAE

### LCM {#lcm}
Latent Consistency Model — consistency-distilled few-step latent diffusion

### LDM {#ldm}
Latent Diffusion Model — diffusion in the latent space of a VAE (i.e., Stable Diffusion)

### LiDAR {#lidar}
Light Detection And Ranging — laser range scanner

### LoRA {#lora}
Low-Rank Adaptation — fine-tune by adding small low-rank matrices, freeze the base

### Loss function {#loss-function}
A mathematical function that measures the difference between a model's prediction and the actual target. The goal of training is to minimize this value using [gradients](/shared/glossary/#gradients).

### Lorax / S-LoRA {#lorax--s-lora}
Multi-LoRA serving engines; one base model + many adapters in HBM

### LQR {#lqr}
Linear-Quadratic Regulator — optimal linear feedback for quadratic cost

### MagViT-v2 {#magvit-v2}
The strongest open recipe for discrete video tokenization

### Manipulability {#manipulability}
Scalar measure of how "easy" motion is from a given configuration (e.g. `sqrt(det(JJᵀ))`)

### MDP {#mdp}
Markov Decision Process — the tuple `(S, A, P, R, γ)`

### Meta-learning {#meta-learning}
"Learning to learn" — training a model to adapt quickly to new tasks with few examples. Many meta-learning algorithms, such as MAML, rely on higher-order gradients to optimize across tasks.

### Memory leak {#memory-leak}
An unintended increase in memory usage over time, often caused in PyTorch by holding onto references to the [loss function](/shared/glossary/#loss-function) or other parts of the [dynamic computation graph](/shared/glossary/#dynamic-computation-graph) across training iterations.

### Micrograd {#micrograd}
A tiny, educational autograd engine implemented in basic Python by Andrej Karpathy to illustrate how reverse-mode differentiation works.

### MMDiT {#mmdit}
Multi-Modal Diffusion Transformer — joint text+image attention layers, used in SD3 and Flux

### Modality gap {#modality-gap}
Empirical finding that different-modality embeddings stay in separable regions

### Mode collapse {#mode-collapse}
GAN failure mode: generator produces few distinct outputs

### MoE {#moe}
Mixture-of-Experts — sparse routing across N expert MLPs; high total params, fixed compute per token

### MoveIt {#moveit}
ROS 2 manipulation-planning framework

### MPC {#mpc}
Model Predictive Control — re-solved finite-horizon optimization at each step

### MPS {#mps}
Metal Performance Shaders — the GPU backend for Apple Silicon

### MuJoCo {#mujoco}
Open-source physics engine; the de facto manipulation/locomotion simulator

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

### non_blocking {#non_blocking}
The `non_blocking=True` flag on `.to()` / `.cuda()` that lets a host→device copy run asynchronously from pinned memory

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

### On-policy {#on-policy}
The data comes from the same policy being optimized (PPO, REINFORCE)

### PagedAttention {#pagedattention}
KV cache managed as fixed-size physical blocks with per-request block tables

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

### Pinocchio {#pinocchio}
Fast rigid-body dynamics library (CRBA, RNEA, ABA)

### Plücker coordinates {#plücker-coordinates}
6D representation of a camera ray; standard for camera-conditioning

### Posterior collapse {#posterior-collapse}
VAE failure mode: encoder collapses to the prior; latent carries no information

### PPO {#ppo}
Proximal Policy Optimization — the workhorse on-policy RL algorithm, used in classic RLHF

### Prefill {#prefill}
Processing the prompt in one parallel forward pass before decoding begins

### Prefix cache {#prefix-cache}
Sharing KV cache across requests that begin with the same tokens (e.g., system prompts)

### Pretraining {#pretraining}
Self-supervised training on a large unlabeled corpus to predict the next token

### PRM {#prm}
Probabilistic Roadmap — multi-query sampling-based planner

### Probability flow ODE {#probability-flow-ode}
The deterministic ODE equivalent of the reverse-time diffusion SDE

### Projector {#projector}
The (usually small) network that maps one modality's features into another's space

### PTQ / QAT {#ptq--qat}
Post-Training Quantization / Quantization-Aware Training

### Q-Former {#q-former}
BLIP-2's learnable-query cross-attention module for distilling images into LLM tokens

### Quantization {#quantization}
Reducing weight / activation precision (FP16, BF16, FP8, INT8, INT4) to save memory and bandwidth

### RadixAttention {#radixattention}
sglang's KV cache organized as a radix tree keyed on prompt prefixes for automatic sharing

### RAG {#rag}
Retrieval-Augmented Generation — fetch documents, prepend to prompt, then generate

### Rectified flow {#rectified-flow}
A flow-matching parameterization with straight-line trajectories; popular in 2024+ models

### Reparameterization trick {#reparameterization-trick}
`z = μ + σ · ε` — lets gradients flow through a random sample

### reshape {#reshape}
Returns a tensor with a new shape, copying only when a no-copy view isn't possible

### Reward hacking {#reward-hacking}
A policy that maximizes the reward signal without doing what was intended

### RLHF {#rlhf}
Reinforcement Learning from Human Feedback — preference learning, classically via PPO + KL

### RLVR {#rlvr}
RL with Verifiable Rewards — RL when the reward is a deterministic checker

### RMSNorm {#rmsnorm}
Root-Mean-Square LayerNorm without mean-centering; the modern default

### RNEA {#rnea}
Recursive Newton-Euler — `O(n)` inverse-dynamics algorithm

### Roofline {#roofline}
Performance model bounding throughput as min(peak FLOPs, memory bandwidth × arithmetic intensity)

### RoPE {#rope}
Rotary Position Embedding — encodes position by rotating Q, K vectors

### ROS / ROS 2 {#ros--ros-2}
Robot Operating System — robotics middleware (ROS 2 is the modern version)

### RRT {#rrt}
Rapidly-exploring Random Tree — single-query sampling-based planner

### SAC {#sac}
Soft Actor-Critic — maximum-entropy continuous-control algorithm; the modern default

### SAE {#sae}
Sparse Autoencoder — interpretability tool decomposing activations into monosemantic features

### Score {#score}
`∇_x log p(x)` — diffusion training implicitly learns this

### SDF {#sdf}
Signed Distance Field — scalar field giving distance to nearest obstacle (negative inside)

### SE(3) / SO(3) {#se3--so3}
Special Euclidean / Orthogonal group — rigid-body motions / rotations in 3D

### SFT {#sft}
Supervised Fine-Tuning — train on demonstration data with cross-entropy

### Shape {#shape}
The size of a tensor along each dimension; the tuple returned by `.shape`

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

### Speculative decoding {#speculative-decoding}
Use a draft model to propose tokens; verify with the target in one parallel pass; accepted tokens are appended

### Storage {#storage}
The 1-D buffer that a tensor is a view into

### Straight-through estimator {#straight-through-estimator}
A technique used to bypass non-differentiable operations by passing gradients unchanged through the operation during the backward pass.

### Stride {#stride}
The number of storage elements to step over for each dimension of a tensor

### SwiGLU {#swiglu}
Gated MLP activation `(xW) · σ(xV)` — the modern default FFN

### Systolic array {#systolic-array}
Data-flow matmul fabric used in TPUs

### T2V {#t2v}
Text-to-Video

### TCP {#tcp}
Tool Center Point — the configurable point on a tool whose pose tracking controls

### TD error {#td-error}
`δ_t = r_t + γV(s_{t+1}) − V(s_t)` — the signal that drives every TD update

### TD3 {#td3}
Twin Delayed DDPG — DDPG plus three stability fixes

### Temporal inflation {#temporal-inflation}
Adding time-axis layers to a pretrained 2D model

### Tensor {#tensor}
A multidimensional array — a (storage, shape, stride, offset, dtype, device, requires_grad) tuple that views a 1-D storage buffer

### Tensor Core {#tensor-core}
Specialized matmul unit in NVIDIA GPUs since Volta

### Tensor parallelism (TP) {#tensor-parallelism-tp}
Sharding each layer's weights across GPUs with all-reduce at attention/MLP boundaries

### TFLOPs {#tflops}
Tera (10¹²) floating-point operations per second

### Token (visual/audio) {#token-visualaudio}
Discrete code from a VQ-VAE or neural codec; lets transformers treat the modality like language

### Tokenizer {#tokenizer}
The mapping from string to integer IDs; trained, frozen, part of the model contract

### TOPP {#topp}
Time-Optimal Path Parameterization — time-parameterize a geometric path under bounds

### TorchScript {#torchscript}
The legacy serialization/IR for PyTorch; superseded by `torch.export`

### Transformer {#transformer}
The decoder-only / encoder-only / encoder-decoder architecture built from attention + MLP blocks

### transpose {#transpose}
Swaps two dimensions by rewriting strides — never copies; the result is usually non-contiguous

### Triton {#triton}
A Python-flavored language for writing GPU kernels, developed by OpenAI

### TTFT {#ttft}
Time to first token — dominated by prefill plus queue wait

### U-Net {#u-net}
Encoder-decoder architecture with skip connections; the standard diffusion backbone before DiT

### Underflow {#underflow}
Condition where a floating-point value is too small to be represented and rounds to zero; common with `float16` when accumulating very small gradients

### URDF / MJCF / USD {#urdf--mjcf--usd}
Robot description formats (ROS, MuJoCo, NVIDIA respectively)

### V2V {#v2v}
Video-to-Video

### Vanishing gradients {#vanishing-gradients}
A problem during training where [gradients](/shared/glossary/#gradients) become extremely small, effectively preventing the weights from changing their value and stalling the learning process.

### VAE {#vae}
Variational Autoencoder — encoder/decoder pair trained on the ELBO

### Value function {#value-function}
Expected return; `V(s)` for state-value, `Q(s, a)` for action-value

### VBench {#vbench}
Comprehensive open evaluation suite for video generation

### view {#view}
A no-copy alias that shares storage with its source; requires a contiguous-compatible layout

### VIO {#vio}
Visual-Inertial Odometry — fuse camera and IMU for high-rate ego-motion

### VLA {#vla}
Vision-Language-Action model — transformer mapping image + instruction → action

### vLLM {#vllm}
The reference open-source inference engine with PagedAttention and continuous batching

### VLM {#vlm}
Vision-Language Model — image (+ text) in, text out

### VP / VE SDE {#vp--ve-sde}
Variance-Preserving / Variance-Exploding — the two SDE families for diffusion

### VQ-GAN {#vq-gan}
VQ-VAE trained with perceptual + adversarial losses; SD's VAE recipe descends from this

### VQ-VAE {#vq-vae}
Vector-quantized VAE — discrete latent codes from a learned codebook

### Warp {#warp}
32 threads scheduled in lockstep on NVIDIA GPUs

### WBC {#wbc}
Whole-Body Control — fast QP solving for joint torques from task-space goals

### World Model {#world-model}
Action-conditioned generative model of the world; a video model with actions

### XLA {#xla}
Accelerated Linear Algebra — a compiler backend (e.g. for TPUs) used via `torch_xla`

### ZeRO {#zero}
DeepSpeed's parameter/gradient/state sharding scheme — comparable to FSDP

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
