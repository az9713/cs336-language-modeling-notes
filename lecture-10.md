# Lecture 10 — Inference as a memory and scheduling problem

Inference serves interactive users, agent trajectories, evaluation, synthetic-data generation, and reinforcement-learning rollouts. The lecture moves from latency and throughput to prefill/decode accounting, cache-aware architectures, quantization, speculative sampling, and serving systems. These are different interventions on the same constrained computation.

## 1. Define “fast” before optimizing [03:00](https://www.youtube.com/watch?v=EfM546A79aM&t=180s)

**Time to first token (TTFT)** includes waiting and prompt processing before generation begins. **Inter-token latency** measures delay between generated tokens for a request. **Throughput** measures aggregate tokens or requests processed per second. A larger batch can improve aggregate throughput while worsening the delay experienced by an individual request.

For a decode iteration taking $t(B)$ seconds and producing one token for each of $B$ active requests, throughput is approximately $B/t(B)$. Per-request inter-token latency is approximately $t(B)$, apart from scheduling and streaming overhead. These formulas explain how both latency and throughput can increase together.

**Worked example.** Batch one takes 10 ms per iteration, producing 100 tokens/s. Batch sixteen takes 20 ms, producing 800 tokens/s in aggregate. Each user now sees half the token rate, while the service produces eight times as many tokens. Which operating point is preferable depends on the latency objective and demand.

Agent workloads make token counts especially important. A long private reasoning or tool-use trace may consume many more generated tokens than the final answer shown to a human. Serving economics should count the actual computation.

## 2. Prefill and decode have different reuse patterns [14:00](https://www.youtube.com/watch?v=EfM546A79aM&t=840s)

During **prefill**, the model processes many prompt positions in parallel and constructs cached keys and values. During **decode**, each request usually supplies one new position per iteration. The model reuses previous KV states rather than recomputing the entire prefix.

For a weight matrix $W\in\mathbb R^{d\times f}$ applied to $M$ token vectors, arithmetic is approximately $2Mdf$ FLOPs. If weight traffic dominates and weights use $b$ bytes, intensity is approximately $2M/b$. Prefill can make $M$ large. In small-batch decode, $M$ is small and reading weights can dominate.

**Worked example.** With two-byte weights and batch one decode, the weight-dominated intensity is approximately one FLOP/byte. Batch 64 raises it toward 64 FLOPs/byte for that projection. This is why batching helps weight reuse. It does not equally improve all operations: each request generally has its own attention cache.

For attention, a new query must read the relevant cached keys and values. Increasing batch adds distinct histories, so cache traffic grows with batch rather than being shared like weights. Long contexts can therefore make cache bandwidth the dominant constraint even when weight reuse is good.

## 3. Cache size determines concurrency [23:00](https://www.youtube.com/watch?v=EfM546A79aM&t=1380s)

For $L$ layers, batch $B$, context $T$, KV heads $H_{kv}$, head width $d_h$, and element size $b$, the usual cache estimate is

$$M_{KV}=2LBTH_{kv}d_hb.$$

Weights are mostly independent of active context length; cache is not. In a growing generation, the service must reserve or dynamically allocate additional cache blocks. Long prompts and long outputs can reduce the number of concurrent requests that fit.

**Worked example.** A 32-layer model with eight KV heads of width 128 and two-byte cache uses 128 KiB per token per request across all layers. A 16,384-token history therefore uses 2 GiB. Eight such histories use 16 GiB before weights and workspaces. This calculation is often more useful than a model's parameter count for predicting long-context concurrency.

An optimistic bandwidth bound for a decode step is bytes that must be read divided by sustained bandwidth. Counting weights plus the current cache gives a starting point, but real kernels may reread data, exploit cache, overlap transfers, and incur communication. Label the bound as an estimate rather than measured latency.

## 4. Architecture can reduce what inference must carry [46:00](https://www.youtube.com/watch?v=EfM546A79aM&t=2760s)

GQA reduces the number of distinct KV heads while retaining more query heads. Multi-head latent attention compresses cached information into a lower-dimensional representation, with additional architectural details needed to recover useful attention and handle position encoding. Sliding-window layers bound direct cache access by a recent window. Recurrent layers use a fixed-size state, and hybrid models mix these mechanisms.

Each changes a different part of the computation. A local window can discard direct access to distant tokens at that layer. A recurrent state compresses history. A sparse indexer selects positions but has its own overhead. These are quality–efficiency choices requiring task-specific evaluation.

**Worked example.** Reducing KV heads from 32 to eight cuts the head-dependent cache term by four. It does not cut every model parameter or every inference operation by four. Reporting a “fourfold memory improvement” without specifying the cache component would overstate the result.

The [GQA paper](https://arxiv.org/abs/2305.13245) and the course's linked model reports provide empirical evidence under particular setups. Conflicting quality results across models are a reason to reproduce an ablation in the intended setting, not to discard the accounting.

## 5. Quantization, pruning, and distillation alter the deployed model [64:00](https://www.youtube.com/watch?v=EfM546A79aM&t=3840s)

Quantization reduces representation precision, often with per-group scales. Weight-only quantization can reduce weight traffic but leaves activation and cache costs unless they are also quantized. The packed format, scaling metadata, and dequantization work belong in the measurement.

Pruning removes or sparsifies parts of a model. A nominally sparse matrix produces a speedup only when the hardware and kernels exploit its pattern. Distillation trains a student to approximate a teacher's outputs or behavior. It changes the model and introduces a training problem, including which prompts and outputs define the approximation.

**Worked example.** Quantizing a hypothetical 8-billion-parameter model from 16-bit to 4-bit weights reduces the ideal raw weight payload from 16 GB to 4 GB. Scale metadata, unquantized tensors, workspaces, and cache increase the actual memory. At long context, the cache may become the dominant remaining cost, so weight compression alone may not enable the expected batch increase.

These methods trade some combination of quality, preprocessing cost, implementation complexity, and hardware compatibility for deployment efficiency. The useful comparison is at a matched quality target and realistic request distribution.

## 6. Speculative sampling can preserve the target distribution [71:00](https://www.youtube.com/watch?v=EfM546A79aM&t=4260s)

Let $q$ be a cheap draft distribution and $p$ the desired target distribution at a given prefix. Sample candidate $x\sim q$ and accept it with probability

$$a(x)=\min(1,p(x)/q(x)).$$

On rejection, sample from the normalized residual distribution

$$r(x)\propto[p(x)-q(x)]_+,$$

where $[z]_+=\max(z,0)$. The accepted probability mass at $x$ is $q(x)a(x)=\min(p(x),q(x))$. The missing mass is exactly $[p(x)-q(x)]_+$. Adding accepted and residual mass recovers $p$.

**Worked example.** On two tokens, let $p=(0.7,0.3)$ and $q=(0.4,0.6)$. The first token is always accepted when drafted. The second is accepted with probability $0.3/0.6=0.5$. Accepted mass is $(0.4,0.3)$; rejected mass is 0.3 and the residual puts it all on the first token. Final probabilities are $(0.7,0.3)$.

For multiple drafted tokens, the target evaluates candidate prefixes in parallel, accepts a prefix according to conditional probabilities, and handles the first rejection with the correction. Exactness depends on implementing the algorithm correctly with the intended sampling transformations. Simply keeping every draft token above an arbitrary confidence threshold is a different procedure.

The original [speculative decoding paper](https://arxiv.org/abs/2211.17192) establishes the sampling method. Speed depends on acceptance rate, draft cost, verification cost, batch size, and hardware; it is not guaranteed for every workload.

## 7. Continuous batching and paged cache allocation serve variable requests [77:00](https://www.youtube.com/watch?v=EfM546A79aM&t=4620s)

Static batches can waste slots after short requests finish while waiting for long requests. Continuous batching updates the active set at iteration boundaries, admitting new requests and removing completed ones. This increases utilization but requires scheduling decisions about prefill, decode, fairness, and latency.

Paged cache allocation divides a request's logical KV sequence into fixed-size blocks mapped to physical storage. It reduces the need for one large contiguous reservation and can support sharing of identical prefixes with suitable reference counting and copy-on-write behavior. The [PagedAttention/vLLM paper](https://arxiv.org/abs/2309.06180) explains the design.

Paging does not make cache bytes free or eliminate internal waste completely. A partly filled final block still wastes space, and block tables and kernels add overhead. The benefit is more flexible management of a resource whose size changes during generation.

## 8. Researched extension — queueing belongs in the latency model

Even a fast kernel can serve a slow application if requests wait in a queue. Little's law relates average number of requests in a stable system, $L_q$, arrival rate $\lambda$, and average time in the system $W$: $L_q=\lambda W$. This is a general queueing identity under the relevant long-run averages, not a prediction of the latency distribution.

For a service sustaining 20 requests/s with average end-to-end time 2 seconds, the average number of in-flight requests is 40. If memory permits only a much smaller active set, admission and scheduling must account for the backlog. Tail latency, prompt-length variability, and prefill interference require measurements beyond averages.

## 9. Exercises and worked solutions

**Exercise 1.** In the speculative example, what fraction of drafts is accepted?

<details><summary>Solution</summary>

Total accepted mass is $\sum_x\min(p(x),q(x))=0.7$. In general this equals one minus total variation distance between the two distributions. High distributional agreement supports more accepted work, but speed still depends on computation cost.

</details>

**Exercise 2.** Why does batching improve weight reuse more directly than cache reuse?

<details><summary>Solution</summary>

Requests use the same weight matrices, so one loaded weight block can serve many token vectors. Their histories are usually distinct, so each request brings its own keys and values. Shared-prefix caching is a special case requiring identical reusable prefixes and careful state management.

</details>

**Exercise 3.** Give four measurements for an inference comparison.

<details><summary>Solution guidance</summary>

Measure TTFT, per-request decode latency, aggregate throughput, and peak memory across representative prompt/output lengths and concurrency. Also measure task quality when the model or numerical representation changes. Report whether queueing and transfers are included.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=EfM546A79aM) and [official code](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_10.py).
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): inference resource models.
- [Speculative decoding](https://arxiv.org/abs/2211.17192), [PagedAttention](https://arxiv.org/abs/2309.06180), and [GQA](https://arxiv.org/abs/2305.13245).
