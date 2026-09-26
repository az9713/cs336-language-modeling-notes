# Lecture 18 — Dan Fu: inference systems and stable looped models

This is the eighteenth recording in the supplied playlist, a guest lecture by Dan Fu. The official course calendar also contains a different guest session; playlist position is used here to preserve the user's requested 18-recording scope. No corresponding Lecture 18 file was present in the pinned official materials repository, so this chapter uses the recording and the speaker's primary research and engineering publications.

The talk follows an inference request through a serving system, examines cache-aware routing and kernel scheduling, and then connects deployment constraints to looped language-model architecture. The equations and numerical examples below expand the lecture's dynamical-systems discussion and correct compressed terminology where necessary.

## 1. Follow a request through the whole serving system [06:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=360s)

A request passes through routing, admission, tokenization, cache lookup, prompt processing, decoding, and response streaming. Each stage can affect latency. A model benchmark that starts with prepared tensors on an idle GPU measures only part of this path.

Let end-to-end time be schematically

$$T=T_{\rm queue}+T_{\rm prepare}+T_{\rm prefill}+T_{\rm decode}+T_{\rm transfer}.$$

Terms can overlap in a real system, so this sum is a bookkeeping starting point rather than a strict decomposition. It makes omitted costs visible and separates waiting from useful computation.

**Worked example.** A kernel optimization cuts decode compute from 100 to 80 ms, but a request also waits 300 ms and spends 100 ms elsewhere. Total latency changes from 500 to 480 ms: a 4% improvement despite a 20% improvement in the optimized component. Amdahl's law is a useful guard against exaggerated application claims.

Interactive speech, ordinary chat, batch generation, and autonomous agents impose different latency requirements. The best scheduler and hardware allocation can differ even when they use the same underlying model.

## 2. Prefill and decode can deserve different resources [12:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=720s)

Prefill processes a prompt's many positions; decode repeatedly advances active requests by a small number of positions. Their arithmetic intensity and latency sensitivity differ. Disaggregating them can allow independent scheduling and hardware choices, but it also requires transferring the resulting cache and coordinating request state.

If disaggregation saves compute time $\Delta T$ but introduces cache transfer time $M_{KV}/\beta_{\rm link}$ plus coordination overhead, the change helps only when the saved time exceeds the added critical-path cost. Cache size and interconnect performance belong in the design, not just GPU FLOP rates.

**Worked example.** Moving a 2 GiB cache over a sustained 50 GiB/s link takes at least 40 ms, ignoring protocol overhead. A design that saves only 10 ms of computation cannot win on that transfer path without overlap, compression, reuse, or some other change. This is a hypothetical accounting example, not a measurement from the talk.

## 3. Prefix caching creates a memory-placement problem [18:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1080s)

Identical prefixes can reuse previously computed states under compatible model, tokenizer, positional, and numerical conditions. The cache may live in GPU memory, host memory, or a lower storage tier. Moving between tiers trades retrieval latency against capacity.

A cache key must identify the computation, not just a human-readable string. A changed model version or adapter can invalidate activations. Tokenization differences, position conventions, and other execution settings can also matter. Incorrect reuse can silently corrupt output.

**Worked example.** Two requests contain the same visible document but use different system prefixes. If the document's causal states depend on those prefixes, its cached activations are not generally interchangeable. Exact prefix reuse is stronger than matching an arbitrary repeated substring.

Cache eviction is an allocation decision under uncertainty. A large rarely reused prefix can displace many smaller frequently reused ones. Estimate saved prefill work per byte and per expected reuse, while accounting for retrieval cost. A simple least-recently-used policy is a baseline, not a universal optimum.

## 4. Route cold and warm work according to its actual cost [31:30](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1890s)

The talk describes separating requests with low cache-hit rates from those whose context is largely reusable. Raw prompt length alone can misrepresent the work: a long warm prompt may require little fresh prefill, while a shorter cold prompt may require much more.

The speaker's engineering team describes this in [cache-aware prefill–decode disaggregation](https://www.together.ai/blog/cache-aware-disaggregated-inference). Its reported performance improvement is workload-specific. The general lesson is to route using estimated uncached work and cache location, and to evaluate both throughput and latency under mixed traffic.

**Worked example.** Request A contains 20,000 tokens with a 95% reusable prefix; request B contains 4000 entirely new tokens. Fresh-token counts are roughly 1000 and 4000. A scheduler based only on total length can classify A as heavier even though B requires more new prompt processing. Full cost also includes attending to cached context, so fresh-token count is a useful feature rather than a complete predictor.

## 5. Kernel timelines reveal unused concurrency [35:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2100s)

The recording examines timelines across GPU streaming multiprocessors. Empty regions can reflect dependencies, uneven tile counts, or work that has not been scheduled concurrently. A kernel can be individually efficient while leaving gaps between stages of a larger computation.

Projection, positional transformation, cache loading, and attention have dependencies, but some data movement can begin before all arithmetic finishes. Overlap requires identifying which subset of data is ready and arranging buffers and synchronization correctly. It is not enough to launch everything asynchronously and hope dependencies are satisfied.

**Worked example.** If an attention tile needs only one completed query tile, it may start before every query projection is finished. This can reduce a global barrier into a producer–consumer pipeline. The benefit depends on tile granularity and resource competition: two kernels that each saturate the same resource may slow each other when overlapped.

This is the talk's broader research claim: understanding the serving engine and kernels can reveal architectural opportunities that are invisible from parameter count alone.

## 6. Looping adds computation without adding distinct weights [42:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2520s)

A looped model applies a shared block repeatedly. Let input representation be $e$, hidden state $h_r$ after recurrence $r$, shared update $F_\theta$, and readout $G$:

$$h_{r+1}=F_\theta(h_r,e),\qquad y=G(h_R).$$

Increasing recurrence count $R$ increases computation while keeping shared parameter count fixed. It is analogous to increasing effective depth with tied weights, but training and inference behavior depend on how the input is injected and how recurrence counts are sampled.

**Worked example.** A four-layer shared block run six times has 24 block applications but only four layers' distinct weights in that region. It does not have the same capacity as 24 independently parameterized layers. It may save weight memory while requiring more sequential work. Activation and KV-cache behavior require separate analysis; constant parameter memory is not a proof of constant total serving memory.

The [Parcae paper](https://arxiv.org/abs/2604.12946) studies stabilizing such models and scaling recurrence alongside data. Its results support a promising design direction within its tested settings, not a universal claim that looping dominates ordinary depth.

## 7. Derive the linear recurrence and distinguish its stability conditions [47:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2820s)

The talk isolates a schematic residual update

$$h_{r+1}=Ah_r+Be+\mathcal R(h_r,e).$$

Temporarily omit the nonlinear remainder $\mathcal R$ to study a linear surrogate. Repeated substitution gives

$$h_R=A^Rh_0+\sum_{j=0}^{R-1}A^jBe.$$

If $I-A$ is invertible, the finite sum is $(I-A^R)(I-A)^{-1}Be$. If $A^R\to0$, the state approaches $(I-A)^{-1}Be$.

For a finite-dimensional, time-invariant discrete linear system, asymptotic stability is characterized by **spectral radius** $\rho(A)=\max_i|\lambda_i(A)|<1$. The spectral norm $\|A\|_2$ is the largest singular value. It is generally different. The condition $\|A\|_2<1$ is sufficient for contraction in Euclidean norm, but not necessary for asymptotic stability.

**Worked example.** For scalar $a=1.1$, the initial-state contribution grows by $1.1^{50}\approx117.4$. For $a=0.9$, it shrinks to about 0.00515. At $a=1$ with nonzero constant injection $be$, the state can grow linearly as $h_R=h_0+Rbe$ even though the homogeneous part does not explode exponentially.

For a nonnormal matrix, powers can initially grow even when all eigenvalues lie inside the unit circle. Thus a spectral-radius statement alone does not bound every finite-depth transient. That distinction matters when a network runs a finite number of loops and uses finite-precision arithmetic.

## 8. Continuous-time negativity must be discretized correctly [51:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3060s)

A negative diagonal matrix in a continuous-time generator can produce stable discrete dynamics through exponentiation. Let $A_c=-\operatorname{diag}(a_1,\ldots,a_d)$ with $a_i>0$ and step $\Delta>0$. Then

$$A_d=e^{\Delta A_c}=\operatorname{diag}(e^{-\Delta a_1},\ldots,e^{-\Delta a_d}),$$

whose diagonal entries lie strictly between zero and one. Parcae uses a related constrained parameterization and input normalization. This clarifies the compressed spoken explanation: simply raising an arbitrary negative discrete coefficient to powers is not stable. For example, $(-2)^R$ diverges in magnitude.

**Worked example.** A continuous decay rate $a=2$ with $\Delta=0.1$ gives discrete multiplier $e^{-0.2}\approx0.819$. By contrast, forward Euler uses $1-a\Delta$ and is stable only when $|1-a\Delta|<1$ in this scalar case. Discretization method and step size matter.

The nonlinear remainder must still be considered. If its Jacobian adds strong expansion, stabilizing $A$ alone is not a global proof for the complete neural network. Empirical state norms, gradients, loss trajectories, and controlled ablations remain essential.

## 9. Researched extension — recurrence is an extra resource-allocation axis

An independently constructed model of training cost is $C\approx\kappa NDR$, ignoring unlooped components and implementation details. A hypothetical loss surface might separate data and recurrence effects as $L=L_\infty+AD^{-\alpha}+BR^{-\beta}$ at fixed parameter count. Under a fixed product $DR$, the optimization has the same mathematical structure as Lecture 9's model–data tradeoff.

This does not assert that the toy formula is Parcae's exact fitted law. It shows how a new architecture introduces a new budget allocation question. One can spend compute on more distinct data, more parameters, or more repeated transformation. The answer depends on measured returns and inference constraints.

A strong small-scale project would compare a fixed-depth baseline, a parameter-matched looped model, and a FLOP-matched baseline; sweep recurrence; and report quality, state norms, training stability, and actual latency. Keep inference-time loops within and beyond the trained recurrence range separate. Extra loops can saturate or fail to help.

## 10. Exercises and worked solutions

**Exercise 1.** For $h_{r+1}=0.8h_r+2$ and $h_0=0$, find the limit and the exact state after $R$ steps.

<details><summary>Solution</summary>

$h_R=2(1-0.8^R)/(1-0.8)=10(1-0.8^R)$, converging to 10. A stable homogeneous multiplier does not force the driven state to zero.

</details>

**Exercise 2.** Is $A=\operatorname{diag}(-0.5,-2)$ stable as a discrete recurrence?

<details><summary>Solution</summary>

No. Its spectral radius is 2. The second component grows in magnitude with alternating sign. Negative eigenvalues are not sufficient for discrete-time stability; their magnitudes must be below one.

</details>

**Exercise 3.** Why might a smaller looped model serve faster despite using more arithmetic?

<details><summary>Solution guidance</summary>

If weight movement or cross-device communication dominates, fewer distinct weights may improve locality or fit within a faster memory tier. Reusing them can offset added arithmetic. The opposite can occur when sequential recurrence or cache traffic dominates. Only matched quality and measured workload latency establish the advantage.

</details>

## Primary sources

- [Dan Fu recording](https://www.youtube.com/watch?v=9EEm4iMAF5s): serving-system lifecycle, cache routing, kernel overlap, and looped-model discussion.
- [Parcae](https://arxiv.org/abs/2604.12946) and its [full primary paper](https://arxiv.org/html/2604.12946v1): architecture, stability parameterization, and scaling experiments.
- [Together AI's cache-aware disaggregation report](https://www.together.ai/blog/cache-aware-disaggregated-inference): the serving design discussed in the talk.

The recurrence calculations and numerical examples in this chapter are independent explanatory derivations. They make the scope of the linear argument explicit rather than treating it as a global stability theorem for a nonlinear language model.
