# Lecture 2 — Predicting the cost of a tensor program

Atlas's first training program fits its parameters into memory but fails during backpropagation. Reducing the batch makes it run, although its throughput collapses. A timing experiment then reports an implausibly fast matrix multiplication. These failures have different causes: live training state exceeds the memory budget, small operations reuse weights poorly, and asynchronous execution can make a timer measure submission rather than completion.

The useful object is the **tensor program**: a graph of numerical operations whose arrays have shapes, numerical formats, storage locations, and lifetimes. Atlas needs an account of that graph before it needs a faster accelerator. This chapter develops one such account and checks selected parts with CPU experiments.

## 1. Tensor shape and numerical format determine stored bytes [04:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=270s)

### Begin with the objects that must remain alive

A **tensor**, in this programming context, is a multidimensional array. Its **shape** is the ordered list of axis lengths. Its **dtype** specifies the numerical representation of each entry. Its **device** identifies where operations and storage reside, such as host CPU memory or accelerator memory. Mathematical tensors have a broader coordinate-transformation meaning; the array meaning is the one needed for resource accounting here.

Let an array have dimensions $n_1,\ldots,n_r$, all nonnegative integers, and let $b$ be bytes per entry. Its logical payload is

$$M=b\prod_{j=1}^{r}n_j.$$

The product counts entries; multiplying by bytes per entry supplies the unit. This calculation excludes allocator bookkeeping, alignment, padding, and any separately allocated scale metadata. A $4\times8$ array of four-byte numbers contains 128 bytes of numerical payload. A view of that array can expose a different shape without allocating another 128-byte buffer.

Atlas's program contains several kinds of arrays. **Parameters** are learned coefficients. **Activations** are intermediate values produced from inputs. **Gradients** are derivatives of a scalar loss with respect to parameters or intermediate values. **Optimizer state** is information retained across updates, such as accumulated moments of past gradients. These objects have different lifetimes, which is why counting the parameter file does not determine peak training memory.

A tensor's **strides** specify how far storage addresses advance along each axis. Transposing a matrix can change its shape and strides without moving its entries. A later operation may nevertheless require a contiguous copy. Thus a harmless-looking change in array layout can alter memory traffic even when the logical arithmetic is unchanged. Shape validity and layout efficiency are separate checks.

### Range and resolution spend the same bit budget differently

A floating-point format represents a sign, an exponent, and a significand. The exponent controls the scale of representable magnitudes. Significand bits control spacing between nearby values at a given scale. **Overflow** occurs when a result exceeds the representable range; **underflow** concerns very small results, which may become subnormal values or zero depending on format and execution mode. **Rounding** replaces an exact result by a representable one.

| Format | Stored bits | Exponent bits | Fraction bits | Main implication |
|---|---:|---:|---:|---|
| float32 | 32 | 8 | 23 | Broad range and finer relative resolution |
| float16 | 16 | 5 | 10 | Less range, finer resolution than bfloat16 near one |
| bfloat16 | 16 | 8 | 7 | Broad exponent range with coarse relative resolution |

The sign consumes one additional bit in each row. The table concerns these conventional binary formats, not every possible low-bit representation. Bfloat16's exponent field resembles float32's, but equal exponent-bit counts should not be read as identical representable sets, especially near subnormal limits.

Near one, adjacent normal bfloat16 values are separated by $2^{-7}=0.0078125$. A sufficiently small update can therefore disappear if it is applied directly to a bfloat16 parameter and immediately rounded. Float16 has more closely spaced values near one but can lose small or large magnitudes sooner. The question is not simply whether sixteen bits are “enough”; it is which operation needs which property.

**Mixed precision** uses different formats for different operations or stored states. Matrix products may use low-precision inputs while accumulating more accurately. Reductions, exponentials, or optimizer updates may need different treatment. The [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) document framework-specific autocasting and gradient-scaling patterns. An autocast context is not a proof that every variable, persistent parameter, and optimizer state has the same dtype.

For Atlas, record an explicit precision policy. Counting two-byte parameters, two-byte gradients, and two four-byte Adam moment arrays gives twelve bytes per parameter. Adding a separate four-byte master parameter copy gives sixteen. A framework may instead retain float32 parameters and cast them for selected operations. Inspect the actual tensors before interpreting a memory estimate as a measurement.

## 2. Index notation makes tensor operations auditable [18:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1080s)

### Name the axes that survive and the axes that are summed

Consider an Atlas projection applied to $m$ token positions. Let $X\in\mathbb R^{m\times d}$ hold their input vectors, each with $d$ features. Let $W\in\mathbb R^{d\times k}$ be the learned weight matrix, and let $Y\in\mathbb R^{m\times k}$ be the output. The symbol $\mathbb R^{m\times d}$ denotes real matrices with the given shape. Matrix multiplication means

$$Y_{ij}=\sum_{a=1}^{d}X_{ia}W_{aj}.$$

Here $i$ identifies a token position, $j$ an output feature, and $a$ an input feature being summed. The expression specifies which entries interact. A tensor implementation should preserve that meaning, rather than merely produce an array of the expected shape.

**Einstein summation notation**, or einsum, expresses such contractions by naming axes. In `torch.einsum("md,dk->mk", x, w)`, the `d` axis appears in both operands and is absent from the output, so it is summed. The `m` and `k` axes survive. The [official einsum documentation](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) specifies the implementation's exact broadcasting and ellipsis conventions.

For batched attention-like scores, let $Q$ have shape $(B,T,d)$ and $K$ shape $(B,S,d)$. The batch index identifies independent examples, $T$ and $S$ count positions, and $d$ is the feature dimension. The contraction `btd,bsd->bts` sums only the feature dimension and preserves the batch. Dropping `b` from the output accidentally sums across examples. An output with plausible dimensions can therefore encode the wrong scientific computation.

### Reshaping is a claim about ordering

Suppose a width of eight contains two heads with four features each. A **head** is one separately parameterized group of features or computations. Splitting the last axis into shape $(2,4)$ assigns a meaning to the original storage order. Splitting into $(4,2)$ assigns a different meaning despite preserving the total number of entries.

The einops library makes such groupings readable, but naming dimensions does not eliminate the need to choose the correct order. Test with unequal axis lengths and recognizable entries. A square example full of ones can conceal both a mistaken transpose and an unwanted reduction.

```python
import torch

x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
w = torch.arange(12, dtype=torch.float64).reshape(3, 4)
named = torch.einsum("md,dk->mk", x, w)
assert torch.equal(named, x @ w)
assert named.tolist() == [[20.0, 23.0, 26.0, 29.0],
                          [56.0, 68.0, 80.0, 92.0]]
```

This CPU check has no external effects. It constructs two arrays, computes the same defined operation through two interfaces, and compares the full result to hand-checkable values. The first output entry is $0(0)+1(4)+2(8)=20$. Agreement here checks indexing; it does not demonstrate a GPU speed advantage for einsum.

Atlas can now attach dimensions to every expensive operation. That ledger serves three purposes at once: a shape test, a count of arithmetic work, and a first estimate of data movement. A dimension omitted from the ledger is often where a cost estimate later fails.

## 3. Arithmetic work is not an execution rate [27:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1650s)

### Count the multiplication before reading a spec sheet

One output entry of $Y=XW$ requires $d$ multiplications and $d-1$ additions under a straightforward dot-product algorithm. There are $mk$ outputs, so the exact elementary count is $mk(2d-1)$. The conventional large-dimension approximation is

$$F\approx2mdk.$$

The symbol $F$ denotes floating-point operations, or FLOPs. The factor two counts multiplication and addition separately. It does not count bytes, and it does not assert that two separate hardware instructions execute. Different conventions can be legitimate, but they must match the hardware-rate convention used in a utilization comparison.

For $m=128$, $d=1024$, and $k=4096$, the approximation gives $1{,}073{,}741{,}824$ FLOPs. The weight matrix contains $4{,}194{,}304$ parameters. Applying it to each of 128 vectors explains the same count as roughly two operations per parameter per vector. This second interpretation will become useful when accounting for a full model.

Elementwise operations scale differently. Applying a simple scalar function to every entry of an $m\times k$ tensor requires work proportional to $mk$, although the cost of a comparison, exponential, or division differs. Large matrix multiplications often dominate nominal FLOP counts. An elementwise operation can still dominate time when it repeatedly reads and writes a large tensor.

### Measure completion rather than submission

Let measured elapsed time be $t$ seconds. Achieved arithmetic rate is $F/t$ FLOP/s. Multiplying $F$ by time gives the wrong unit. This corrects a verbal slip in the recording and provides a useful dimensional check for every timing calculation.

Accelerators frequently execute asynchronously: a host call submits work and returns before that work finishes. A host timer around only the submission can severely underestimate execution time. A valid benchmark needs a timing mechanism that includes the required completion boundary, such as appropriate synchronization or device events. Warm-up, compilation, memory allocation, and unrelated queued operations also need an explicit inclusion policy.

Suppose a kernel submits in ten microseconds but completes in two milliseconds. Dividing by submission time exaggerates achieved rate by a factor of 200. No numerical error in the result is required for this performance claim to be wrong. Correct output and correct timing require different tests.

Let $P$ be an applicable hardware peak in FLOP/s, including a specified numerical format and dense or sparse convention. For $G$ identical devices running a workload with model work $F_{\rm model}$ in time $t$, define **model FLOPs utilization** as $F_{\rm model}/(tGP)$. It describes useful model work relative to that declared peak. Work added by recomputation can increase device activity without proportionately increasing this model-work numerator.

Atlas should not copy a utilization target from a different system as a correctness rule. Tiny matrix shapes, memory-bound operations, communication, and a different denominator can all lower the ratio. First establish what the ratio measures. Then explain it from the program's work and traffic.

## 4. Data movement creates a second lower bound [40:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2400s)

### Derive the roofline from two conservation arguments

Let $Q$ be the bytes transferred across a specified memory boundary during an operation, and let $\beta$ be the sustainable or peak bandwidth assigned to that boundary, in bytes per second. The boundary might be device main memory to on-chip storage; using host-to-device bandwidth for the same calculation would answer a different question. Let $P$ again be an applicable arithmetic rate.

Executing $F$ operations cannot take less than $F/P$ seconds at that rate. Moving $Q$ bytes cannot take less than $Q/\beta$ seconds at that bandwidth. Both requirements hold, even if movement and arithmetic overlap. Therefore

$$t\ge\max\left(\frac{F}{P},\frac{Q}{\beta}\right).$$

This is a lower bound. Treating equality as a runtime prediction assumes enough overlap, parallelism, and efficiency to approach both rates without other bottlenecks. Launch overhead, synchronization, cache behavior, and imperfect scheduling can make the actual time much larger.

Define **arithmetic intensity** $I=F/Q$ in FLOPs per byte for $Q>0$. Rearranging the bound gives an upper envelope for achieved rate:

$$\frac{F}{t}\le\min(P,\beta I).$$

For small $I$, the envelope grows linearly with intensity because additional reuse permits more work per byte moved. For sufficiently large $I$, it reaches the compute ceiling $P$. The crossover occurs at $I_*=P/\beta$. A workload below this crossover is memory-bound within the model; one above it can be compute-bound if it achieves the assumed reuse and rate.

**Worked example.** Assign a hypothetical device $P=100\times10^{12}$ FLOP/s and $\beta=10^{12}$ bytes/s. Its crossover is 100 FLOPs/byte. An operation with $F=20\times10^9$ FLOPs and $Q=2\times10^9$ bytes has intensity ten. Its arithmetic lower bound is 0.2 milliseconds, its memory lower bound is two milliseconds, and its optimistic runtime is two milliseconds. Doubling arithmetic peak alone does not improve this bound.

### Explain why batching changes the projection

Assume the matrices $X$, $W$, and $Y$ use $b$ bytes per element and each input is read once while output is written once. The compulsory payload for $Y=XW$ is

$$Q_{\min}=b(md+dk+mk).$$

This is an ideal traffic count, not a guarantee that a kernel reads each value only once. Combining it with the conventional operation count gives $I_{\rm ideal}=2mdk/[b(md+dk+mk)]$.

When $m=1$ and the weight matrix is large, the $dk$ term dominates traffic, so intensity approaches $2/b$. At two bytes per element, that is approximately one FLOP/byte. Each weight is used for one input vector. Increasing $m$ reuses each weight across more vectors and initially raises intensity roughly in proportion to $m$. This is the resource difference between a matrix-vector product and a sufficiently large matrix-matrix product.

For square matrices of side length $n$, ideal traffic is $3bn^2$ and work is approximately $2n^3$, giving $I\approx2n/(3b)$. With two-byte entries, this becomes $n/3$. Increasing dimensions improves reuse only if the implementation organizes the computation to realize that reuse. A matrix multiplication can remain inefficient when its dimensions are small, badly shaped, or distributed across too many devices.

Atlas's smaller microbatch therefore has a double effect. It lowers live activation memory but can reduce weight reuse. Gradient accumulation can restore the number of examples per optimizer update without restoring the size of each individual matrix multiplication. A statistical batch and a hardware batch are related but not interchangeable objects.

The [resource module](cs336_lab/resources.py) makes these assumptions executable:

```python
from cs336_lab.resources import matmul_cost, roofline_seconds

cost = matmul_cost(1, 4096, 4096, bytes_per_element=2)
assert 0.99 < cost.intensity < 1.0
seconds = roofline_seconds(20e9, 2e9, 100e12, 1e12)
assert seconds == 0.002
```

The functions return accounting values and perform no measurement. They reject invalid dimensions or rates. Passing these assertions confirms the arithmetic of the stated model, while a device benchmark would test how closely a real operation approaches it.

## 5. Backpropagation explains the training-work multiplier [57:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3420s)

### Derive both matrix gradients from indexed entries

Let $\ell$ be a scalar loss depending on $Y=XW$. Define the **upstream gradient** $G\in\mathbb R^{m\times k}$ by $G_{ij}=\partial\ell/\partial Y_{ij}$. It records how the loss changes with each output entry. The task is to compute gradients with respect to both inputs and weights.

For one input entry $X_{ia}$, only outputs with the same row $i$ depend on it. Differentiating the indexed matrix product gives $\partial Y_{ij}/\partial X_{ia}=W_{aj}$. Applying the chain rule and summing over affected output features yields

$$\frac{\partial\ell}{\partial X_{ia}}=\sum_{j=1}^{k}G_{ij}W_{aj},
\qquad \nabla_X\ell=GW^\top.$$

For one weight entry $W_{aj}$, every row contributes. Its derivative in output row $i$ is $X_{ia}$. Thus

$$\frac{\partial\ell}{\partial W_{aj}}=\sum_{i=1}^{m}X_{ia}G_{ij},
\qquad \nabla_W\ell=X^\top G.$$

The transpose symbol $\top$ exchanges matrix rows and columns. The formulas can now be checked by shape: $GW^\top$ is $(m,k)(k,d)$, giving $(m,d)$; $X^\top G$ is $(d,m)(m,k)$, giving $(d,k)$. More importantly, the indexed derivation explains which dimension is summed. Shape agreement alone would not establish the chain rule.

Each gradient matrix multiplication performs approximately $2mdk$ FLOPs, just like the forward product. Computing both gradients therefore costs roughly twice the forward arithmetic. If the first input is fixed data and its gradient is not requested, that particular input-gradient computation can be omitted. The factor of three is a repeated-block approximation, not an exact count for every graph boundary.

### From a projection to the 6ND estimate

Suppose a dense model applies parameter-associated matrix products to $D$ training token positions and has $N$ parameters counted in those products. Forward work is approximately $2ND$. Weight and input gradients contribute roughly $4ND$, giving $6ND$ for forward plus backward.

The hinge is repeated use of a learned matrix per token, not merely the existence of $N$ stored parameters. A sparse expert model can store parameters that are not activated for every token. Attention between positions introduces operations dependent on sequence length that are not proportional to a fixed parameter count. Embedding lookup, tied weights, recomputation, and optimizer steps require additional care.

For Atlas, use $6ND$ as a first budget calculation and separately estimate terms likely to matter. Long sequences can make attention's pairwise interactions substantial. Short examples and small models can make vocabulary or overhead terms comparatively large. An estimate that is good at one context length should not silently become a universal law at another.

The following CPU experiment checks the weight-gradient formula against automatic differentiation:

```python
import torch

x = torch.arange(6, dtype=torch.float64).reshape(3, 2)
w = torch.tensor([[0.2], [-0.1]], dtype=torch.float64,
                 requires_grad=True)
y = x @ w
loss = y.square().sum() / 3
loss.backward()
upstream = 2 * y.detach() / 3
assert torch.allclose(w.grad, x.T @ upstream)
expected = torch.tensor([[0.933333333333], [1.133333333333]],
                        dtype=torch.float64)
assert torch.allclose(w.grad, expected)
```

The output entries are $-0.1$, $0.1$, and $0.3$, so the upstream gradient is $(-0.2,0.2,0.6)/3$. Multiplying by $X^\top$ gives $(2.8/3,3.4/3)$, approximately $(0.9333,1.1333)$. Both assertions pass. The full companion test also checks the analytical expression directly, avoiding dependence on a rounded decimal constant. This agreement verifies a local derivative; it does not establish that the full Atlas training objective is appropriate.

## 6. Training memory depends on state and lifetime [67:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4020s)

### Optimizer state is not an optional footnote

**Stochastic gradient descent**, or SGD, updates parameters using an estimated loss gradient. **AdaGrad** additionally accumulates squared gradients and rescales each coordinate's update using that history. **Adam** maintains moving estimates of first and second gradient moments. These are different algorithms, but resource accounting needs to know how many persistent arrays each one requires and their formats.

For $N$ parameters, let $b_w$ be bytes per weight, $b_g$ bytes per gradient, $b_o$ total optimizer-state bytes per parameter, and $b_m$ any separate master-copy bytes. Persistent model-state storage is

$$M_{\rm state}=N(b_w+b_g+b_o+b_m).$$

Under a two-byte weight and gradient convention, two float32 Adam moments contribute eight additional bytes, giving $12N$ bytes. A separate float32 master copy raises this to $16N$. A one-billion-parameter model therefore requires 12 or 16 decimal GB under these two conventions before saved activations and workspaces.

The peak is larger than this persistent subtotal. During backpropagation, inputs needed for weight gradients may remain alive while gradients and temporary outputs are being produced. Memory allocators can reserve more space than currently live tensors occupy. A diagnostic should distinguish allocated tensor payload, allocator reservation, and whole-device occupancy.

In a simplified chain of $L$ layers, each saving one $m\times d$ activation with $b_a$ bytes per entry, saved activation payload is approximately $Lmdb_a$. Increasing microbatch size $m$ grows that term without increasing parameter storage. This explains Atlas's apparent contradiction: the checkpoint fits, yet the training step fails. The checkpoint describes only part of the live graph.

### Gradient accumulation preserves an objective only with correct weights

Divide a desired update batch into microbatches indexed by $r$. Let microbatch $r$ contain $n_r$ selected training items, with individual losses $\ell_{ri}(\theta)$. The desired mean loss over all $n=\sum_r n_r$ items is

$$\bar\ell(\theta)=\frac{1}{n}\sum_r\sum_{i=1}^{n_r}\ell_{ri}(\theta).$$

Linearity of differentiation permits accumulating gradients of each summed microbatch loss divided by the same global denominator $n$. Parameters must remain fixed while the contributions are accumulated. The optimizer then performs one update using the completed gradient.

Averaging microbatch means equally instead gives each microbatch equal weight, irrespective of its size. For microbatches with one and three items, the single item receives weight one half under that rule instead of one quarter under the global mean. This error is easy to hide when all microbatches happen to have equal length.

Atlas's selected items may be tokens rather than sequences. A batch with variable-length documents needs an explicit loss-mask denominator. Changing the denominator changes the objective, even if the code still calls the procedure gradient accumulation. Floating-point summation order, dropout randomness, and operations depending on batch statistics can also prevent bitwise equivalence with a physically larger batch.

Performing an optimizer step after every microbatch is a different procedure. Parameters move between gradient evaluations, and adaptive optimizer states change. Accumulating gradients is not generally equivalent to several small updates with a rescaled learning rate. The companion test uses unequal microbatches to show both the correct equality and the incorrect mean-of-means result.

## 7. Researched extension — recomputation buys a different memory schedule [73:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4380s)

### Derive the checkpoint tradeoff without confusing memory and work

**Activation checkpointing**, also called rematerialization, retains selected intermediate states and recomputes omitted ones when their gradients are needed. The mathematical function being differentiated can remain the same, while the execution schedule changes. Any randomness or mutable state inside recomputed regions must be handled consistently enough to preserve the intended computation.

Consider a chain of $L$ equal-sized layers. Divide it into segments of length $s$, assuming for the moment that $s$ divides $L$. Keep roughly one boundary activation per segment. During backward processing of one segment, reconstruct and temporarily retain its internal activations. In units of one layer's activation size, a simplified peak is

$$A(s)\approx\frac{L}{s}+s.$$

The first term counts retained boundaries; the second counts the currently reconstructed segment. Treating $s$ as continuous, differentiation gives $A'(s)=-L/s^2+1$. Its zero occurs at $s=\sqrt L$, and $A''(s)=2L/s^3>0$ for positive $s$, so this is a minimum of the simplified expression. The minimum is approximately $2\sqrt L$ activation units rather than $L$.

For 64 equal-sized layers, eight-layer segments require roughly eight boundaries plus eight live reconstructed activations: sixteen units. One-layer segments keep many boundaries; one giant segment reconstructs many internal activations at once. The interior optimum emerges from balancing two storage terms.

The work overhead is not automatically a factor of $\sqrt L$. In this segmentwise scheme, each segment can be recomputed once and then backpropagated through while its temporary activations remain available. Total additional forward work is then on the order of one full forward pass. Different schedules that repeatedly restart from the beginning have different costs. The [sublinear-memory training paper](https://arxiv.org/abs/1604.06174) develops this memory–recomputation tradeoff; its assumptions should be checked before applying the simplified chain model to a transformer implementation.

```python
from cs336_lab.resources import checkpoint_slots, model_state_bytes

assert checkpoint_slots(layers=64, segment=8) == 16
assert checkpoint_slots(layers=64, segment=1) == 65
assert model_state_bytes(1_000_000_000) == 12_000_000_000
assert model_state_bytes(1_000_000_000, master=4) == 16_000_000_000
```

These calculations check a storage model, not actual allocator peaks. Layer activations differ in size, residual branches may extend lifetimes, and recomputed kernels require workspaces. The [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html) describes implementation-specific behavior, including variant and randomness considerations. The installed CPU checks use PyTorch 2.11.0; linked stable documentation was resolving to 2.14 during source verification, so version-specific operational claims require checking against the installed version.

For Atlas, recomputation can permit a larger physical microbatch and recover some matrix efficiency. Whether that compensates for extra work is an empirical question. Compare tokens processed per second at the same objective and memory limit, rather than celebrating either lower allocated bytes or higher arithmetic utilization in isolation.

## 8. Limits, exercises, and fully worked solutions

The accounting model describes operations, logical traffic, and retained state under explicit conventions. It omits many implementation effects: cache reuse across operations, fused kernels, communication, launch latency, graph compilation, irregular sparsity, and allocator fragmentation. The CPU examples verify shapes, gradients, and arithmetic relationships. They do not benchmark accelerator performance or reproduce a large training run.

**Exercise 1 — derive a gradient and its cost.** For $X$ of shape $(m,d)$ and $W$ of shape $(d,k)$, derive the weight gradient from an upstream gradient $G$. Explain why its leading arithmetic count matches the forward product.

<details><summary>Worked solution</summary>

An entry $W_{aj}$ contributes to every output $Y_{ij}$ with derivative $X_{ia}$. The chain rule sums contributions from all rows: $\partial\ell/\partial W_{aj}=\sum_iX_{ia}G_{ij}$. Collecting entries gives $X^\top G$, of shape $(d,k)$. There are $dk$ outputs, each reducing over $m$ rows, so work is $dk(2m-1)$ under elementary multiplication-plus-addition counting, approximately $2mdk$. Forward reduction instead occurs over $d$, but the leading product of dimensions is identical. The result assumes both operands participate in the dense computation.

</details>

**Exercise 2 — diagnose a hardware upgrade.** An operation has intensity four FLOPs/byte on a device with $P=200$ TFLOP/s and $\beta=2$ TB/s, using decimal prefixes. Which upgrade doubles its roofline ceiling: twice the arithmetic rate or twice the bandwidth?

<details><summary>Worked solution</summary>

The crossover is $P/\beta=100$ FLOPs/byte. At intensity four, the bandwidth ceiling is $\beta I=8$ TFLOP/s, below the 200-TFLOP/s compute ceiling. Doubling arithmetic peak leaves the minimum at eight. Doubling bandwidth raises it to sixteen, still below compute peak. This doubles the upper bound, not necessarily measured speed: launch or dependency overhead can remain dominant. The example concerns the selected memory boundary and assumes the traffic count remains unchanged.

</details>

**Exercise 3 — expose a loss-normalization error.** One microbatch contains one selected token with loss two; another contains three with losses one, one, and four. Compare the mean of microbatch means with the global token mean.

<details><summary>Worked solution</summary>

The first mean is two and the second is also two, so this particular loss value hides the weighting difference: both aggregation rules report two. That equality does not establish gradient equivalence. Let the respective per-token gradients be one, zero, zero, and zero. The global token gradient is one quarter, while averaging microbatch means gives one half. A test using only loss scalars can miss the bug. Weight each microbatch sum by the common total token count, or weight each mean by its selected-token fraction, to recover the declared global objective.

</details>

**Exercise 4 — test a memory estimate.** A model has 500 million parameters stored with two-byte weights, two-byte gradients, eight bytes of moments, and a four-byte master copy per parameter. It saves 24 activation tensors of shape $(1024,2048)$ in two-byte format. Estimate the subtotal and explain why it is not a peak-memory guarantee.

<details><summary>Worked solution</summary>

Persistent model state is $500\times10^6\times16=8\times10^9$ bytes. Each activation has $1024\times2048\times2=4{,}194{,}304$ bytes. Twenty-four consume $100{,}663{,}296$ bytes, so the subtotal is $8{,}100{,}663{,}296$ bytes, approximately 7.545 GiB. The calculation excludes additional saved operands, temporary gradients, matrix workspaces, outputs, allocator fragmentation, and framework overhead. It also assumes every listed activation is simultaneously live. Measure actual lifetimes to refine the peak estimate.

</details>

**Exercise 5 — distinguish a theorem from a schedule.** Why does the $L/s+s$ checkpoint expression not imply that every network admits a $2\sqrt L$ activation-memory implementation with exactly one extra forward pass?

<details><summary>Worked solution</summary>

The derivation assumes a chain, comparable activation sizes, separable segments, and a backward procedure that can consume each reconstructed segment before releasing it. Networks with branches can keep activations alive across segment boundaries. A single layer may dominate memory, making a layer-count model misleading. Randomness and stateful operations may require additional saved information. Kernel workspaces are not represented by boundary counts. The expression establishes an optimum within one simplified schedule model; applying it to Atlas requires a concrete graph partition and measured execution.

</details>

## Primary sources

- [Official executable Lecture 2](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_02.py): tensor formats, index notation, work accounting, rooflines, and training memory.
- [PyTorch einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html): precise contraction-interface semantics.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): autocasting, scaling, and accumulation considerations.
- [Chen and colleagues: Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174): checkpoint scheduling and memory–work tradeoffs.
- [PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html): implementation behavior and operational boundaries.

The [architecture chapter](lecture-03.html) applies these accounting tools to the components of a transformer.
