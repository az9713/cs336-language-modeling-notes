# Lecture 5 — Explain accelerator performance through data movement

Atlas's training loop produces a puzzle: making a matrix slightly larger makes its multiplication much slower, while padding a different matrix makes it faster. A colleague proposes buying an accelerator with twice the advertised arithmetic throughput. Another proposes storing fewer bits. Neither suggestion explains the observed discontinuities, and neither guarantees that Atlas will run faster.

The central object in this chapter is a computation mapped onto a memory hierarchy. We will track its arithmetic, transferred bytes, temporary storage, and scheduled blocks. These quantities explain different limitations. The running task is Atlas's attention and feed-forward computation; the final construction is an exact attention reduction that avoids storing the full score matrix. CPU experiments verify accounting and numerical identities. They are not measurements of GPU performance.

The recording moves from the accelerator execution model to six practical ideas—control flow, precision, fusion, recomputation, coalescing, and tiling—then combines them in FlashAttention. Hardware examples below identify a specific device or an explicitly simplified model. A historical chip specification is not a specification for every accelerator in its family.

## 1. Throughput comes from parallel work and a memory hierarchy [06:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=360s)

### Separate latency from throughput

Latency is the elapsed time to complete one operation or request. Throughput is the number of operations or requests completed per unit time. A machine can have high throughput while individual tasks wait a long time, because many tasks overlap. GPU design emphasizes abundant parallel arithmetic and keeping many execution contexts available; CPU design typically spends more resources on fast execution of complex, branching instruction streams. These are design tendencies, not a claim that CPUs cannot process vectors or GPUs cannot branch.

Let a computation require $F$ floating-point operations, move $Q$ bytes across a specified memory boundary, and run on a device with applicable arithmetic rate $P$ FLOP/s and bandwidth $\beta$ bytes/s. Chapter 2 introduced the lower bound

$$t\ge\max(F/P,Q/\beta).$$

The arithmetic intensity is $I=F/Q$, in FLOP per byte. Dividing work by the time bound gives a throughput ceiling $\min(P,\beta I)$. The crossover intensity is $I_*=P/\beta$. This is the roofline model: performance is limited by either arithmetic or movement at the chosen boundary, with additional constraints able to lower it further.

For an illustrative device with $P=100$ trillion FLOP/s and $\beta=1$ trillion bytes/s, $I_*=100$ FLOP/byte. A computation with intensity 10 cannot exceed 10 trillion FLOP/s under this model, even though the device's arithmetic ceiling is ten times larger. Doubling only $P$ leaves that bound unchanged. Reducing transferred bytes can matter more than increasing arithmetic capability.

### Name the memory being counted

A streaming multiprocessor, or SM, is a GPU execution unit containing schedulers, arithmetic resources, registers, and access to nearby memory. High-bandwidth memory, or HBM, holds large arrays outside the main logic die. On-chip caches and programmer-managed shared memory provide smaller, faster storage. Registers hold thread-local working values and addresses. The same logical tensor may move among these locations during one operation.

“Local” has a specific and potentially misleading CUDA meaning. A thread's local-memory address space is private to that thread but is backed by device memory, with caching; it is not synonymous with on-chip shared memory. Register spills can therefore create extra memory traffic. Shared memory is explicitly used for cooperation within a thread block, whereas caches automatically retain selected memory accesses. [CUDA's SIMT programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) specifies these distinctions.

The relevant $Q$ depends on the boundary. Reusing a value from shared memory can reduce HBM traffic without reducing shared-memory traffic. Counting only the final input/output tensor sizes can miss repeated reads, intermediates, and spills. Conversely, assuming every source-level read reaches HBM can overcount because caches serve some accesses.

This explains the role of the hierarchy: expensive nearby storage is used to reuse data supplied by larger storage. More arithmetic units help only when the computation supplies enough independent work and data. Neither a memory-capacity number nor an arithmetic-rate number describes that complete mapping.

For Atlas, record the dtype, operation shape, memory boundary, and whether rates are theoretical or measured whenever using a roofline estimate. A ratio constructed from dense floating-point work and a peak rate that assumes structured sparsity is not a valid utilization measure. The numerator and denominator must describe compatible arithmetic.

## 2. Map the program onto threads, blocks, and matrix units [14:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=840s)

### A block is a cooperation unit; a warp is an execution group

A kernel is a program launched across many GPU threads. Threads are organized into blocks, and blocks form a grid. Threads in a block can cooperate using shared memory and supported synchronization. In the conventional execution model, a block executes on one SM, though an SM can host multiple blocks subject to resource limits. The grid can contain many more blocks than fit simultaneously.

On NVIDIA GPUs, a warp groups 32 threads. Single-instruction, multiple-thread execution, abbreviated SIMT, issues work for active threads in such groups. The thread-block size is selected by the programmer or compiler within device limits; it is not a universal fixed number of warps. A block of 128 threads contains four warps. The [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) defines the hierarchy and its cooperation boundaries.

Resource use limits residency. If a block needs many registers or much shared memory, fewer blocks may fit on an SM. Occupancy measures resident active warps relative to a hardware maximum. More occupancy can help hide waits, but it is not identical to utilization or performance: a well-tiled kernel may profitably use more resources per block and run with lower occupancy.

### Divergence spends execution slots on different paths

Suppose half the threads in one warp require a costly branch A and the rest require branch B. If both paths must be executed with different active masks, the warp spends time on A while B's threads are inactive, then on B while A's threads are inactive. This is control divergence. If all threads take the same branch, that particular divergence penalty is absent.

As a simple slot model, let both paths contain $c$ arithmetic instructions and let 16 threads take each path. The useful work is $32c$ thread-instructions, while executing both masked paths provides $64c$ possible slots. Utilization of those slots is one half. Unequal path lengths, predication, reconvergence, and modern scheduling change the details, so this is an explanatory model rather than an exact cycle prediction.

The lesson is to avoid unnecessary disagreement within execution groups, not to delete every `if`. A uniform branch can be cheap; a compiler may convert a small conditional into predicated instructions. Replacing a conditional with unconditional evaluation can also execute invalid arithmetic that a branch would have avoided. Preserve semantics and inspect generated behavior before claiming a faster version.

### GPUs and TPUs share a workload, not identical terminology

Matrix-multiply units accelerate dense multiply-accumulate operations. On NVIDIA GPUs, Tensor Cores are such units within the SM architecture. In Google's TPU terminology, a TensorCore is a larger processing unit containing matrix-multiply units, or MXUs, plus vector and scalar resources. [Google's TPU architecture documentation](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) describes this organization. The identical word “core” should not be used to compare counts without identifying what is counted.

Both platforms combine matrix arithmetic, non-matrix operations, and a memory hierarchy, but differ in unit sizes, compiler interfaces, and interconnects. Their internals need not be identical circuits for the same high-level reasoning to apply. The transferable question is whether Atlas's operation supplies matrix-shaped work with enough reuse; the device-specific questions concern supported shapes, layouts, precision modes, and communication.

The recording's matrix-throughput curves therefore have several possible causes: insufficient work, excess bytes, awkward tiling, or poor scheduling. Before optimizing, identify which part of the hardware model a change is supposed to improve. Otherwise an apparent speedup on one shape can become a regression on another.

## 3. Lower precision reduces bytes but changes a numerical contract [35:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=2100s)

### Count traffic and arithmetic separately

Consider an elementwise transformation that reads $n$ numbers and writes $n$ results. If each number occupies $b$ bytes, its compulsory input/output traffic is $2nb$ bytes. For a ReLU modeled as one comparison-like operation per element, intensity is $1/(2b)$ operations per byte. At four bytes it is $1/8$; at two bytes it is $1/4$. Whether the hardware counts that comparison as a FLOP does not change the traffic argument.

Halving storage can halve this ideal traffic, but casting, scaling, packing, and extra reads can reduce the end-to-end benefit. Matrix units may also provide a higher arithmetic rate for lower-precision operands. These are two different gains: less movement and faster supported arithmetic. Either can be irrelevant if another part of the pipeline dominates.

Mixed-precision arithmetic assigns different formats to operands, accumulators, outputs, and reductions. A matrix product may multiply low-precision inputs while accumulating into a wider format. This protects the sum from some rounding error; it does not restore information already lost when inputs were quantized. Softmax exponentials, normalization reductions, and router calculations can require different precision policies from large matrix multiplications.

### A block scale couples nearby values

In a generic scaled quantizer, a real value $x_i$ is represented approximately as $s_b q_i$, where $q_i$ belongs to a low-precision format and $s_b$ is a scale shared by block $b$. Choosing a scale from the block's largest magnitude protects range but can coarsen small values. A smaller block can isolate outliers at the cost of storing more scales.

For a transparent toy integer quantizer with values $q_i\in\{-7,\ldots,7\}$, use $s=\max_i|x_i|/7$ and round $x_i/s$ to the nearest integer. Quantizing $(0.1,0.2,10)$ together gives scale approximately 1.429, so the first two values round to zero. Quantizing the first two separately with scale $0.2/7$ retains much finer resolution. This is not an FP4 implementation; it isolates the shared-scale mechanism without requiring accelerator hardware.

[NVIDIA's MXFP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html) specifies one E8M0 scale per 32 elements. E8M0 denotes an exponent-only scale representation. Its blockwise structure means that quantizing rows and quantizing columns generally produce different groupings and different rounding. Transposing an already quantized layout need not give the representation expected by an operation requiring scales along the other direction. Implementations can prepare both orientations from a higher-precision source.

[NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) describes 16-element blocks with E4M3 block scales and additional scaling details. This identifies the format behind the recording's 16-element FP4 example; it should not be relabeled MXFP4. Format names encode materially different storage and scaling contracts.

### Metadata and orientation consume part of the saving

For MXFP8's one-byte elements and one-byte scale per 32 elements, an ideal packed block uses 33 bytes for 32 values, or 8.25 bits per value. Against 16-bit storage, its ratio is $8.25/16\approx0.516$, before padding and other state. If an implementation also retains a separately quantized transposed orientation, those two copies alone use 16.5 bits per logical value. This does not negate its compute benefit; it explains why “8-bit training” does not mean every training allocation is half the previous size.

For Atlas, evaluate loss and throughput with the actual precision recipe. Record which tensors stay wider, how scales are selected, whether both orientations are stored, and whether calibration statistics change with data. A high arithmetic rate on a quantized matrix multiplication is compatible with a small full-step gain when quantization and other operators dominate.

## 4. Fuse operations and recompute values when movement is dearer [47:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=2820s)

### Fusion removes intermediate trips through memory

Take a pointwise computation $y=(\sin x)^2+(\cos x)^2$ over $n$ elements. It is a convenient graph for counting, though mathematically the exact expression equals one. Assume a naive implementation launches separate kernels for sine, cosine, each square, and addition, materializing every intermediate in global memory.

Sine and cosine each read $x$ and write an output: four element transfers. The squares each read an input and write an output: four more. Addition reads two squared values and writes the result: three more. The total is $11n$ element transfers. A fused implementation can read each $x$ once, keep intermediate values nearby, and write each result once, reducing the ideal count to $2n$. These counts assume no beneficial cache reuse and ignore compiler simplification; they make the eliminated materializations explicit.

The algorithm still performs the elementary functions unless it applies a valid mathematical simplification. Under finite precision, domains with NaNs or infinities complicate replacing the expression by a constant. A compiler cannot freely change observable numerical behavior simply because an identity holds over finite real numbers. Fusion ordinarily combines execution while preserving the intended operations and their permitted numerical tolerance.

Fusion can also reduce kernel-launch overhead, the cost of submitting and starting work. But a larger fused kernel may need more registers, spill intermediates, reduce occupancy, or prevent use of a specialized library kernel. Fewer launches is not itself a proof of lower runtime. For Atlas, fuse small compatible operations around a large matrix product only when the resulting implementation improves the relevant measurements.

### Recomputation trades arithmetic for saved traffic

Backpropagation needs selected forward values to evaluate derivatives. Storing them writes memory during the forward pass and reads it during the backward pass. If an activation containing $A$ bytes can instead be reconstructed using $C$ extra operations, the simplified trade compares $2A/\beta$ seconds of avoided traffic with $C/P$ seconds of extra compute.

Under this isolated model, recomputation is promising when $C/P<2A/\beta$, or $C/(2A)<P/\beta$. The left side is extra operations per avoided byte. The right side is the machine's compute-to-bandwidth ratio. If computation is plentiful relative to bandwidth, even doing additional arithmetic can improve wall time.

For $A=100$ million bytes, $C=1$ billion operations, $P=100$ trillion operations/s, and $\beta=1$ trillion bytes/s, the saved traffic time is 0.2 ms and recomputation time is 0.01 ms. This favorable comparison ignores overlap, caches, launch overhead, and the actual bottleneck. It is a screening calculation, not a benchmark result.

Checkpointing from Chapter 2 groups this trade across a larger computation graph. Here the same principle applies inside a fused kernel: reconstruct a small intermediate while its inputs are already nearby. The reduction in peak memory can also enable a larger batch, producing a second effect that should be measured separately from the direct kernel saving.

Recomputation must reproduce the needed forward function. Random masks, stateful updates, and mutable inputs require care; rerunning them differently can change the gradient. The memory optimization has a correctness contract as well as a performance hypothesis. In FlashAttention, the recomputed quantities are deterministic score and probability tiles under the specified inputs and masking, allowing the backward pass to avoid storing a full quadratic activation.

## 5. Count memory transactions, not only requested elements [53:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=3180s)

### Coalescing depends on the addresses issued together

A group of threads can request nearby memory locations, allowing their requests to be served with fewer aligned transactions. This is coalescing. The exact rules depend on architecture, operation width, and cache behavior. [CUDA's best-practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) describes coalesced global access and the importance of alignment. A 128-byte span need not correspond to one indivisible transaction in every implementation.

Use a simple sector model: each transfer touches aligned regions of $s$ bytes. For a byte address $a$ and element size $b$, the element spans sector indices from $\lfloor a/s\rfloor$ through $\lfloor(a+b-1)/s\rfloor$. The number of distinct indices touched by a warp's requests is a proxy for its transaction requirement. This model counts addresses, not DRAM timing or cache hits.

Suppose 32 threads read consecutive four-byte values beginning at byte zero and sectors are 32 bytes. Their 128 requested bytes occupy four sectors. Shift the starting address by four bytes: the same number of requested bytes now spans five sectors. Instead give each thread an address 1,024 bytes apart: the 32 requests touch 32 sectors, even though useful data is still 128 bytes.

The cumulative `memory_sectors` helper accepts nonnegative integer byte offsets and positive element/sector sizes. It returns the number of aligned sectors touched, including elements that straddle a boundary, without allocating GPU memory. Invalid offsets or dimensions raise `ValueError`.

```python
from cs336_lab.resources import memory_sectors

contiguous = memory_sectors(range(0, 128, 4))
shifted = memory_sectors(range(4, 132, 4))
strided = memory_sectors([1024 * lane for lane in range(32)])
assert (contiguous, shifted, strided) == (4, 5, 32)
print(contiguous, shifted, strided)
```

The output `4 5 32` makes the distinction between useful bytes and transaction bytes concrete. Cache reuse, broadcast rules, and the memory controller can change realized traffic, so this proxy should guide profiling rather than replace it.

### A transpose changes layout and grouping

For a row-major matrix with $n$ columns and element size $b$, address $(i,j)$ is $a_0+b(in+j)$, where $a_0$ is the base address. Adjacent columns in one row differ by $b$ bytes; adjacent rows in one column differ by $nb$. Assigning consecutive lanes to columns can therefore coalesce a row read, while assigning them to rows can scatter a column read. The physical addresses, not the names “row” and “column,” determine the outcome.

A tensor transpose can be a view that changes indexing strides without physically moving data. A subsequent kernel may handle those strides efficiently, choose another implementation, or require a contiguous copy. The cost of making such a copy belongs to the end-to-end operation if it was required solely for that operation. Timing only the final multiply can conceal an expensive layout conversion.

Padding can align successive rows and improve supported matrix shapes. For Atlas's vocabulary projection, however, adding storage columns is not the same as adding valid vocabulary entries. If padded logits participate in softmax, they change the denominator and therefore the model. A correct implementation must exclude unused entries or otherwise preserve the original probability domain. This is an example of a systems optimization whose semantic boundary is visible at the modeling level.

## 6. Reuse tiles, then account for partially occupied waves [58:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=3480s)

### Derive the benefit of local reuse

Consider multiplying square matrices $A,B\in\mathbb R^{n\times n}$ to produce $C$. A naive output-by-output algorithm reads $n$ elements from each input for each of $n^2$ outputs: roughly $2n^3$ input-element reads. Real caches can reduce that count, but the naive schedule exposes little explicit reuse.

Tile the output into $b_t\times b_t$ blocks, with $b_t$ dividing $n$ for this derivation. For one output tile and one step along the inner dimension, load one $b_t\times b_t$ tile from each input. Those $2b_t^2$ elements support $2b_t^3$ multiply-add operations, counting a multiplication and addition separately. Each loaded value is reused across $b_t$ outputs in the other direction.

There are $(n/b_t)^2$ output tiles and $n/b_t$ inner steps, so input reads become $2n^3/b_t$ elements. Output writes add $n^2$ elements. For element size $b_e$ bytes, the leading intensity from input traffic is approximately $b_t/b_e$ FLOP/byte. Increasing tile width increases reuse until fast-memory capacity, registers, occupancy, or implementation constraints intervene.

At $b_t=32$ with two-byte inputs, the input-traffic intensity is about 16 FLOP/byte. At $b_t=128$, it is about 64. The input tiles alone need $2b_t^2b_e$ bytes; the output accumulator and any buffering require more. There is no free choice of arbitrarily large tiles. Holding an output accumulator in registers instead of shared memory changes the resource constraint but does not remove it.

### Edges waste some tile work

For an $m\times n$ output and tile dimensions $b_m,b_n$, the tile count is $N_t=\lceil m/b_m\rceil\lceil n/b_n\rceil$. An elementary area-efficiency measure is $mn/(N_tb_mb_n)$. It is one for a perfect fit and smaller when edge tiles contain many masked positions. Actual kernels can use specialized edge handling, so this is a shape diagnostic rather than a universal utilization formula.

A 256-by-256 output with 128-by-128 tiles uses four tiles. Increasing both dimensions to 257 requires nine tiles; many newly allocated positions are outside the real matrix. The arithmetic workload changed only slightly, but the fixed tile schedule changed abruptly. An autotuner may choose another tile shape to avoid that cost, subject to the reuse and resource tradeoffs above.

### Waves explain a second discontinuity

Suppose a simplified scheduler runs one equal-duration tile block per SM at a time, with $S$ SMs. It needs $\lceil N_t/S\rceil$ waves. Average slot utilization in that model is $N_t/[S\lceil N_t/S\rceil]$. A small increase in tiles can add a mostly empty final wave.

The recording's example uses square outputs of size 1,792 and 1,793 with tiles 256 by 128. Counts are $7\cdot14=98$ and $8\cdot15=120$. On a modeled 108-SM device, one wave becomes two. The average slot fraction falls from $98/108\approx0.907$ to $120/216\approx0.556$. [NVIDIA's performance background guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) identifies 108 SMs for its A100 example, resolving the inconsistent earlier counts in the recording.

```python
from math import ceil


def waves(n, rows=256, columns=128, sms=108):
    tiles = ceil(n / rows) * ceil(n / columns)
    count = ceil(tiles / sms)
    return tiles, count, tiles / (sms * count)


assert waves(1792)[:2] == (98, 1)
assert waves(1793)[:2] == (120, 2)
print(round(waves(1792)[2], 3), round(waves(1793)[2], 3))
```

This prints `0.907 0.556`. The function is an isolated scheduling calculation for the stated positive dimensions, not a hardware simulator. Real SMs can host multiple blocks, edge tiles can take different times, and kernel selection can change. The calculation explains one plausible mechanism behind a cliff; a measured trace is needed to identify the mechanism for Atlas's actual kernel.

## 7. Build an online softmax without retaining every score [72:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=4320s)

### Retain the sufficient quantities for one output row

For one query $q\in\mathbb R^{d_k}$, allowed keys $k_j\in\mathbb R^{d_k}$, and values $v_j\in\mathbb R^{d_v}$, define scores $z_j=q^\top k_j/\sqrt{d_k}$. Attention returns $o=\sum_j e^{z_j}v_j/\sum_j e^{z_j}$. A stable dense implementation subtracts $m=\max_j z_j$ before exponentiating. The challenge is that the final maximum and denominator appear to require seeing all keys before computing the weighted output.

An online algorithm maintains three quantities for processed keys: the maximum $m$, the denominator $\ell=\sum_j e^{z_j-m}$, and the unnormalized vector $u=\sum_j e^{z_j-m}v_j$. The current output is $u/\ell$. Suppose a new block has maximum $m_b$, denominator $\ell_b$, and numerator $u_b$, each using its own maximum. Set $m'=\max(m,m_b)$. Then

$$\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b,\qquad
u'=e^{m-m'}u+e^{m_b-m'}u_b.$$

The factors convert both summaries to the same exponential reference point. To verify the first term, $e^{m-m'}\sum_j e^{z_j-m}=\sum_j e^{z_j-m'}$; the numerator follows identically with each value multiplied in. The algorithm has not approximated softmax or discarded a key. It has retained exactly the quantities required to combine processed blocks, up to floating-point rounding.

[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) develops the running normalization idea. Extending the summary with a weighted numerator gives the attention reduction used in our teaching implementation. The important condition is to rescale the old numerator as well as the old denominator when the maximum increases. Rescaling only one changes the output.

### Check a maximum that changes dramatically

Take two scalar scores 1,000 and 1,001 with values 2 and 8. Directly exponentiating the raw scores can overflow. After the first item, $(m,\ell,u)=(1000,1,2)$. The next maximum is 1,001, so old terms are scaled by $e^{-1}$. The final output is $(2e^{-1}+8)/(e^{-1}+1)\approx6.38635$, comfortably within the value range. A stable implementation must return this result without ever forming $e^{1000}$.

Our cumulative `streaming_attention_row` accepts a finite query and nonempty matching key/value sequences, plus a positive integer chunk size. It returns a tuple of output components and does not mutate its inputs. Shape errors and nonfinite inputs are rejected. It is a scalar CPU forward reference, without automatic differentiation, dropout, or an implicit causal mask; pass only the allowed key/value prefix for a causal row. Arithmetic still has the finite-range limits of Python floating-point values.

```python
import torch
from cs336_lab.attention import streaming_attention_row

q = [1.0]
k = [[1000.0], [1001.0]]
v = [[2.0], [8.0]]
scores = torch.tensor([1000.0, 1001.0], dtype=torch.float64)
values = torch.tensor([2.0, 8.0], dtype=torch.float64)
expected = torch.softmax(scores, 0) @ values
for chunk_size in (1, 2, 10):
    actual, = streaming_attention_row(q, k, v, chunk_size)
    assert abs(actual - expected.item()) < 1e-12
print(round(actual, 5))
```

It prints `6.38635`. Varying chunk size tests that changing the grouping preserves the result. The full test suite also compares causal prefixes and changing maxima against an independent dense softmax. Different reduction orders need not be bitwise identical; use a justified numerical tolerance.

### Combine the summary with tiling and recomputation

[FlashAttention](https://arxiv.org/abs/2205.14135) organizes query/key/value tiles so score tiles and probability work remain in fast memory while small row summaries carry information between tiles. The output and selected row statistics can be retained without writing a complete $T\times T$ score or probability array to HBM. Backward computation reconstructs needed tiles from saved inputs and statistics.

The main attention computation has two large matrix products, $QK^\top$ and the probability-weighted multiplication by $V$, with normalization between them. Avoiding a stored quadratic intermediate does not remove the quadratic number of pairwise scores. “Exact attention” here distinguishes the mathematical function from sparse or recurrent alternatives; finite-precision evaluation order can still change the final bits.

For Atlas, this synthesis is the payoff of the hardware model. Tiling creates reuse; fusion keeps related work together; online normalization removes a global materialization requirement; recomputation reduces backward storage. None of these requires changing the learned attention function. Their combination can outperform a superficially cheaper-looking computation whose memory access is poorly organized.

## 8. Limits, exercises, and worked solutions

Roofline and sector counts are simplified models, not measurements. Tile and wave calculations depend on explicit scheduling assumptions. Low-precision storage calculations exclude some implementation state. The CPU attention reference verifies a forward identity but does not implement FlashAttention's GPU kernel or backward pass. To establish a performance result, benchmark representative shapes and dtypes after warm-up, synchronize correctly, include required conversions, and retain correctness checks against a reference.

**Exercise 1 — identify an ineffective upgrade.** A kernel performs $10^{12}$ operations and transfers $10^{11}$ bytes on a device with $P=10^{14}$ operations/s and $\beta=10^{12}$ bytes/s. Compute the roofline time bound. What happens if only arithmetic throughput doubles?

<details><summary>Worked solution</summary>

Compute time is $10^{12}/10^{14}=0.01$ seconds; transfer time is $10^{11}/10^{12}=0.1$ seconds. The bound is 0.1 seconds. Doubling $P$ lowers the compute term to 0.005 seconds while the transfer term remains 0.1, so the bound does not improve. Intensity is 10 operations/byte, below the original ridge point of 100. Reducing traffic or increasing applicable bandwidth addresses the modeled bottleneck. Actual runtime can exceed the bound because of launch, synchronization, layout, or other execution costs.

</details>

**Exercise 2 — count a strided load.** Thirty-two threads each read one four-byte value from a row-major matrix with 256 columns. Compare consecutive columns of the same aligned row with the same column in consecutive rows using 32-byte sectors.

<details><summary>Worked solution</summary>

Consecutive columns have addresses $a_0+4j$, for $j=0,\ldots,31$. With sector-aligned $a_0$, they occupy four sectors. Consecutive rows have addresses $a_0+4(256i+j_0)$ for fixed column $j_0$. Addresses differ by 1,024 bytes, so they touch 32 distinct sectors. Both request 128 useful bytes, but the second touches 1,024 sector bytes under this model. That eightfold ratio is a transaction-efficiency comparison, not necessarily an eightfold runtime difference because caches and other bottlenecks can intervene.

</details>

**Exercise 3 — find the missing quantization storage.** An array contains exactly 3,200 values. Store them in MXFP8 with one byte per value and one one-byte scale per 32 values. Compare the packed size with 16-bit storage, then include a separately quantized transposed copy.

<details><summary>Worked solution</summary>

The element payload is 3,200 bytes, and 100 blocks require 100 scale bytes. One orientation uses 3,300 bytes compared with 6,400 bytes at 16 bits. Two independently quantized orientations use 6,600 bytes, before padding or a higher-precision source retained for training. This does not imply the quantized computation is pointless: it can still use faster matrix arithmetic and lower operand traffic for each operation. It means persistent storage and arithmetic throughput must be counted separately rather than inferred from the nominal element precision.

</details>

**Exercise 4 — diagnose a performance cliff without overclaiming.** Under the one-block-per-SM equal-time model, compute tile counts and waves for square sizes 1,792 and 1,793, tiles 256 by 128, and 108 SMs. Name two reasons a measured device might depart from this model.

<details><summary>Worked solution</summary>

The first shape uses $\lceil1792/256\rceil\lceil1792/128\rceil=7\cdot14=98$ tiles and one wave. The second uses $8\cdot15=120$ tiles and two waves. The model predicts a discontinuity because the last twelve tiles require another wave. A real SM may host multiple blocks, which changes the number of simultaneous slots. Edge tiles may do less work, so wave durations need not match. Kernel autotuning can also select another tile size or algorithm. The calculation offers a causal hypothesis that profiling can test, not a universal twofold slowdown prediction.

</details>

**Exercise 5 — repair an online-normalization bug.** A streaming attention implementation updates its running maximum and denominator correctly but leaves the accumulated numerator unchanged when a larger maximum appears. Explain the error using scores 0 and 1 with values 10 and 0.

<details><summary>Worked solution</summary>

After the first entry, the summary is $(m,\ell,u)=(0,1,10)$. When score 1 arrives, the new reference maximum is 1. The old denominator must become $e^{-1}$ and the old numerator must become $10e^{-1}$. Adding the new entry gives denominator $e^{-1}+1$ and numerator $10e^{-1}$, so the correct output is $10/(1+e)\approx2.6894$. Leaving the old numerator at 10 instead yields $10/(1+e^{-1})\approx7.3106$. The bug combines numerator and denominator expressed relative to different maxima. Both must be rescaled together.

</details>

**Exercise 6 — protect the modeling boundary.** Atlas pads its output vocabulary projection from $V$ real tokens to $V+r$ storage columns for faster multiplication. Why can this change validation loss even if all original logits are unchanged, and what must the implementation preserve?

<details><summary>Worked solution</summary>

Softmax divides by the sum of exponentials over all included columns. If padded columns participate, their positive exponential contributions increase the denominator, lowering every original token probability. Cross-entropy for a real target therefore rises even though its logit is unchanged. Exclude padded columns from the probability calculation or mask them so they contribute zero probability, while preserving target-ID mapping. The optimization should alter physical storage or arithmetic scheduling while retaining the original vocabulary domain. Timing a version with extra valid tokens would compare different models.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=izZba4UA7iY) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_05.pdf): hardware progression, optimization mechanisms, and the attention synthesis.
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [SIMT kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html), and [best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): execution, memory, and coalescing definitions.
- [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) and [NVIDIA performance background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html): accelerator terminology and the A100 scheduling example.
- [MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html) and [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html): current official format and implementation documentation, checked during this rewrite.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) and [FlashAttention](https://arxiv.org/abs/2205.14135): stable streaming normalization and exact attention organized around memory movement.

The traffic calculations, sector simulator, simple wave model, Atlas padding example, and CPU numerical checks are instructional additions. Live documentation can change; the internal research ledger records the retrieved source identities.
