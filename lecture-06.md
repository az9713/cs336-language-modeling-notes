# Lecture 6 — From tensor expressions to GPU kernels

The recording turns the previous lecture's hardware model into executable reasoning. It covers blocks and warps, memory-access constraints, measurement, a fused activation, row-wise softmax and reductions, and tiled matrix multiplication. Although the playlist title includes XLA, the detailed demonstrations in this recording center on PyTorch and Triton; the XLA discussion below is explicitly an extension.

## 1. Choose the unit of work before writing arithmetic [03:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=180s)

At the tensor level, a statement describes a complete operation. At a lower level, a kernel must decide which part each program instance owns. A useful design begins with three questions: which output region does one instance produce, which inputs does it need, and can another instance write the same locations?

For an elementwise operation on $N$ entries, let each instance own a block of $B$ consecutive indices. Instance $p$ handles offsets

$$i=pB+j,\qquad j=0,\ldots,B-1.$$

Only indices $i<N$ are valid. A mask is necessary because $N$ may not be a multiple of $B$. The number of instances is $\lceil N/B\rceil$.

**Worked example.** For $N=1000$ and $B=256$, launch four instances. The last owns logical indices 768 through 1023, but only 232 entries are valid. Masking the load and store prevents out-of-bounds access. It does not permit an invalid pointer dereference before the mask is applied.

Triton presents arrays of offsets and block operations, while the compiler maps them to threads, warps, registers, and machine instructions. It is useful to reason in tiles without pretending each source-level tensor lives in one particular physical memory space. Inspect generated code and profiles when placement matters.

## 2. Coalescing, bank conflicts, and occupancy are separate constraints [12:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=720s)

Coalescing concerns how global-memory accesses combine into transactions. Shared-memory bank conflicts concern simultaneous accesses to addresses mapped to the same bank. A layout can improve one while harming the other. Padding or swizzling a shared-memory tile can change bank mapping without changing the mathematical tensor.

**Occupancy** describes how much of the device's thread/warp capacity is resident. It is limited by registers, shared memory, block limits, and thread limits. Higher occupancy can hide latency, but it is not automatically higher performance. A kernel already saturating a matrix unit may gain nothing from extra resident warps; a register-heavy kernel may be faster at lower occupancy if it avoids spills.

**Worked example.** Suppose an SM has a hypothetical 64 KiB shared-memory budget. A block using 40 KiB allows at most one such block resident on that constraint; reducing usage to 32 KiB allows two. That discrete change can improve latency hiding. But if the smaller tile requires twice as many HBM reads, the total kernel may still be slower. An isolated occupancy number cannot decide the tradeoff.

## 3. Benchmarking answers how fast; profiling explains where time goes [22:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1320s)

A benchmark measures the target workload under defined conditions. A profiler reveals operations, kernel launches, data movement, and overlap. Neither substitutes for the other. A trace can identify many small kernels, but a clean benchmark is needed to establish the effect of fusing them.

Separate compilation, warmup, allocation, host-to-device transfer, and steady-state execution according to the question being asked. If the application repeatedly reuses tensors, measure that case. If it receives fresh inputs over a link, include the transfer cost in an application benchmark. Synchronization is required to avoid timing only asynchronous submission.

Correctness comes before performance. Compare the custom kernel with a reference across representative shapes, including awkward dimensions, dtypes, extreme values, and noncontiguous inputs if supported. State numerical tolerances and the supported domain. A faster kernel that silently assumes contiguous memory is not a drop-in replacement for a general operator.

## 4. Fusing an activation removes intermediate trips to memory [30:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1800s)

The approximate GELU used in many implementations is

$$\operatorname{GELU}(x)\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right].$$

Writing this as several eager tensor operations can create multiple kernels and intermediates. A fused implementation loads $x$, evaluates the expression locally, and writes the result. A compiler may fuse the expression automatically; a handwritten kernel is not guaranteed to outperform a mature built-in implementation.

Here is the ownership pattern in pseudocode rather than a version-specific Triton API:

```text
offsets = program_id * BLOCK + arange(BLOCK)
valid = offsets < N
x = masked_load(input + offsets, valid, default=0)
y = gelu_formula(x)
masked_store(output + offsets, y, valid)
```

The mask is a correctness condition. The block size is a performance parameter. The mathematical formula defines the operator. Keeping these roles distinct makes debugging far easier.

**Worked example.** A vector with $N=2^{20}$ float32 entries requires about 8 MiB for one full input read and output write. If an eager expression creates five read/write passes, nominal traffic can be several times larger. A fused version approaches the minimum only if its compiler does not spill excessive temporaries and the memory system behaves favorably.

## 5. Row softmax requires a reduction and a safe padding value [57:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3420s)

For row $x\in\mathbb R^n$, stable softmax computes maximum $m=\max_jx_j$, exponentials $e_j=e^{x_j-m}$, sum $s=\sum_je_j$, and output $e_j/s$. Assigning one program to a row allows these intermediates to remain local when the row fits.

For a padded block, invalid entries should act as $-\infty$ before the maximum and exponential. Padding with zero is wrong: zero can exceed every valid negative logit and contributes a nonzero exponential to the denominator.

**Worked example.** A row contains logits $(-2,-3)$ and is padded to width four. Correct probabilities are approximately $(0.731,0.269)$. Treating the two padded entries as zero gives denominator $e^{-2}+e^{-3}+2$, shrinking both valid probabilities and making their sum less than one. Masking only the final store does not repair the incorrect reduction.

An entirely masked row needs an explicit convention. Subtracting $-\infty$ from $-\infty$ creates an undefined floating-point result. Attention implementations should ensure each valid query has a valid key or define a safe output for empty rows.

## 6. Large reductions and matrix multiplication require tiling [65:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3900s)

If a row is too large for an efficient single local reduction, process it in chunks and combine partial sums or stable softmax summaries. Larger blocks increase work per instance but can increase register pressure. A hierarchical reduction makes those tradeoffs explicit.

For matrix multiplication $C=AB$, let an output tile have dimensions $B_M\times B_N$ and reduction chunks have width $B_K$. For each reduction chunk, load an $A$ tile of shape $B_M\times B_K$ and a $B$ tile of shape $B_K\times B_N$, multiply, and accumulate into the output tile. Only after all chunks are processed is the result written.

**Worked example.** Multiplying $A\in\mathbb R^{256\times512}$ by $B\in\mathbb R^{512\times128}$ with output tiles $64\times64$ requires eight output programs. If $B_K=32$, each program executes sixteen reduction steps. Every program must accumulate all sixteen contributions before applying a final nonlinearity such as ReLU. Applying ReLU to each partial product and summing would compute a different function.

Boundary masks apply independently to rows, columns, and the reduction dimension. Matrix layout determines pointer strides. Accumulation precision affects numerical error. These are correctness details even when the main motivation is speed.

## 7. Researched extension — compilers and handwritten kernels meet at the same constraints

[Triton's official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) expose block-level programming directly. [OpenXLA](https://openxla.org/xla) describes a compiler route from higher-level array programs to optimized accelerator execution. Both ultimately confront layout, fusion, memory capacity, scheduling, and hardware instruction constraints.

The useful decision rule is empirical. Begin with a clear high-level implementation, profile it, enable an appropriate compiler path, and inspect the remaining bottleneck. Write a custom kernel when a specific repeated computation is poorly served and its benefit justifies maintenance. A custom kernel also creates obligations: gradients, multiple shapes, numerical tolerances, hardware coverage, and future compatibility.

An instructive solo experiment is a fused `matmul + bias + activation` benchmark with both convenient and awkward dimensions. Predict traffic and tile counts first, then compare predictions with measurements. Disagreement is valuable: it reveals an omitted cache effect, launch cost, library optimization, or occupancy constraint.

## 8. Exercises and worked solutions

**Exercise 1.** For $N=1025$ and block width 256, how many instances run and how many valid elements belong to the last?

<details><summary>Solution</summary>

Five instances run. The last begins at 1024 and has one valid element. This is a useful boundary test because almost the entire last block is masked.

</details>

**Exercise 2.** Why is $\operatorname{ReLU}(A_1B_1+A_2B_2)$ generally different from $\operatorname{ReLU}(A_1B_1)+\operatorname{ReLU}(A_2B_2)$?

<details><summary>Solution</summary>

ReLU is nonlinear. For scalar partial products 2 and -3, the first expression is zero and the second is two. Apply the nonlinearity after the reduction when that is the reference operation.

</details>

**Exercise 3.** A fused kernel is faster on a million-element tensor but slower on 64 elements. Give a plausible explanation and a measurement that distinguishes it.

<details><summary>Solution guidance</summary>

At small size, launch and dispatch overhead can dominate; a built-in path may have lower overhead or better specialization. At large size, saved memory traffic matters more. Measure a shape sweep after warmup, inspect launch counts, and separate compilation. Do not infer a universal speedup from one favorable size.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=xnDHaNUvHBg) and [official executable lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_06.py).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html): vector operations, fused softmax, and tiled matrix multiplication.
- [OpenXLA overview](https://openxla.org/xla): compiler architecture; an extension beyond the recording's main demonstrations.
