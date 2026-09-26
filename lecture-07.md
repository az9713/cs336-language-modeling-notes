# Lecture 7 — Collective communication and parallel training

The recording builds distributed training from small communication operations. This is the right abstraction level for understanding correctness: before naming a parallelism framework, determine what each device owns and which values must be combined. The lecture proceeds through collectives, interconnects, distributed programming, and simple data, tensor, and pipeline parallel examples.

## 1. Replicate what you can afford; shard what you cannot [00:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=0s)

Multiple accelerators provide more arithmetic throughput and more aggregate memory. These benefits are distinct. Replicating a model across eight GPUs increases processing capacity but does not make an oversized model fit on one GPU. Sharding the model can increase feasible size, but then devices must exchange intermediate information.

A **rank** identifies a process in a communication group. The **world size** is the number of ranks in that group. One process per GPU is a common configuration, including the lecture examples, but rank and physical GPU are not universal synonyms.

For each tensor, describe its ownership: fully replicated, partitioned by batch, partitioned by feature, or partitioned by another axis. Also state whether a local value is a complete result or a partial contribution needing reduction. Many distributed bugs are ownership misunderstandings disguised as shape-correct code.

## 2. Collectives specify data transformations across ranks [06:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=360s)

A **broadcast** copies one rank's value to the group. A **scatter** distributes pieces from one rank. An **all-gather** collects each rank's piece so every rank receives the concatenated result. A **reduce-scatter** first combines corresponding contributions, typically by sum, and distributes shards of the reduced result. An **all-reduce** combines corresponding contributions and returns the result to every rank. An **all-to-all** sends different pieces from every rank to different destinations.

**Worked example.** Two ranks hold vectors $a=(1,2)$ and $b=(3,4)$. A sum all-reduce gives $(4,6)$ on both ranks. A sum reduce-scatter gives scalar 4 on rank zero and scalar 6 on rank one. An all-gather of those scalars reconstructs $(4,6)$ everywhere. This illustrates the logical decomposition of all-reduce into reduce-scatter followed by all-gather.

Collectives are coordinated operations. Ranks must call compatible operations in a compatible order, with matching groups and tensor specifications. A mismatch can hang rather than produce an informative numerical error. A barrier synchronizes arrival but does not repair an inconsistent communication schedule.

All-to-all is particularly natural for MoE dispatch: tokens begin distributed by input batch but must travel to the devices owning their selected experts. A second exchange returns expert outputs to their original token positions. Routing metadata is part of correctness, not mere bookkeeping.

## 3. The interconnect is another memory hierarchy [22:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1320s)

Communication within a device, within a server, and across servers can have very different bandwidth and latency. Fast local links do not imply equal speed across the entire cluster. Remote direct memory access can reduce CPU involvement in transfers, but it does not eliminate network contention or synchronization costs.

A simple communication model is

$$t_{\rm comm}\approx n_{\rm messages}\alpha+\frac{Q}{\beta},$$

where $\alpha$ is per-message latency, $Q$ transferred bytes, and $\beta$ effective bandwidth. This model omits topology and overlap but explains why many tiny transfers can be expensive despite modest total bytes.

**Worked example.** With $\alpha=10$ microseconds and $\beta=50$ GB/s, one 1 MB message takes roughly 30 microseconds under the simplified model. One thousand separate 1 KB messages carry similar total data but incur roughly 10 milliseconds of latency. Bucketing gradients can therefore matter even before changing the total communication volume.

[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) implements collectives using the detected hardware and topology. The [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html) defines the higher-level process-group interfaces. A collective name specifies a result, not one fixed physical algorithm.

## 4. Ring accounting makes bandwidth measurements interpretable [46:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2760s)

For a ring all-reduce over $P$ ranks and tensor size $M$ bytes, a standard decomposition sends about $2(P-1)M/P$ bytes per rank: one reduce-scatter phase and one all-gather phase. Ignoring latency, an approximate time is

$$t\approx\frac{2(P-1)M}{P\beta}.$$

The algorithmic bandwidth $M/t$ differs from an estimate corrected for collective traffic. These quantities should not be compared blindly with a link's peak specification. Duplex conventions and aggregation across links can introduce further factors.

**Worked example.** For $P=8$, $M=1$ GB, and effective per-rank transfer rate 100 GB/s, the bandwidth-only estimate is 17.5 ms. A reported $M/t\approx57.1$ GB/s is compatible with that network rate because the collective moves more than one tensor's worth per rank.

Benchmark all participating ranks with appropriate synchronization and state which time summary is reported. For end-to-end progress, the slowest rank often controls the step, so an average can conceal a straggler.

## 5. Data parallelism reconstructs the global gradient [55:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3300s)

Suppose rank $r$ holds $n_r$ examples and computes a local mean gradient $g_r$. The global mean gradient is

$$g=\frac{\sum_{r=1}^P n_rg_r}{\sum_{r=1}^P n_r}.$$

For equal local sizes, this becomes $P^{-1}\sum_rg_r$. After an all-reduce and the correct normalization, every rank can make the same optimizer update. This requires initially equal parameters, identical optimizer state, and consistent updates.

**Worked example.** Two equal-sized shards produce scalar gradients 2 and 6. Their mean is 4. If each replica updates independently, they move by different amounts and cease to represent one model. If gradients are summed but not divided while the intended objective is a mean, the update becomes twice as large.

For language modeling, valid predicted-token count may be the relevant weight instead of example count. Padding, variable lengths, and masked prompt tokens make equal numbers of sequences an insufficient guarantee of equal denominators.

Gradient communication can overlap with backward computation as parameter buckets become ready. This helps only when dependencies and available bandwidth permit it. The final communication tail can remain exposed even when much of the transfer is hidden.

## 6. Tensor parallelism changes where matrix products finish [62:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3720s)

For $Y=XW$, split $W$ by output columns: $W=[W_1\;W_2]$. Each rank computes $Y_i=XW_i$, and concatenation reconstructs $Y$. By contrast, split the input feature axis as $X=[X_1\;X_2]$ and split $W$ into matching row blocks. Then

$$Y=X_1W_1+X_2W_2.$$

Each local output is a partial sum requiring reduction. The two sharding patterns have different communication requirements even if their local matrices have similar sizes.

For a two-layer MLP, an output-column split of the first matrix can pair with an input-row split of the second. A pointwise activation is computed locally between them, and only the final partial results need summation. This avoids gathering the full intermediate activation unnecessarily.

Backward communication must match the forward operation's derivative. For example, a forward sum has gradients that propagate to each contributing operand; a forward gather requires routing and combining the appropriate gradient pieces. Raw communication calls do not automatically provide every desired autograd behavior.

## 7. Pipeline parallelism trades layer ownership for scheduling [69:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4140s)

Pipeline parallelism assigns successive layers to different devices. Activations move forward between stages and gradients move backward. One batch moving serially through all stages leaves most devices idle. Splitting it into microbatches allows stages to work on different microbatches concurrently.

If there are $S$ equally timed stages and $m$ microbatches, a forward-only ideal pipeline takes $m+S-1$ stage-time units. Its utilization is approximately $m/(m+S-1)$. Training schedules add backward work, activation storage, and weight-version constraints, so this is an introductory scheduling model rather than a complete training estimate.

**Worked example.** Four stages and four microbatches give forward utilization $4/7\approx57\%$. With sixteen microbatches it becomes $16/19\approx84\%$. Smaller microbatches reduce bubbles but can make each matrix multiplication less efficient. The tradeoff links scheduling back to single-device kernels.

## 8. Researched extension — write a distributed invariant before scaling

The [Megatron-LM paper](https://arxiv.org/abs/1909.08053) develops intra-layer model parallelism. Its broader engineering lesson is to exploit mathematical structure so that communication happens at a few well-defined boundaries.

For a small prototype, compare one complete single-device update with a distributed update using the same data and initial state. Check loss, gradients, and new parameters within justified tolerances. This tests an invariant more informative than “the distributed loss decreases.” A distributed system can decrease loss while optimizing the wrong normalization or using duplicated data.

## 9. Exercises and worked solutions

**Exercise 1.** Three ranks contain local mean gradients 1, 2, and 5 with valid-token counts 100, 100, and 200. Find the global mean.

<details><summary>Solution</summary>

The weighted sum is $100+200+1000=1300$ over 400 tokens, giving 3.25. The unweighted rank mean $8/3$ is incorrect for the intended token objective.

</details>

**Exercise 2.** Which collective naturally combines partial outputs from a row-sharded linear layer when every rank needs the full answer?

<details><summary>Solution</summary>

A sum all-reduce. If only a shard of the final answer is needed on each rank, reduce-scatter may avoid reconstructing the full replicated output.

</details>

**Exercise 3.** Why can adding GPUs slow a fixed small batch?

<details><summary>Solution</summary>

Local compute shrinks while synchronization, collective latency, and some communication remain. Small local matrices may also use the accelerator less efficiently. Parallelism is beneficial only while saved computation exceeds the new overhead and lost utilization.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=SzpOcwdIL0Y) and [official code](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_07.py).
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) and [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html).
- [Megatron-LM](https://arxiv.org/abs/1909.08053): tensor-parallel transformer training.
