# Lecture 8 — Sharding, scheduling, and multidimensional parallelism

The second parallelism lecture asks how the primitives from Lecture 7 are combined in real training systems. Its sequence runs from data parallelism and ZeRO through pipeline, tensor, sequence, and expert parallelism to hybrid configurations and activation recomputation. The central question is where to spend communication to save memory and expose useful concurrent work.

## 1. Data parallelism duplicates more than weights [12:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=720s)

In replicated data parallelism, every rank holds parameters, gradients, and optimizer state. Only the input batch is partitioned. Let $N$ be parameter count and $b_w,b_g,b_o$ the bytes per parameter for weights, gradients, and total optimizer state. Model-state storage per rank is

$$M_{\rm replicated}=N(b_w+b_g+b_o).$$

Activations and workspaces are additional. Increasing the number of replicas does not reduce this quantity. If optimizer state dominates, it is wasteful to keep identical copies on every rank when each update coordinate can be computed by an assigned owner.

**Worked example.** Assume two-byte weights, two-byte gradients, and eight bytes of Adam moments per parameter. A billion parameters require 12 GB of replicated model state. A separate four-byte master copy would raise it to 16 GB. This chapter's examples use the 12-byte convention unless stated otherwise.

## 2. ZeRO progressively removes redundant state [15:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=900s)

For $P$ ranks, an idealized memory model for ZeRO stages is

$$M_1=N(b_w+b_g+b_o/P),$$
$$M_2=N(b_w+(b_g+b_o)/P),$$
$$M_3=N(b_w+b_g+b_o)/P.$$

Stage 1 partitions optimizer state; Stage 2 also partitions gradients; Stage 3 also partitions parameters. These are persistent-state estimates, not peak memory. Stage 3 gathers parameter groups when needed and can temporarily hold both local shards and a materialized group.

**Worked example.** With one billion parameters, eight ranks, and the 12-byte convention, the three estimates are 5 GB, 3.25 GB, and 1.5 GB per rank, compared with 12 GB for replication. The Stage 3 number alone is insufficient to size a device because gathered layers, activations, and workspaces may dominate the peak.

An optimizer owner receives the reduced gradient for its shard, updates its parameters and moments, and distributes updated parameters when replicas need them. The logical decomposition of all-reduce into reduce-scatter plus all-gather makes some of these transformations possible without a dramatic increase in communication volume. Fully sharded approaches add parameter materialization and require more careful overlap.

The [ZeRO paper](https://arxiv.org/abs/1910.02054) is the primary source. “FSDP” names a family of fully sharded implementations; exact prefetching, resharding, and memory behavior depend on configuration.

## 3. Overlap is a dependency question [21:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1260s)

Suppose a layer takes compute time $c$ and gathering the next layer's weights takes time $g$. Perfect overlap can reduce the combined interval toward $\max(c,g)$ instead of $c+g$. But the next weights must arrive before use, buffers must fit, and the communication must not contend destructively with current work.

**Worked example.** If current compute takes 8 ms and the next gather takes 5 ms, a correctly prefetched schedule can hide the gather. If it takes 12 ms, at least 4 ms remain exposed. Prefetching two layers ahead may help but holds more materialized weights, increasing peak memory. Memory saving and overlap are therefore not independent knobs.

Strong scaling fixes total work and adds devices. Eventually each rank's compute becomes too small to hide communication. Weak scaling grows work with device count, but in training that often increases global batch size and changes optimization. A systems scaling curve must state which experiment it measures.

## 4. Pipeline schedules manage bubbles and saved activations [30:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1800s)

Pipeline parallelism partitions layers across stages. Microbatches allow different stages to execute concurrently. A fill-and-drain schedule first runs forwards and later backwards; a one-forward-one-backward schedule interleaves them after startup, often reducing the number of live activations.

For an ideal balanced pipeline with $S$ stages and $m$ microbatches, a common introductory utilization expression is $m/(m+S-1)$. Actual schedules include separate forward, input-gradient, and weight-gradient work. The lecture discusses exploiting those distinctions to fill otherwise idle intervals. A “zero-bubble” result is conditional on schedule, workload balance, and memory assumptions, not a guarantee that any pipeline has no idle time.

**Worked example.** Eight stages and eight microbatches give the simple utilization estimate $8/15\approx53\%$. Increasing to 64 gives $64/71\approx90\%$. But a fixed global batch divided into many microbatches can leave each GEMM too small. The best schedule balances pipeline utilization against per-microbatch kernel efficiency.

Stage imbalance also matters. If one stage takes twice as long as the others, the steady-state throughput is constrained by that stage. Equal layer counts need not imply equal stage times, especially with embeddings, output heads, and heterogeneous expert blocks.

## 5. Tensor and sequence parallelism divide different objects [40:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2400s)

Tensor parallelism partitions large matrix operations. A column-sharded up-projection and row-sharded down-projection can keep the MLP intermediate local, as derived in Lecture 7. Attention heads can also be distributed when architecture and head counts permit it.

Pointwise operations and normalization may otherwise leave some activations replicated. **Sequence parallelism**, in the Megatron-style usage discussed here, partitions sequence positions for selected operations to reduce this replication, switching layouts around tensor-parallel regions with collectives. **Context parallelism** can partition a long context while arranging attention communication across its pieces. Terminology varies across systems; specify what tensors are actually sharded.

An activation tensor with batch $B$, sequence $T$, width $d$, and element size $b$ consumes $BTdb$ bytes. If a particular activation is cleanly partitioned across $P$ ranks, its local share is $BTdb/P$. Other saved tensors and temporary gathered representations do not necessarily shrink by the same factor.

**Worked example.** A $B=4$, $T=16{,}384$, $d=4096$ bfloat16 activation is 512 MiB. Sharding that tensor eight ways reduces its persistent local portion to 64 MiB. This is a statement about one tensor, not the complete model's activation memory.

## 6. Expert parallelism turns routing into network traffic [53:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3180s)

Expert parallelism assigns different experts to different devices. Tokens move to the ranks that own their selected experts, and outputs move back. The traffic depends on token count, hidden width, routing multiplicity, and expert placement. Uneven routing creates both compute and communication imbalance.

For $T_b$ tokens in a local dispatch batch, hidden width $d$, top-$k$ routing, and $b$ bytes per activation element, a simple upper accounting term for sent activations is about $T_bkdb$ before considering local experts, metadata, compression, and return traffic. Returning expert outputs adds a comparable term.

**Worked example.** Dispatching 8192 tokens to two experts each with width 4096 and two-byte activations represents 128 MiB of activation payload before the return path. A router with low arithmetic cost can therefore create a substantial network demand.

Tensor and expert parallelism can conflict: aggressively splitting already small expert matrices lowers matrix efficiency. Attention and MLP regions may prefer different device groupings. More parallel dimensions increase the design space; they do not automatically improve utilization.

## 7. Choose a hybrid according to the physical hierarchy [62:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3720s)

A hybrid configuration assigns degrees of data, tensor, pipeline, context, and expert parallelism. Some dimensions multiply into a total device count; others overlap or use different process groups, depending on the implementation. It is unsafe to multiply every named degree without understanding the mapping.

Frequent latency-sensitive collectives usually benefit from fast local interconnects. Coarser communication can span slower links when enough computation occurs between transfers. The intended global batch constrains data parallelism. Memory limits constrain parameter and context partitioning. Kernel shapes constrain how finely tensors and experts can be split.

The lecture's real-model examples illustrate these interacting constraints at the time of the course. They should be treated as case studies, not fixed recipes to copy onto a different cluster. The [Megatron-LM work](https://arxiv.org/abs/1909.08053) and [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) supply primary technical treatments of several mechanisms.

## 8. Researched extension — useful throughput can rise when FLOPs increase

Suppose a memory-constrained run can use batch 1 at 20% of peak useful arithmetic throughput. Checkpointing adds 25% recomputation but permits a larger batch that reaches 40% hardware throughput. In a simplified model, useful throughput becomes $0.40/1.25=0.32$ of peak, exceeding the original 0.20.

This independently constructed example explains why minimizing arithmetic is not the same as minimizing time to a target loss. It also explains why reporting only hardware FLOPs can mislead: recomputation raises executed operations without increasing training-token progress. Measure tokens per second and time to the same quality target, with the altered batch and optimization recipe accounted for.

## 9. Exercises and worked solutions

**Exercise 1.** Under two-byte parameters, two-byte gradients, and eight-byte moments, compute Stage 2 state memory for two billion parameters on four ranks.

<details><summary>Solution</summary>

$2\times10^9[2+(2+8)/4]=9\times10^9$ bytes, or 9 GB in decimal units. This excludes activations, master weights, and temporary buffers.

</details>

**Exercise 2.** Why might fully sharding a tiny model be slower than ordinary data parallelism?

<details><summary>Solution</summary>

The model already fits, so additional parameter gathers provide little practical benefit. Their latency and bookkeeping can dominate short compute intervals. Sharding solves a resource problem; it has an overhead when that problem is absent.

</details>

**Exercise 3.** Which measurements would you collect before increasing expert parallel degree?

<details><summary>Solution guidance</summary>

Measure per-expert token counts, expert GEMM sizes and times, dispatch/return time, network utilization, and peak memory. Increasing the degree can save memory while shrinking GEMMs or worsening communication. The critical path identifies which effect matters.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=6-cXp-aOmdg) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_08.pdf).
- [ZeRO](https://arxiv.org/abs/1910.02054): optimizer, gradient, and parameter sharding.
- [Megatron-LM](https://arxiv.org/abs/1909.08053): tensor parallelism.
- [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198): activation memory and sequence-parallel techniques.
