# Lecture 4 — Compress history, select computation, and account for what is lost

Atlas must answer questions about a long engineering notebook. Early pages assign a tolerance of 3 units to component A and 5 units to component B. A later revision changes A to 7. The model must retrieve the latest value for A without corrupting B, even after thousands of unrelated tokens. Keeping every past key and value is expensive. Compressing the notebook into a fixed-size state is cheaper, but an additive memory might report 10 for A by summing its old and new values.

A second constraint is independent of history: Atlas needs more learned capacity for code and multilingual documentation, but its compute budget cannot grow in proportion to all stored parameters. A mixture of experts can select a few feed-forward networks for each token. That saves arithmetic only if routing, communication, and imbalance do not consume the gain.

This chapter develops both mechanisms around that notebook. The first half changes how information from past positions is represented. The second changes which learned parameters process a position. They can be combined, but their storage and failure modes differ. Recording-time model comparisons motivate the design space; the derivations and CPU experiments here establish narrower, reproducible properties rather than current leaderboard rankings.

## 1. Reassociate a computation only after identifying its function [05:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=350s)

### A nonlinear operation cannot be moved like a parenthesis

Let $T$ be sequence length, $d_k$ query/key width, and $d_v$ value width. With one sequence and one head, row-stacked matrices $Q,K\in\mathbb R^{T\times d_k}$ and $V\in\mathbb R^{T\times d_v}$ contain projected token representations. Full attention produces $Y=\operatorname{softmax}(QK^\top/\sqrt{d_k})V$, with softmax applied separately to each row. Causal attention also excludes future columns from each row's normalization.

The score matrix has $T^2$ entries before causal exclusions. Its multiplication and the subsequent value aggregation require work proportional to $T^2(d_k+d_v)$. Feed-forward transformations instead have work linear in $T$ at fixed widths. Thus growing the notebook eventually changes which part of Atlas dominates computation.

If we remove softmax and absorb any fixed scale into the projections, the resulting expression admits

$$Y=(QK^\top)V=Q(K^\top V).$$

The first evaluation forms a $T\times T$ matrix. The second forms a $d_k\times d_v$ matrix, with leading work proportional to $Td_kd_v$. It is linear in sequence length when the feature widths are fixed. This is an exact associativity identity for the new linear expression, not an identity for the original softmax attention.

With scalar query 1, keys 1 and 2, and values 1 and 0, the linear output is $1\cdot1+2\cdot0=1$. The softmax output is $e/(e+e^2)\approx0.269$. Reparenthesizing cannot erase this difference. Linear weights can also be negative and need not sum to one; interpreting the output as a convex average would import a property that was removed.

### Causality requires a prefix state

The unmasked product $Q(K^\top V)$ lets every query use every key, including future keys. For a causal model, switch to column vectors $q_t,k_t\in\mathbb R^{d_k}$ and $v_t\in\mathbb R^{d_v}$ at position $t=1,\ldots,T$. Define state $S_t\in\mathbb R^{d_k\times d_v}$ by

$$S_0=0,\qquad S_t=S_{t-1}+k_tv_t^\top,\qquad y_t=S_t^\top q_t.$$

Expanding the recurrence gives $S_t=\sum_{i\le t}k_iv_i^\top$, hence $y_t=\sum_{i\le t}v_i(k_i^\top q_t)$. Every output uses exactly the permitted prefix. The state has $d_kd_v$ entries regardless of $T$. By contrast, explicit keys and values contain $T(d_k+d_v)$ entries. With both widths 64 and context 8,192, these counts are 4,096 versus 1,048,576 per head, excluding other state.

This reduction is purchased by aggregation. Once two histories produce the same $S_t$, this head cannot distinguish them through subsequent reads of that state alone. For Atlas, repeatedly adding the key for A writes 3 and then 7 into the same direction, yielding 10. The representation has no built-in notion of “latest revision.”

The CPU experiment below compares a masked dense expression with its prefix recurrence. Inputs are finite floating-point CPU matrices with matched sequence and feature dimensions. The example creates its own tensors, performs no file operations, and uses a tiny quadratic reference only as an independent check. The masked form and recurrence agree to numerical tolerance; agreement does not establish equivalence with softmax attention.

```python
import torch

torch.manual_seed(4)
q = torch.randn(5, 3, dtype=torch.float64)
k = torch.randn(5, 3, dtype=torch.float64)
v = torch.randn(5, 2, dtype=torch.float64)
mask = torch.ones(5, 5, dtype=torch.float64).tril()
reference = ((q @ k.T) * mask) @ v
state = torch.zeros(3, 2, dtype=torch.float64)
outputs = []
for qt, kt, vt in zip(q, k, v):
    state = state + torch.outer(kt, vt)
    outputs.append(state.T @ qt)
assert torch.allclose(torch.stack(outputs), reference)
assert not torch.allclose(q @ (k.T @ v), reference)
```

Normalized kernel attention is another construction: for a positive feature map $\phi$, one may retain $S_t=\sum_{i\le t}\phi(k_i)v_i^\top$ and $z_t=\sum_{i\le t}\phi(k_i)$, then divide $S_t^\top\phi(q_t)$ by $z_t^\top\phi(q_t)$ when the denominator is positive. That restores a normalization for the chosen kernel. It does not generally reproduce the exponential dot-product kernel with a small finite feature map. Naming the actual function is essential before claiming an optimization preserves it.

## 2. Learn when to forget and what to overwrite [11:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=660s)

### Global forgetting is different from targeted correction

Introduce an input-dependent scalar $\gamma_t\in[0,1]$ and replace the additive update by $S_t=\gamma_tS_{t-1}+k_tv_t^\top$. The old state is retained when $\gamma_t=1$ and erased when $\gamma_t=0$. With constant $0<\gamma<1$, an association written $r$ steps earlier is weighted by $\gamma^r$. Its half-life in token steps is $\log(1/2)/\log\gamma$: at $\gamma=0.99$, approximately 69 steps. Long retention and rapid reset pull this simple mechanism in different directions.

The [Mamba-2 state-space duality paper](https://arxiv.org/abs/2405.21060) connects structured recurrent computations with attention-like forms and efficient evaluation. A scalar-gated matrix memory captures an important mechanism, not the complete architecture with all projections, convolutions, and output terms. A direct value-dependent skip can carry current-token information to the output without changing how the historical state is updated.

For Atlas, a global reset may be useful at the boundary between unrelated documents. It is a poor way to revise only A: multiplying all of $S$ also weakens B. A delta update instead corrects the prediction for the incoming key. Let $\widetilde S_t=\gamma_tS_{t-1}$ be the decayed state and $\beta_t\in[0,1]$ the write strength. Define

$$e_t=v_t-\widetilde S_t^\top k_t,\qquad
S_t=\widetilde S_t+\beta_t k_te_t^\top.$$

The vector $e_t\in\mathbb R^{d_v}$ is the discrepancy between the desired value and the memory's current read for that key. For a unit-length key, reading immediately after the update gives $(1-\beta_t)\widetilde S_t^\top k_t+\beta_tv_t$. At $\beta_t=1$, it writes the new value exactly along that direction.

Expanding the update yields $S_t=(I-\beta_tk_tk_t^\top)\widetilde S_t+\beta_tk_tv_t^\top$, where $I$ is the $d_k$-dimensional identity matrix. The first factor is an orthogonal projector only when the key has unit length and $\beta_t=1$. Otherwise it is a rank-one modification. Its eigenvalue in the key direction is $1-\beta_t\|k_t\|_2^2$; perpendicular directions have eigenvalue one. Large unnormalized keys can therefore produce overshooting rather than gentle erasure.

### Memory update as a small learning problem

Consider the squared prediction error $\ell_t(S)=\tfrac12\|S^\top k_t-v_t\|_2^2$. Differentiating each matrix entry gives $\nabla_S\ell_t=k_t(S^\top k_t-v_t)^\top$. A gradient step of size $\beta_t$, starting at $\widetilde S_t$, is exactly the delta update above. This is a learning computation inside a forward pass: the state acts like the weight matrix of a temporary linear predictor. It does not mean that all pretrained model weights are being optimized during inference.

[Gated Delta Networks](https://arxiv.org/abs/2412.06464) develops the combination of forgetting and delta updates, including hardware-efficient training algorithms. Our orientation stores $S$ as key-width by value-width; transposing it produces the equivalent value-width by key-width convention used in that paper. Shape conventions matter more than matching the visual order of symbols.

The cumulative `delta_step` helper accepts floating-point CPU tensors: a state matrix, key vector, and value vector with matching shapes and dtype, plus finite scalar gates in $[0,1]$. It rejects invalid shapes and nonfinite inputs with `ValueError`, does not mutate inputs, and preserves PyTorch's differentiation graph. It deliberately does not normalize keys; the caller must enforce any unit-key assumption.

```python
import torch
from cs336_lab.memory import delta_step

state = torch.tensor([[3.0], [5.0]], dtype=torch.float64)
key_a = torch.tensor([1.0, 0.0], dtype=torch.float64)
new_value = torch.tensor([7.0], dtype=torch.float64)
updated = delta_step(state, key_a, new_value)
assert torch.equal(updated, torch.tensor([[7.0], [5.0]], dtype=state.dtype))
assert torch.equal(state, torch.tensor([[3.0], [5.0]], dtype=state.dtype))
learnable = state.clone().requires_grad_()
loss = 0.5 * ((learnable.T @ key_a - new_value) ** 2).sum()
gradient, = torch.autograd.grad(loss, learnable)
assert torch.equal(updated, learnable - gradient)
```

B survives because its key is orthogonal to A's. For another key $h$, the change in its read is $\beta_t e_t(k_t^\top h)$. Similar keys interfere in direct proportion to their inner product. Exact overwriting on the selected key does not imply a perfect dictionary over arbitrarily many keys.

### Why the recurrence can still be trained in parallel

For gates and keys computed from the current layer's input, the state update is affine: $S_t=A_tS_{t-1}+B_t$, with $A_t=\gamma_t(I-\beta_tk_tk_t^\top)$ and $B_t=\beta_tk_tv_t^\top$. Two successive maps compose as $(A_2,B_2)\circ(A_1,B_1)=(A_2A_1,A_2B_1+B_2)$. Function composition is associative, permitting prefix-scan and chunked evaluation.

Associativity alone does not make a generic dense-matrix scan cheap. Useful implementations exploit the structure of $A_t$, divide work into chunks, and map substantial work to matrix multiplication. If the coefficients instead depend arbitrarily on the evolving state through nonlinear functions, this particular affine composition no longer applies. The benefit rests on a restricted recurrence, not merely on calling a model an RNN.

## 3. Preserve selected access paths without hiding their cost [19:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1140s)

### Hybrids reduce a coefficient before they change an order

Suppose Atlas replaces three of every four attention layers with recurrent memory but keeps the fourth global. Recurrent layers can cheaply mix nearby or compressible information, while a global layer can still revisit individual past entries. This is a different compromise from requiring every layer to compress everything into a fixed-size state.

For $L$ layers and a fixed global fraction $f>0$, an illustrative prefill work model is $C(T)=L[f aT^2+(1-f)bT]$, where $a$ and $b$ summarize feature dimensions and arithmetic constants. The quadratic component remains for arbitrarily large $T$. Reducing $f$ from one to one quarter can still be valuable: asymptotic notation does not erase a factor of four or a memory-allocation constraint.

Global layers also retain their growing KV caches. A mixed memory budget sums their per-token cache costs and the fixed recurrent states. Calling the entire hybrid a constant-memory model because most layers are recurrent would omit the surviving cache. For Atlas's revised notebook, evaluate both long-range fact access and throughput; a hybrid ratio is an experimental variable with a representational cost.

The [FlashAttention paper](https://arxiv.org/abs/2205.14135) supplies a different kind of improvement: an exact attention algorithm organized to reduce transfers between memory levels. Avoiding a stored quadratic score matrix does not remove quadratic attention arithmetic. Its role is to make a given mathematical operation execute more efficiently, whereas replacing it with recurrent memory changes the operation itself.

### Sparse attention pays to discover what it will read

A sparse attention layer retains individually addressable history but selects a set $\mathcal I_t$ of at most $K_s$ earlier positions for query $t$. It then runs attention over that set. The subscript distinguishes selection count $K_s$ from the key matrix $K$. A learned indexer assigns scores and takes the highest-scoring allowed positions. Selection must respect causality just as dense attention does.

The [DeepSeek-V3.2-Exp primary repository and report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) describes a lightweight indexer followed by sparse attention, introduced through continued training. This supports the architecture's mechanism; the notebook example here is not a reproduction of its measured quality.

If the indexer compares every pair in width $d_i$, its score work scales as $T^2d_i$. Selected query/key comparisons and value aggregation scale as $TK_s(d_k+d_v)$. Selection, projection, gathering, and communication add further costs. Reducing the indexer's width or precision can greatly lower its constant while leaving the all-pairs component quadratic.

Take $T=32768$, $d_k=d_v=128$, $d_i=16$, and $K_s=512$. Ignoring a common multiply-add factor and causal triangular constants, dense attention has a work proxy $T^2(128+128)=274{,}877{,}906{,}944$. Indexing plus selected attention has proxy $T^2\cdot16+T\cdot512\cdot256=21{,}474{,}836{,}480$, a factor of 12.8 smaller. This is an accounting comparison, not a 12.8-fold runtime prediction. It also ignores the possibility that the selector misses a crucial earlier revision.

If the selector omits the only relevant token, exact attention on the selected set cannot recover it from that set. Atlas should therefore measure selection recall for known evidence positions, followed by answer accuracy conditional on selection. This separates failures of finding evidence from failures of using it. The distinction mirrors retrieval systems outside the neural layer and suggests a practical diagnostic even when average language-model loss looks healthy.

## 4. Store more parameters than each token executes [34:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2040s)

### A sparse mixture changes the feed-forward transformation

Return to a token representation $x\in\mathbb R^d$ in Atlas's residual stream. A mixture-of-experts layer has $E$ expert functions $F_i:\mathbb R^d\to\mathbb R^d$ and a router that chooses a subset $\mathcal T(x)$ of $k$ experts. With nonnegative gate weights $g_i(x)$, its output is

$$F_{\rm MoE}(x)=\sum_{i\in\mathcal T(x)}g_i(x)F_i(x).$$

The surrounding residual addition and attention layer can remain as in Chapter 3. In the common design considered here, routing occurs per token and per MoE layer. The same document can therefore send different tokens to different experts; it is not assigned once to a single specialist model.

Let each expert contain $P_e$ parameters, and let $P_s$ count all other parameters, including any always-active shared components and router. Total storage is governed by $P_s+EP_e$, while a token activates roughly $P_s+kP_e$. For eight experts of 100 million parameters and 200 million shared parameters, the totals are one billion stored and 400 million active under top-2 routing.

Inactive experts still require parameter storage and, during ordinary training, optimizer state. Their gradients may be absent for one token, but they are not absent from the model. Dense matrix arithmetic in selected experts can be much lower than activating all eight, yet routing and moving activations introduce work not summarized by active parameter count.

[Switch Transformers](https://arxiv.org/abs/2101.03961) provides a foundational sparse-routing design and evidence for scaling expert capacity. Its experiments motivate the distinction between stored and executed parameters. They do not imply a free reduction in all resources: more experts can increase memory and communications even when selected-expert FLOPs are held fixed.

### Granularity and shared processing change the allocation

Fine-grained experts divide a larger feed-forward budget into more, smaller networks. If expert size halves while the number selected doubles, leading active expert parameters can remain unchanged. The number of possible selected combinations can increase, but that combinatorial count is not a proof of better representations. Smaller per-expert token batches may produce inefficient matrix multiplications and greater dispatch overhead.

An always-active shared expert supplies a common transformation in addition to routed experts. The intended benefit is that common processing need not be relearned independently everywhere. The cost is spending some active capacity on the same component for all tokens. To compare fairly, replace part of the routed budget with the shared component rather than simply adding it for free.

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) develops fine-grained segmentation and shared expert isolation. [OLMoE's controlled shared-expert comparison](https://arxiv.org/html/2409.02060v2#S4.SS1.SSS3) finds that enforced sharing did not improve its matched setting. These results need not be contradictory: useful parameter allocation depends on the rest of the model and training protocol. Atlas should treat “shared expert” as a hypothesis to test, not a required ingredient inferred from another system's popularity.

Nor does an expert's index certify a human-readable topic. Routing may correlate with token identity, language, punctuation, domain, or context. A simple linear router can still receive a contextual representation produced by many previous layers. Claims about semantic specialization require inspection and controlled interventions, rather than either assuming rich specialists or declaring that specialization is impossible.

## 5. Understand where gradients flow through a hard router [48:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2880s)

### Selection and weighting are distinct operations

A simple router computes $r_i=w_i^\top x$, where $w_i\in\mathbb R^d$ is a learned expert vector, and probabilities $p_i=e^{r_i}/\sum_j e^{r_j}$. Token-choice routing selects the $k$ largest scores for each token. Expert-choice routing instead gives each expert a quota of favored tokens; this balances expert counts directly but can assign a variable number of experts to each token. Global assignment can optimize a score under capacity constraints, at the cost of solving a coupled allocation problem.

Hash routing uses a fixed rule instead of learning all assignments. It is a useful baseline because it tests whether additional capacity helps without an elaborate learned selector. Reinforcement-learning formulations can treat selection as a discrete action, and noisy routing can perturb close scores to explore alternatives. These approaches introduce their own variance, overhead, and design choices. The recording emphasizes that practical systems often use simpler hard selection plus auxiliary mechanisms.

For a fixed selected set, two gate conventions are important. One uses the original full-softmax weights $g_i=p_i$ for selected experts. Another renormalizes them, $g_i=p_i/\sum_{j\in\mathcal T}p_j$. The latter is equivalent to a softmax restricted to the selected logits. The conventions have different derivatives and should never be silently interchanged.

Away from a tie boundary, selected indices are locally constant. Differentiation follows the selected computations and their gate weights; it does not evaluate how a small hypothetical move across the boundary would change the selected set. At a tie, the discrete selection can change abruptly. Calling top-$k$ differentiable without these qualifications would obscure the learning problem.

### A top-1 normalization can remove the task gradient

If exactly one expert is selected and its gate is renormalized to sum to one, the gate is always one. Holding the selected index fixed, the task output then has no dependence on the router logits through that gate. A router may still receive an auxiliary-loss gradient or use another estimator, but the ordinary task gradient through that normalized scalar is zero.

If instead the selected expert is multiplied by its original full-softmax probability, the gate varies with the logits. Even an unselected logit's router parameter can receive a gradient through the softmax denominator. This does not mean the unselected expert's network receives a task gradient: it was never executed. Separating router parameters from expert parameters prevents a common misconception.

The following CPU check uses distinct logits, so there is no tie ambiguity. Expert outputs are fixed scalar values solely to isolate gate differentiation. It compares the same top-1 choice under the two conventions, with no learning loop or external effects.

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
probabilities = torch.softmax(logits, dim=0)
chosen = probabilities.argmax()
expert_value = torch.tensor(3.0)
plain = probabilities[chosen] * expert_value
normalized = plain / probabilities[chosen]
g_plain, = torch.autograd.grad(plain, logits, retain_graph=True)
g_normalized, = torch.autograd.grad(normalized, logits)
assert g_plain[0] > 0 and g_plain[1] < 0
assert torch.allclose(g_normalized, torch.zeros_like(logits), atol=1e-6)
```

This demonstrates why a seemingly harmless normalization choice can alter training. It does not prove one convention is universally superior. Atlas's implementation must specify the gates, selection rule, tie handling, and auxiliary objectives together. Reproducing only “top-1 MoE” is insufficient to reproduce a training recipe.

## 6. Balance learning opportunities and device load [63:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3780s)

### Popular experts can reinforce their own popularity

An expert selected early receives more examples, can improve faster, and may become still more attractive. Other experts can remain poorly trained. This positive feedback wastes stored capacity and can overload a device even though total token assignments are unchanged. The problem is both statistical and operational.

For a top-1 batch of $N$ token representations, let $f_i$ be the fraction assigned to expert $i$ and $P_i=N^{-1}\sum_t p_{t,i}$ its average router probability. A common auxiliary objective is

$$\mathcal L_{\rm bal}=\alpha E\sum_{i=1}^E f_iP_i,$$

where $\alpha\ge0$ controls its strength. Treat $f_i$ as fixed when differentiating the current routing decision. Then $\partial\mathcal L_{\rm bal}/\partial P_i=\alpha E f_i$. The probability vector is constrained by softmax, however, so this intermediate derivative is not yet the derivative of a router logit.

Using $\partial p_{t,i}/\partial r_{t,j}=p_{t,i}(\delta_{ij}-p_{t,j})$, with $\delta_{ij}$ the equality indicator, gives

$$\frac{\partial\mathcal L_{\rm bal}}{\partial r_{t,j}}
=\frac{\alpha E}{N}p_{t,j}\left(f_j-\sum_i f_ip_{t,i}\right).$$

An expert more loaded than the probability-weighted average receives a positive logit gradient, so gradient descent tends to lower its logit. The sign interpretation is now precise. The auxiliary term does not compare all counterfactual expert outputs, and it is not a guarantee of a globally optimal assignment.

For four experts with both $f$ and $P$ concentrated on one expert, the objective divided by $\alpha$ is four. Under uniform $f=P$, it is one. This illustrates a useful comparison, not a proof that every feasible nonuniform configuration has a larger value. With uniform fixed $f$, the objective equals $\alpha$ for any probability vector and has zero logit gradient. Its behavior depends on the coupled evolution of assignments and probabilities.

Here is a numerical gradient check for the complete logit derivative. The hard assignment histogram is explicitly detached. All tensors are tiny CPU values, and the assertion compares automatic differentiation with the independently derived expression.

```python
import torch

logits = torch.tensor([[2.0, 0.0], [1.0, 0.0]],
                      dtype=torch.float64, requires_grad=True)
p = torch.softmax(logits, dim=-1)
f = torch.bincount(p.argmax(dim=-1), minlength=2).to(p.dtype) / 2
f = f.detach()
loss = 2 * (f * p.mean(dim=0)).sum()
gradient, = torch.autograd.grad(loss, logits)
expected = p * (f - (p * f).sum(dim=-1, keepdim=True))
assert torch.allclose(gradient, expected)
assert (gradient[:, 0] > 0).all()
```

### Expert balance does not exhaust the systems objective

If several experts share a device, its load is the sum of their assignments. Perfect per-expert uniformity with equal expert placement would imply device balance, but training usually encourages approximate balance rather than enforcing it exactly. A separate device-level objective can target the bottleneck more directly without forcing every expert to equal usage.

Another approach adjusts an expert's selection bias according to observed load. Lower an overloaded expert's selection score; raise an underused expert's score. This is a feedback controller over routing, distinct from differentiating an auxiliary loss. Gate values can still be computed from the unbiased affinity scores, separating allocation control from mixture weighting.

The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) uses this distinction and also retains a small sequence-level balancing objective. “Auxiliary-loss-free” in that design therefore does not mean that every auxiliary term has disappeared. Its affinity weights use sigmoid scores normalized across selected experts, not a second softmax applied to those sigmoid values. The exact formulation matters when reproducing gradients.

For Atlas, log both expert and device histograms, dropped assignments, and task loss. A lower imbalance score can conceal damaged language modeling if the balance pressure is excessive. The decision is to allocate enough learning and compute fairly while retaining useful specialization, not to optimize uniformity in isolation.

## 7. Account for communication, capacity, and adaptation [72:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4320s)

### Dispatch is part of the layer

Expert parallelism places different experts on different devices. Tokens must be grouped by destination, sent to those devices, processed, returned, and restored to their original order. For $N$ tokens, top-$k$ routing creates $Nk$ expert-token assignments. If every assignment travels remotely in residual width $d$ with $b$ bytes per component, dispatch and return carry a logical payload of approximately $2Nkd b$ bytes, excluding metadata and network overhead. Local assignments reduce remote traffic.

With $N=1024$, $k=2$, $d=768$, and $b=2$, that payload is 6 MiB. Across eight evenly loaded experts, the average is 256 assignments each. If one receives 1,024 while others share the remainder, its work can dominate the step despite unchanged total assignments. This is why active parameter count does not determine latency.

Grouping tokens into larger expert batches improves matrix multiplication reuse. [MegaBlocks](https://arxiv.org/abs/2211.15841) formulates MoE computation using block-sparse operations to handle variable expert loads without dropping tokens. This is an algorithm and kernel design, not an automatic consequence of owning a GPU with some unrelated sparsity feature. Different sparsity patterns require different implementations.

A lower-dimensional dispatch representation can reduce the payload. Projecting width $d$ to width $r<d$ before communication changes the proxy to $2Nkrb$. At $r=d/2$, payload halves, but projections add arithmetic and the bottleneck can discard useful information. An always-local shared branch may retain a wider path. This is an architectural tradeoff to evaluate, not a lossless compression guarantee.

### Capacity policies can change the model's output

A capacity factor $c$ often allocates approximately $\lceil cNk/E\rceil$ slots per expert. At $N=1024$, $k=2$, $E=8$, and $c=1.25$, this is 320 slots. The count provides headroom above the balanced average of 256, but does not prevent one expert from exceeding it. A policy must pad, drop, reroute, or dynamically accommodate the excess.

Dropping an assignment can return zero for that expert branch while preserving the residual path. The original token is not necessarily deleted from the sequence, but its transformation changes. If admission depends on other requests in the batch, the same request can receive a different output when its neighbors change. Calling that behavior sampling randomness would misidentify the cause.

Dropless execution removes this particular truncation mechanism. It does not prove bitwise reproducibility across batch sizes, devices, or parallel reduction orders. For Atlas, compare deterministic decoding of a fixed request alone and amid adversarially skewed routing loads, and inspect assignment counts when outputs differ.

### Stable routing and successful fine-tuning need separate evidence

Router softmax introduces the same logit-scale concerns studied in Chapter 3. Higher precision in the router and a penalty on its log partition can improve stability in particular designs. [ST-MoE](https://arxiv.org/abs/2202.08906) studies stability and transfer in sparse expert models. A finite router is still capable of collapsing assignments, so numerical stability does not replace load monitoring.

Fine-tuning can expose the large total capacity of an MoE to a small dataset. Sparse activation does not make overfitting impossible. Freezing some components or updating only attention reduces the set of fitted parameters; expanding task data changes the statistical regime. Compare training and held-out performance rather than assuming the pretraining recipe transfers unchanged.

Sparse upcycling initializes multiple experts from a trained dense feed-forward block. If all copied expert functions are initially identical and selected gates sum to one, their weighted output equals the original function. Subsequent routing and updates can break the symmetry. With unnormalized selected gates, exact preservation need not hold. [Sparse Upcycling](https://arxiv.org/abs/2212.05055) studies this reuse strategy; whether it beats continued dense training or training an MoE from scratch depends on the remaining budget and comparison protocol.

## 8. Combine mechanisms only after keeping their contracts separate [81:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4860s)

### A latent cache compresses each stored entry

The recording closes by following several generations of DeepSeek design. The broader lesson is that routing, storage, and training objectives can be co-designed. MoE selects feed-forward parameters. A latent attention cache instead compresses the per-token history retained by attention. They solve different resource problems even when used in one model.

For an illustrative column-vector model, let each token have latent vector $c_j\in\mathbb R^r$, key $k_j=U_Kc_j$, and value $v_j=U_Vc_j$, where $U_K\in\mathbb R^{d_k\times r}$ and $U_V\in\mathbb R^{d_v\times r}$. Then $q_t^\top k_j=(U_K^\top q_t)^\top c_j$. Also, for attention weights $a_{tj}$, $\sum_j a_{tj}v_j=U_V\sum_j a_{tj}c_j$. These identities move projections outside repeated per-history operations, permitting a latent representation to be retained rather than separately expanded keys and values.

The cache has $Tr$ components in this simplified model, compared with $T(d_k+d_v)$ for expanded vectors. It still grows with history length; it is not the fixed-size state of linear attention. Unlike an aggregate state, separate $c_j$ entries remain individually addressable. The compression limitation lies in their representational width and learned factorization.

Position complicates the projection identity. If a key is rotated by position-dependent matrix $R_j$, its score is $q_t^\top R_jU_Kc_j$. The transformed query $U_K^\top R_j^\top q_t$ now depends on which past position is being scored. One position-independent absorption no longer suffices. The [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) describes a separate positional component in its latent-attention construction. Our factorization explains the issue without claiming that this small example specifies the full implementation.

### Predicting several future tokens changes training and can aid serving

A multi-token prediction objective adds supervision for future offsets beyond the next token. If $\ell_h$ is the mean negative log-likelihood for a valid target at future offset $h$ and $\lambda_h$ its weight, an illustrative training objective is $\sum_{h=1}^{H_p}\lambda_h\ell_h$, where $H_p$ is the maximum prediction offset. End-of-sequence targets must be masked and normalized consistently. The exact conditioning and shared components depend on the architecture.

Extra predictions can train more future-aware representations and supply candidate tokens for speculative decoding, in which a main model verifies proposals. Training a head to guess several tokens does not by itself authorize emitting all guesses without verification while claiming the original model's distribution. The objective, proposal mechanism, and acceptance procedure are separate contracts; inference chapters develop the last two.

For Atlas, retain a four-column design record: historical information retained, parameters stored, parameters activated per token, and communication per step. A fixed recurrent state changes the first column. Expert sparsity primarily changes the third relative to the second. Latent caching reduces the first while keeping growth in $T$. Sparse selection reduces reads and arithmetic while requiring retained entries and a selector. This accounting makes combinations intelligible and exposes costs that a single “efficient architecture” label hides.

## 9. Limits, exercises, and worked solutions

The toy memories assume prescribed keys and values rather than learned representations. Their algebra and interference tests establish mechanism-level properties, not a language-model quality ranking. MoE accounting excludes detailed kernel and topology effects. Routing gradients are local derivatives away from selection boundaries. None of the CPU examples trains a production MoE or benchmarks accelerator throughput.

**Exercise 1 — find the false equivalence.** A developer computes causal linear attention as $Q(K^\top V)$ and reports a speedup over a masked dense implementation. Identify the error and give a two-token counterexample.

<details><summary>Worked solution</summary>

The unmasked product includes every key/value pair for every query. Take scalar queries $(1,1)$, keys $(1,1)$, and values $(2,9)$. Correct causal outputs are 2 at the first position and 11 at the second. The unmasked product outputs 11 at both positions, leaking the second value into the first. A prefix recurrence fixes causality while preserving the linear function. Adding softmax afterwards would not repair the same computation; it would introduce a different normalization whose dependence must be handled explicitly.

</details>

**Exercise 2 — quantify interference.** Starting from state $S$, use an ungated delta update with unit key $k$, write strength $\beta$, and error $e=v-S^\top k$. Derive the change in the read at another unit key $h$. What happens for identical, orthogonal, and negatively correlated keys?

<details><summary>Worked solution</summary>

The update is $\Delta S=\beta ke^\top$, so $\Delta S^\top h=\beta e(k^\top h)$. Identical keys give the full correction $\beta e$, which is exactly the intended overwrite direction. Orthogonal keys give zero change. Negative inner product reverses the sign of the disturbance. For angle 60 degrees, the inner product is one half, so half the correction appears in the other read. Exact overwrite on one unit key therefore coexists with interference on other keys; increasing state dimensions can permit more separated directions but does not automatically learn them.

</details>

**Exercise 3 — derive a stability condition for a write.** Keep one key $k\ne0$ and target $v$ fixed. Repeatedly apply the ungated delta update with constant $\beta$. Under what condition does the prediction error along that key contract?

<details><summary>Worked solution</summary>

Let $e=v-S^\top k$. After one update, the read is $S^\top k+\beta e\|k\|_2^2$, leaving error $(1-\beta\|k\|_2^2)e$. Strict contraction requires $|1-\beta\|k\|_2^2|<1$, equivalently $0<\beta\|k\|_2^2<2$. Exact correction occurs at $\beta=1/\|k\|_2^2$. At the upper boundary, the error alternates sign without shrinking; beyond it, its magnitude grows. Restricting $\beta$ to $[0,1]$ is therefore not sufficient for arbitrary unnormalized keys. This is one fixed-key update analysis, not a complete stability theorem for changing gates, keys, and surrounding nonlinear layers.

</details>

**Exercise 4 — distinguish parameter and dispatch budgets.** An MoE stores sixteen experts of 50 million parameters plus 200 million other parameters, uses top-2 routing, and processes 1,024 tokens. Compute total and active parameters, assignments, and balanced expert load. With capacity factor 1.25, how many slots are allocated per expert?

<details><summary>Worked solution</summary>

Total parameters are $200+16\cdot50=1000$ million. Approximate active parameters per token are $200+2\cdot50=300$ million. There are $1024\cdot2=2048$ assignments, averaging $2048/16=128$ per expert. Capacity is $\lceil1.25\cdot128\rceil=160$ slots per expert. This allocates 2,560 slots overall, but aggregate spare capacity does not prevent one expert from receiving more than 160 tokens. Its overflow policy remains necessary. Training-state memory depends on the billion stored parameters and their optimizer representation, not merely on 300 million activated parameters.

</details>

**Exercise 5 — explain two missing gradients.** Under top-1 routing renormalized over selected experts, why can the task gradient through the router gate vanish? Does that imply all unselected router logits always have zero gradient under every gate convention?

<details><summary>Worked solution</summary>

For selected index $j$, renormalization gives $g_j=p_j/p_j=1$. Away from a selection boundary its derivative is zero, and the task output equals the selected expert output without a varying router factor. Auxiliary losses can still train the router. Under an unrenormalized full-softmax gate, however, $\partial p_j/\partial r_i=-p_jp_i$ for $i\ne j$, which is nonzero. Thus an unselected router logit can receive a task gradient through the denominator even though its expert network was not executed. The missing gradient through a discrete change of selected set is a third issue and is not repaired merely by having denominator gradients.

</details>

**Exercise 6 — choose the right failure experiment.** Atlas forgets a revised specification in a long notebook. Propose distinct tests for a recurrent state, an indexed sparse-attention layer, and an MoE layer. Explain why one aggregate accuracy score cannot locate all three failures.

<details><summary>Worked solution</summary>

For recurrent memory, control key similarity, number of intervening facts, and overwrite order; inspect whether the latest value replaces the old one and whether unrelated reads change. This tests retention and interference. For indexed attention, mark the evidence position and measure whether the selector includes it; then compare answer accuracy with forced inclusion. This separates retrieval failure from use of retrieved information. For MoE, record expert assignments and repeat the same request under different batch loads, checking for dropped assignments or capacity-dependent routing. These tests target different boundaries: compressed representation, selected evidence, and executed parameters. Aggregate answer accuracy detects a problem but does not identify which boundary caused it.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_04.pdf): lecture sequence and architectural comparisons.
- [Mamba-2 / state-space duality](https://arxiv.org/abs/2405.21060), [Gated Delta Networks](https://arxiv.org/abs/2412.06464), and [FlashAttention](https://arxiv.org/abs/2205.14135): structured state updates and exact attention execution.
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp): primary sparse-attention report and implementation resources.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), [DeepSeekMoE](https://arxiv.org/abs/2401.06066), and [OLMoE](https://arxiv.org/abs/2409.02060): routing, expert allocation, and controlled ablations.
- [MegaBlocks](https://arxiv.org/abs/2211.15841), [ST-MoE](https://arxiv.org/abs/2202.08906), and [Sparse Upcycling](https://arxiv.org/abs/2212.05055): systems, stability, transfer, and checkpoint reuse.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): an integrated architecture with routing control, latent attention, and multi-token prediction.

The Atlas notebook, numerical costs, counterexamples, gradient checks, and design exercises are teaching constructions. Empirical findings attributed to papers remain tied to their experimental settings.
