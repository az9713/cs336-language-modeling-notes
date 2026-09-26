# Lecture 3 — Designing a transformer that can be trained and served

Atlas, our multilingual technical-library model, has reached an awkward design decision. Two candidate transformers have almost identical parameter counts. One trains without incident but serves only a few long documents concurrently. The other stores much less history but occasionally develops large attention scores. A third proposal changes the activation function and appears better until we discover that its feed-forward blocks contain 50% more parameters. Which comparison should determine the architecture?

An architecture specifies a computation, its trainable parameters, and the intermediate state that training and inference must maintain. These are different objects. Parameter count measures neither all computation nor all memory. This chapter develops one decoder model and changes it systematically, asking what each intervention preserves, what it changes, and what evidence would justify adopting it.

Our numerical design case uses twelve layers, residual width 768, twelve query heads of width 64, and an 8,192-token serving context. These are teaching choices, not recommended universal defaults. Small CPU experiments test mathematical properties; they do not establish the quality or accelerator speed of a trained Atlas model. The recording supplies the progression from normalization through activations and position to stability and serving. The examples and deductions below are independent teaching constructions.

## 1. Preserve a useful path through the residual stream [06:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=360s)

### The computation that remains fixed

Let $B$ be batch size, $T$ sequence length in tokens, $d$ residual width, and $L$ the number of layers. The residual stream $X_\ell\in\mathbb R^{B\times T\times d}$ stores the representation entering layer $\ell$, where $\ell=0,\ldots,L-1$. An embedding lookup initializes $X_0$ from token IDs. Each layer modifies these representations, and an output projection ultimately produces one score, or logit, for each possible next token. During training, positions are processed together, but a causal mask permits each position to use only itself and earlier positions.

Two transformations do the principal work. Attention, denoted $A_\ell$, mixes information across allowed positions. A feed-forward network, denoted $F_\ell$, transforms each position's features independently with shared weights. A normalization operation $N$ rescales each token vector across its $d$ features. Omitting dropout and allowing each sublayer its own normalization parameters, a serial pre-normalized layer is

$$U_\ell=X_\ell+A_\ell(N_{a,\ell}(X_\ell)),\qquad
X_{\ell+1}=U_\ell+F_\ell(N_{f,\ell}(U_\ell)).$$

The addition requires each branch to return the same shape as the residual stream. This constraint is useful: we can alter the interior of attention or the feed-forward network while keeping the interface between layers unchanged. It also shows why residual width and internal feed-forward width are different parameters.

The [original transformer paper](https://arxiv.org/abs/1706.03762) establishes attention-based sequence modeling and the basic attention/MLP decomposition. Its complete encoder-decoder architecture is not identical to this causal decoder. Treating every subsequent decoder design as a literal copy would obscure the changes that matter here.

### Why moving normalization changes differentiation

Consider one token vector $x\in\mathbb R^d$ and a branch $G:\mathbb R^d\to\mathbb R^d$. Write $J_G$ for its Jacobian, the matrix whose $(i,j)$ entry is $\partial G_i/\partial x_j$, evaluated at the relevant argument. Pre-normalization gives $y=x+G(N(x))$. Applying the chain rule,

$$J_{\rm pre}=I+J_GJ_N,\qquad
J_{\rm post}=J_N(I+J_G)$$

for the alternative post-normalized update $y=N(x+G(x))$. Here $I$ is the identity matrix. In the first expression, the residual route bypasses normalization. In the second, normalization acts on both the branch and the shortcut. Backpropagation multiplies an incoming gradient by the transpose of the corresponding Jacobian, so the distinction affects the gradient as well as the forward representation.

A scalar check isolates the mechanism. Suppose the local derivative of $G$ is 0.1 and that of $N$ is 0.5. The pre-normalized derivative is 1.05; the post-normalized derivative is 0.55. Repeating these local factors twenty times gives approximately 2.65 versus $6.42\times10^{-6}$. This is an illustrative linearization, not a simulation of an actual transformer: real Jacobians are matrices, change with data, and do not generally commute.

The identity term is therefore a route, not a guarantee. If $J_GJ_N=-I$, the pre-normalized Jacobian is zero. If its expanding directions align across layers, gradients can still grow. [Xiong and colleagues](https://arxiv.org/abs/2002.04745) provide a more specific initialization analysis and experiments connecting normalization placement to gradient scale and learning-rate warm-up. Those assumptions are much narrower than a universal claim of stable training.

For Atlas, changing placement means changing the function and its optimization dynamics. It cannot be justified solely by timing one normalization kernel. Record loss, gradient norms, activation magnitudes, and failures over complete training runs. A finite loss after ten steps only establishes that those ten steps were finite.

## 2. Normalize scale without silently changing the invariance [12:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=720s)

### Centering and rescaling solve different problems

For a token vector $x=(x_1,\ldots,x_d)$, define its mean $\mu=d^{-1}\sum_i x_i$ and feature variance $s^2=d^{-1}\sum_i(x_i-\mu)^2$. Layer normalization returns components $g_i(x_i-\mu)/\sqrt{s^2+\epsilon}+\beta_i$, with learned gain $g_i$, learned offset $\beta_i$, and positive numerical constant $\epsilon$. RMSNorm removes centering and usually the learned offset:

$$N(x)_i=g_i\frac{x_i}{\sqrt{d^{-1}\sum_j x_j^2+\epsilon}}.$$

The denominator is the root mean square, abbreviated RMS, with a small stabilizing addition. Every reduction is over one token's features, not over the batch or sequence. Normalizing across the wrong axis would couple different examples or positions and implement a different model.

For unit gain and $x=(3,4)$, the mean square is 12.5, giving approximately $(0.849,1.131)$. These components have a nonzero mean. Their mean square approaches one as $\epsilon$ becomes negligible. By comparison, centered normalization with negligible $\epsilon$ produces $(-1,1)$. The methods discard different information even though both regulate scale.

To see the distinction algebraically, let $a>0$. With $\epsilon=0$ and $x\ne0$, the numerator of RMSNorm applied to $ax$ gains $a$, and the denominator also gains $a$; they cancel. With finite $\epsilon$, cancellation is only approximate. Near the zero vector, $\epsilon$ dominates and the map is nearly linear. For negative $a$, the output changes sign. Adding a common offset to every coordinate generally changes RMSNorm, whereas centering removes that offset before LayerNorm's variance calculation.

### What the numerical constant controls

The stabilizing constant is not just a division-by-zero patch. Put $r=\sqrt{x^\top x/d+\epsilon}$ and temporarily set all gains to one. Differentiating $N(x)=x/r$ gives

$$J_N=\frac{I}{r}-\frac{xx^\top}{dr^3}.$$

For a perturbation $v$ perpendicular to $x$, the second term vanishes, so $J_Nv=v/r$. In the radial direction, $J_Nx=\epsilon x/r^3$. As $\epsilon$ tends to zero, radial sensitivity disappears: scaling the input stops changing its normalized output. At $x=0$, sensitivity is instead $1/\sqrt\epsilon$ in every direction. Thus a small $\epsilon$ prevents undefined arithmetic while still permitting a large local derivative. Learned gains further change these sensitivities.

This derivation explains why choosing a normalization constant solely from machine precision is incomplete. Its interaction with activation scale affects the actual function. It also shows why monitoring only normalized outputs can hide a growing unnormalized residual stream.

The [RMSNorm paper](https://arxiv.org/abs/1910.07467) introduces the method and evaluates the removal of centering. Its measured speed benefits depend on the tested architectures and implementations. On Atlas, a fused normalization may already combine reductions efficiently. Removing a reduction does not imply the same percentage reduction in end-to-end training time.

Our cumulative CPU module accepts a nonempty sequence of finite numbers and a finite positive $\epsilon$, returning unit-gain normalized values without modifying the input. Invalid numerical values raise `ValueError`. It uses a stable Euclidean-norm calculation; it does not simulate low-precision accumulation or learn gains.

```python
from math import isclose
from cs336_lab.architecture import rms_normalize

x = rms_normalize((3.0, 4.0))
mean_square = sum(v * v for v in x) / 2
assert isclose(mean_square, 12.5 / 12.500001)
assert rms_normalize((0.0, 0.0)) == (0.0, 0.0)
assert rms_normalize((3.0, 4.0)) != rms_normalize((4.0, 5.0))
print(tuple(round(v, 6) for v in x))
```

The printed result is `(0.848528, 1.131371)`. The final assertion is deliberately a counterexample to additive-offset invariance. It tests a property that a mistaken implementation using centering would conceal.

## 3. Allocate feed-forward capacity and expose branch dependencies [18:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1080s)

### Nonlinearities determine how features interact

Use a row vector $x\in\mathbb R^d$ for one position and an intermediate width $m$. A conventional feed-forward block is $F(x)=\phi(xW_1)W_2$, where $W_1\in\mathbb R^{d\times m}$ and $W_2\in\mathbb R^{m\times d}$. The activation $\phi$ acts separately on each component. Without it, the two matrices collapse to a single linear map and do not provide the intended nonlinear transformation.

ReLU maps a scalar $z$ to $\max(0,z)$, suppressing negative coordinates. GELU maps it to $z\Phi(z)$, where $\Phi$ is the standard normal cumulative distribution function. Its smooth weighting can retain small negative outputs. Neither description, by itself, predicts downstream quality; that requires a trained comparison.

A gated linear unit instead multiplies two learned projections component by component. SwiGLU uses the sigmoid $\sigma(z)=1/(1+e^{-z})$ and the activation $\operatorname{SiLU}(z)=z\sigma(z)$:

$$F(x)=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d.$$

Here $W_g,W_u\in\mathbb R^{d\times m}$, $W_d\in\mathbb R^{m\times d}$, and $\odot$ means coordinatewise multiplication. One branch modulates another. Calling it a gate is descriptive: SiLU values are not constrained to the interval from zero to one, so this is not a probability of opening a binary switch. Even a scalar example shows a richer interaction. If the two projected coordinates are $a$ and $b$, their product is $a\sigma(a)b$, which responds jointly to both projections.

### A fair comparison must name the budget

Ignoring biases, the conventional block has $2dm$ weights, whereas SwiGLU has $3dm$. A conventional expansion $m=4d$ uses $8d^2$ parameters. Matching that count with the gated block requires $3dm=8d^2$, hence $m=8d/3$. At Atlas width 768, widths 3,072 and 2,048 both give 4,718,592 weights.

The leading dense-multiplication work also matches under this comparison: a matrix product performs approximately two floating-point operations per weight per processed token. The elementwise gates add work and intermediate storage, so equal leading FLOPs do not imply equal wall time or peak memory. Efficient matrix dimensions may require rounding $m$ to a hardware-friendly multiple, slightly changing the count. Report the rounded count rather than retaining an idealized equality.

The [GLU variants study](https://arxiv.org/abs/2002.05202) compares nonlinear choices in transformer feed-forward layers and provides empirical motivation for gated variants. It does not prove that the best activation is independent of data, optimizer, width, or training budget. For Atlas, the useful experiment pairs matched parameter and approximate compute budgets, then reports both final validation loss and measured throughput.

### Parallel branches alter the represented computation

The serial layer lets the feed-forward branch consume attention's newly updated representation $U_\ell$. A parallel variant instead computes

$$X_{\ell+1}=X_\ell+A_\ell(N(X_\ell))+F_\ell(N(X_\ell)).$$

Both branches now consume the same normalized input. Their input projections can potentially share a larger matrix multiplication, and neither branch must wait for the other's output. But the feed-forward branch no longer transforms the information that attention has just gathered within that layer. An identity-normalization scalar example makes the difference unambiguous: if $A(x)=x$ and $F(x)=x$, the serial layer outputs $4x$, while the parallel layer outputs $3x$.

[PaLM's architecture section](https://arxiv.org/html/2204.02311v5#S2) reports a parallel formulation and a training-speed benefit in its setting, alongside scale-dependent quality observations. This supports an engineering possibility, not a universal equivalence. For Atlas, test the trained model as well as the kernel. A compiler optimization that preserves a graph and an architectural change that removes a dependency require different correctness arguments.

## 4. Make attention depend on content and relative position [30:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1800s)

### The attention score before adding position

For clarity, suppress the batch index and consider one attention head. From normalized token rows, learned projections produce query, key, and value matrices $Q,K,V\in\mathbb R^{T\times d_h}$, where $d_h$ is head width. A query asks which previous representations are relevant, a key provides a matching representation, and a value supplies the information to aggregate. These descriptions explain their roles; all three are learned numerical vectors.

For query position $i$ and allowed key position $j\le i$, define score $s_{ij}=q_i^\top k_j/\sqrt{d_h}$. Normalize scores with softmax, $p_{ij}=e^{s_{ij}}/\sum_{r\le i}e^{s_{ir}}$, and output $o_i=\sum_{j\le i}p_{ij}v_j$. Disallowed positions receive zero probability. Multiple heads calculate different mixtures, concatenate their outputs, and project back to width $d$. Atlas uses $H=12$ heads with $Hd_h=d=768$.

The square-root scale follows a simple initialization model. If query and key components are independent, zero-mean, unit-variance random variables, their dot product has variance $d_h$. Dividing by $\sqrt{d_h}$ makes its variance one. Training can violate those assumptions; the scale is not a bound on actual scores. This distinction will matter when we examine stability.

### Derive the rotation identity

Content-only scores lack an explicit distance signal. Absolute position embeddings add a position-dependent vector to the residual representation. Relative schemes instead make attention depend on the displacement between positions. Rotary position embedding, or RoPE, supplies a relative dependence through rotations of queries and keys.

For angle $\alpha$ in radians, define

$$R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix}.$$

Take one two-dimensional coordinate pair from a query $q$ and key $k$. Rotate the query at position $i$ by $i\omega$ and the key at position $j$ by $j\omega$, with fixed angular frequency $\omega$. Orthogonality gives $R(\alpha)^\top=R(-\alpha)$; angle addition gives $R(\alpha)R(\beta)=R(\alpha+\beta)$. Therefore

$$[R(i\omega)q]^\top[R(j\omega)k]
=q^\top R(-i\omega)R(j\omega)k
=q^\top R((j-i)\omega)k.$$

For fixed content vectors, position enters only through $j-i$. A higher even-dimensional head uses separate two-dimensional coordinate pairs and frequencies. The score is the sum of these contributions. Norms are preserved by each rotation, so RoPE changes orientation without changing query or key length.

The [RoFormer paper](https://arxiv.org/abs/2104.09864) defines this positional mechanism. Standard RoPE rotates queries and keys in two-dimensional planes. These definitions resolve the recording's occasional verbal or caption ambiguity about values and three-dimensional rotations. Rotating values would be an additional architectural choice, not the score identity just derived.

The CPU helper `rotate2` accepts two finite components and a finite angle, returns a new pair, and raises `ValueError` for invalid dimensions or nonfinite inputs. The following check uses nontrivial content vectors, so it tests more than the special case of two identical unit vectors.

```python
from math import isclose
from cs336_lab.architecture import rotate2

q, k = (2.0, -1.0), (0.5, 3.0)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


score = dot(rotate2(q, 0.9), rotate2(k, 2.1))
relative = dot(q, rotate2(k, 1.2))
shifted = dot(rotate2(q, 3.9), rotate2(k, 5.1))
assert isclose(score, relative, abs_tol=1e-12)
assert isclose(score, shifted, abs_tol=1e-12)
print(round(score, 6))
```

The output is approximately `-6.78297`. Simultaneously moving both positions preserves this score because their difference is unchanged. Moving real text in a deep causal model can change the content vectors and available history, so the complete model need not be translation invariant.

### Causality must survive the implementation

An architecture can satisfy every equation above and still fail because a mask is reversed. Our next CPU experiment constructs one three-position attention head, then changes a future value by a large amount. Earlier outputs must remain unchanged. The code has no training or file effects; it assumes matching query/key head widths, finite tensors, and a square causal sequence. It explicitly materializes the score matrix and therefore has quadratic storage in $T$.

```python
import torch

q = torch.eye(3, dtype=torch.float64)
k = q.clone()
v = torch.arange(6, dtype=torch.float64).reshape(3, 2)
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
scores = (q @ k.T) / (3 ** 0.5)
weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
before = weights @ v
changed = v.clone()
changed[2] += 1000
after = weights @ changed
assert torch.equal(before[:2], after[:2])
assert not torch.equal(before[2], after[2])
assert torch.allclose(weights.sum(dim=-1), torch.ones(3, dtype=torch.float64))
```

Value width need not equal query/key width; this example deliberately uses value width two, while the score scale is determined by query/key width three. The test isolates value leakage. A full model should also perturb future input tokens, thereby changing keys and queries, and check earlier outputs end to end.

Finally, a rotation is mathematically defined at positions far beyond training. That does not establish reliable length extrapolation. Atlas may encounter unfamiliar phase relationships, more distractors, or different dependencies in long technical documents. Evaluate retrieval at multiple distances and document lengths rather than inferring generalization from the formula's domain.

## 5. Treat architecture ratios as experimental variables [43:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2580s)

### Count what changes when width or depth changes

With total query, key, and value width each equal to $d$, the three input projections and one output projection contain approximately $4d^2$ weights. A conventional width-$4d$ feed-forward block adds $8d^2$, as does the matched gated block. Ignoring embeddings, biases, and normalization gains, the decoder therefore has

$$N_{\rm blocks}\approx12Ld^2.$$

Atlas's twelve layers of width 768 contain approximately 84,934,656 block parameters. Doubling width at fixed depth quadruples this term. Doubling depth doubles it but also doubles the number of serial layer transitions. For a roughly fixed count, doubling depth requires width to shrink by $\sqrt2$. The resulting matrix shapes can have different hardware efficiency even if leading arithmetic is similar.

This accounting assumes dense standard multi-head projections. Grouped-query attention later changes the key/value contribution. It also excludes vocabulary matrices: for vocabulary size $V_{\rm tok}$, one embedding table contains $V_{\rm tok}d$ weights. If the output projection shares those weights, the table is counted once; otherwise there are two such matrices. At $V_{\rm tok}=32{,}000$ and $d=768$, each adds 24,576,000 weights, a substantial fraction of this small model.

Vocabulary enlargement may reduce sequence length for Traditional Chinese or technical identifiers while increasing embedding storage and output-logit work. Thus architecture and tokenization interact through both $T$ and $V_{\rm tok}$. Compare probability losses on the same underlying held-out text using the Chapter 1 convention for bits per byte; token perplexities from different tokenizers use different units. Preserve document boundaries and preprocessing. A lossless representation alone does not make altered evaluation corpora comparable.

Rules such as an intermediate expansion near four or an approximately fixed width-to-depth ratio summarize historically useful regions of a design space. They are starting points for experiments, not identities. Likewise $Hd_h=d$ is a convenient architecture convention, not a mathematical requirement of attention. Projection matrices can map between unequal aggregate head width and residual width.

### Optimization interventions are not only overfitting interventions

Dropout randomly suppresses selected activations during training, usually scaling retained values to preserve their conditional expectation. It changes the training computation and injects noise. Weight decay shrinks parameters. Both are often introduced as regularizers, but the reason an intervention helps must be established in the actual regime.

Let $\theta_t$ be the parameter vector at step $t$, $u_t$ the optimizer's proposed gradient-based direction, $\eta_t$ the learning rate, and $\lambda$ a decay coefficient. A decoupled update has the form $\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_tu_t$. For ordinary gradient descent, adding a squared-norm penalty produces a closely related shrinkage term. With coordinatewise adaptive scaling, inserting the penalty into the gradient and applying shrinkage outside the adaptive transformation are generally different operations. [AdamW's primary paper](https://arxiv.org/abs/1711.05101) develops this distinction.

Even without an obvious training/validation gap, shrinkage can alter parameter scales and subsequent optimization. For a simplified scale-invariant objective $f(a\theta)=f(\theta)$ with $a>0$, differentiation implies $\nabla f(a\theta)=\nabla f(\theta)/a$. Increasing parameter norm changes the gradient magnitude despite preserving the represented objective value. Shrinking the norm can therefore affect the size of later directional changes. This argument identifies a possible mechanism; real networks only partly satisfy the invariance, and adaptive optimizers introduce additional state.

One pass over a corpus also does not prove that memorization or overfitting is absent. Duplicate documents can repeat examples within that pass, and rare distinctive sequences can behave differently from average loss. For Atlas, keep validation data separate, measure train and validation loss under the same evaluation procedure, and vary decay jointly with the learning-rate schedule. If both losses improve together, the result is compatible with improved optimization. It does not, by itself, identify the causal mechanism or certify privacy.

## 6. Distinguish numerical stability from changing the objective [65:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3900s)

### A harmless probability symmetry can hide drifting logits

At one output position, let $z_v$ be the logit for vocabulary item $v$, $Z=\sum_v e^{z_v}$ the partition function, and $p_v=e^{z_v}/Z$ its probability. For target $y$, cross-entropy is $-z_y+\log Z$. Adding the same constant $a$ to every logit multiplies both numerator and denominator by $e^a$, leaving all probabilities and cross-entropy unchanged.

This symmetry leaves a common-offset direction unconstrained by the likelihood. A z-loss adds $\mathcal L_z=\lambda_z(\log Z)^2$, where $\lambda_z\ge0$ is its coefficient. Since $\partial\log Z/\partial z_v=p_v$, its derivative is

$$\frac{\partial\mathcal L_z}{\partial z_v}=2\lambda_z\log Z\,p_v.$$

For fixed probability ratios, shifting logits by $a=-\log Z$ makes the new partition function one and the penalty zero. This chooses an offset without changing those ratios in that restricted thought experiment. During actual training, the parameters couple offsets and ratios, so the added gradient can also affect predictions. A nonzero coefficient is an objective choice, not an exact arithmetic rearrangement.

By contrast, stable log-sum-exp changes evaluation while preserving the mathematical function. Let $m=\max_v z_v$. Then $\log Z=m+\log\sum_v e^{z_v-m}$. All exponential arguments are nonpositive, avoiding overflow from large positive logits. The computed partition function still includes $m$; forgetting to restore it would change z-loss even though softmax probabilities could remain correct.

The following check compares logits differing by 1,000. It uses PyTorch's stable operations on CPU, returns no persistent state, and tests that probability invariance does not imply penalty invariance.

```python
import torch

z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
shifted = z + 1000
assert torch.allclose(torch.softmax(z, 0), torch.softmax(shifted, 0))
a = torch.logsumexp(z, 0)
b = torch.logsumexp(shifted, 0)
assert torch.allclose(b - a, torch.tensor(1000.0, dtype=z.dtype))
assert b.square() > a.square()
```

### Attention magnitude has its own failure modes

The attention scale $1/\sqrt{d_h}$ corrects an initialization variance model. It cannot prevent learned query and key norms from growing. By the Cauchy–Schwarz inequality, $|q^\top k|\le\|q\|_2\|k\|_2$, where $\|\cdot\|_2$ is Euclidean length. Explicitly normalizing these vectors controls this upper bound, but the exact normalization, learned gains, and score scale determine its value. “QK normalization” is a family of choices, not one unique formula.

If each vector is normalized to unit length and the original $1/\sqrt{d_h}$ factor is retained, scores lie between $-1/\sqrt{d_h}$ and $1/\sqrt{d_h}$. This can make attention much less selective than intended. A learned or fixed additional scale changes that behavior. Stability and useful selectivity must be considered together.

Soft-capping replaces a score $z$ by $\widetilde z=c\tanh(z/c)$ for positive cap $c$. Its magnitude is less than $c$, and its derivative is $1-\tanh^2(z/c)$. Large inputs approach the cap while receiving progressively smaller gradients. A cap therefore changes both attention probabilities and learning dynamics. It is not interchangeable with subtracting a maximum before exponentiating, which preserves softmax probabilities exactly.

For two competing capped scores, the largest possible difference approaches $2c$, so their probability odds cannot exceed $e^{2c}$. For a vocabulary of size $V_{\rm tok}$ with all output logits capped, even the most favored token has probability at most $1/[1+(V_{\rm tok}-1)e^{-2c}]$. This independently derived bound exposes a cost of capping: sufficiently small caps impose a confidence ceiling. Atlas should log score distributions and failure incidence while measuring final quality; “all outputs stayed finite” is only one outcome dimension.

## 7. Design the history that inference must retain [75:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4500s)

### Cache vectors that a future query can reuse

Prefill processes an existing prompt and constructs its intermediate representations. Decode then adds new tokens, usually one per request per step. In a causal decoder with unchanged weights and positional conventions, an earlier token's key and value in a layer do not depend on future tokens. They can be retained and reused instead of recomputed.

The KV cache stores those keys and values. It does not ordinarily store all past query–key dot products: the next token has a new query, so its scores must be computed against the retained keys. This distinction corrects an imprecise explanation in the recording and determines the memory scaling. Stored vectors grow linearly with context length; a complete score matrix would grow quadratically.

Let $H_{kv}$ be the number of distinct key/value heads and $b$ bytes per cached component. There are $LBTH_{kv}d_h$ key components and the same number of value components. Therefore

$$M_{KV}=2LBTH_{kv}d_hb.$$

This is logical tensor storage in bytes. It excludes padding, allocation fragmentation, weights, temporary scores, and other workspace. It also assumes every layer uses the same head count and cache length. A mixed architecture requires summing the corresponding per-layer sizes.

With ordinary multi-head attention, Atlas has $H_{kv}=H=12$. At batch one, 8,192 tokens, and two bytes per component, its cache uses 301,989,888 bytes, or 288 MiB. One MiB is $2^{20}$ bytes. This history is separate from the roughly 85 million block parameters counted earlier; no contradiction arises when two equal-parameter models have different serving capacities.

### Share history without sharing every query

Multi-query attention uses one shared key head and one shared value head for all query heads. Grouped-query attention, abbreviated GQA, uses an intermediate $H_{kv}$; an equal partition requires $H$ to be divisible by $H_{kv}$. With twelve query heads and three KV heads, groups of four queries share the same keys and values. Queries remain distinct, so their scores and mixtures can differ even within a group.

Atlas's GQA cache is one quarter of the multi-head cache: 72 MiB per request. If exactly 1,024 MiB is available for cache after other allocations, the logical upper bounds on concurrent requests are $\lfloor1024/288\rfloor=3$ and $\lfloor1024/72\rfloor=14$. These are capacity bounds, not throughput predictions; real allocation and latency constraints can lower them.

The [GQA paper](https://arxiv.org/abs/2305.13245) studies intermediate sharing and conversion of existing models, providing empirical evidence for a useful quality/efficiency compromise. Sharing reduces distinct representational channels in keys and values, so a cache calculation cannot prove unchanged quality. It also reduces key/value projection parameters: with full query/output widths, attention weights total $2d^2+2dH_{kv}d_h$, rather than $4d^2$ when $H_{kv}d_h=d$.

The cumulative cache helper accepts positive integer dimensions and an element size in bytes, rejects invalid inputs, and returns an integer count without allocating a tensor. Thus the check runs on CPU even when describing a cache too large to allocate conveniently.

```python
from cs336_lab.architecture import kv_cache_bytes

mha = kv_cache_bytes(12, 1, 8192, 12, 64)
gqa = kv_cache_bytes(12, 1, 8192, 3, 64)
assert mha == 288 * 2**20
assert gqa == 72 * 2**20
assert mha == 4 * gqa
print((1024 * 2**20) // mha, (1024 * 2**20) // gqa)
```

It prints `3 14`. This directly tests the serving consequence of one architectural choice while leaving performance claims to measurement.

### A local window changes the path to distant information

Sliding-window attention permits each position to inspect only itself and the preceding $w-1$ positions, where $w$ is window size. For $L$ stacked local layers without other long-range mixing, the maximum backward dependency distance is $L(w-1)$: each layer adds at most $w-1$ to a path through the computation graph. This is reachability, not a guarantee that the representation preserves the desired fact.

A model with window 512 and twelve local layers has a backward reach of at most 6,132 token positions. It cannot depend on a fact 7,000 positions earlier through those paths alone. Interleaving a global attention layer changes the graph and can restore direct access at that layer. It also requires retaining the relevant global keys and values, so its memory must be counted separately.

For Atlas's technical documents, a specification near the beginning may be needed near the end. Test that exact dependency at varying distances, alongside average language-model loss. A model can improve aggregate loss while losing a rare but consequential retrieval capability. Window size, head sharing, and positional encoding solve different problems; none can serve as a substitute for evaluating the actual document task.

## 8. Limits, exercises, and worked solutions

This chapter fixes a dense causal decoder and examines local architectural interventions. It does not establish an optimal configuration, predict large-scale training stability from toy examples, or claim CPU timings transfer to GPUs. Normalization invariances are conditional mathematical statements; paper results are empirical within their settings; cache counts are accounting identities under explicit storage assumptions. Keep these evidence classes separate when selecting Atlas's next experiment.

**Exercise 1 — differentiate an invariance.** For unit-gain RMSNorm with positive $\epsilon$, derive the Jacobian's action on a vector perpendicular to $x$ and on $x$ itself. Explain what happens as $\epsilon$ approaches zero and why the origin must be handled separately.

<details><summary>Worked solution</summary>

Set $r=\sqrt{x^\top x/d+\epsilon}$. Differentiating the scalar denominator gives $\partial r/\partial x_j=x_j/(dr)$. Applying the product rule to $x_i/r$ gives $\partial N_i/\partial x_j=\delta_{ij}/r-x_ix_j/(dr^3)$, where $\delta_{ij}$ is one when $i=j$ and zero otherwise. If $x^\top v=0$, the second term vanishes and $J_Nv=v/r$. Multiplying by $x$ instead yields $x(1/r-\|x\|_2^2/(dr^3))=\epsilon x/r^3$. For nonzero $x$, this radial response tends to zero as $\epsilon$ vanishes, matching positive-scale invariance. At the origin, $r=\sqrt\epsilon$ and $J_N=I/\sqrt\epsilon$; setting $\epsilon=0$ would make normalization undefined there. Taking a limit away from the origin does not define a derivative at it.

</details>

**Exercise 2 — control the activation comparison.** An engineer compares a two-matrix MLP and a three-matrix gated MLP at width $d=768$, both with intermediate width 3,072. The gated model improves validation loss. What has the experiment established, and how would you repair the comparison?

<details><summary>Worked solution</summary>

The conventional block contains $2\cdot768\cdot3072=4{,}718{,}592$ weights. The gated block contains $7{,}077{,}888$, exactly 50% more. Leading matrix-multiplication work increases by the same factor, so the observation establishes the performance of two differently resourced systems, not an isolated benefit of gating. Reduce the gated intermediate width to 2,048 to match its $3dm$ count to the conventional $2d(4d)$ count. Hold tokenizer, data order, training tokens, optimizer protocol, and evaluation fixed; repeat when run variability matters. Measure actual throughput and peak memory because gate intermediates and kernel efficiency can differ despite matched leading counts. A deployment decision may still favor the larger system, but that is a different stated budget.

</details>

**Exercise 3 — test the boundary of relative position.** Prove simultaneous-shift invariance of one RoPE score for fixed content vectors. Give a reason the complete decoder can nevertheless change its answer when text is shifted later in a prompt.

<details><summary>Worked solution</summary>

For shift $a$, the rotated score becomes $q^\top R(((j+a)-(i+a))\omega)k=q^\top R((j-i)\omega)k$. Summing over coordinate pairs preserves equality. This proof holds $q$ and $k$ fixed. Adding a prefix to move text changes what earlier positions can attend to, which can change the hidden representations and therefore the projected queries and keys. Truncation, window boundaries, or a different available context can also change the computation graph. The algebra proves a positional property of fixed-vector score construction, not invariance of arbitrary end-to-end prompts.

</details>

**Exercise 4 — derive a confidence ceiling.** Suppose all output logits are soft-capped to the interval $(-c,c)$. Bound the probability of the most likely token for vocabulary size $V_{\rm tok}$. Evaluate the limiting bound when $c=2$ and $V_{\rm tok}=100$.

<details><summary>Worked solution</summary>

The largest probability is approached by moving the preferred logit toward $c$ and all others toward $-c$. Its limiting value is $e^c/[e^c+(V_{\rm tok}-1)e^{-c}]=1/[1+(V_{\rm tok}-1)e^{-2c}]$. With the specified values, the denominator is $1+99e^{-4}\approx2.81325$, so the probability cannot exceed approximately 0.3555. Finite uncapped inputs to tanh make the endpoints unattainable, hence the actual value is strictly below that supremum. Stable softmax implemented by subtracting a maximum imposes no analogous confidence ceiling: it changes numerical evaluation, not score differences. A small output cap can therefore conflict with learning very confident predictions.

</details>

**Exercise 5 — separate capacity, reachability, and quality.** Atlas has 1 GiB free for cache and uses the twelve-layer, width-768 configuration in this chapter. Compare twelve KV heads with three KV heads at context 8,192. Then consider replacing all attention with a window of 512. Can a final-position prediction depend on a fact 7,000 positions earlier?

<details><summary>Worked solution</summary>

The full-context caches are 288 MiB and 72 MiB per request. Integer division into 1,024 MiB gives logical concurrency limits of three and fourteen requests. Neither number includes fragmentation or latency constraints, and the smaller cache does not prove equal quality. For twelve purely local layers, each dependency edge crosses at most 511 positions. A path through all layers reaches at most $12\cdot511=6132$ positions backward, so a fact 7,000 positions earlier is unreachable under the assumed graph. A global layer could alter that result, with its own storage and compute costs. The experiment should therefore report cache-supported capacity, measured serving latency, and a distance-controlled retrieval test separately. Combining them into one score requires explicit priorities rather than an unstated preference for parameter count.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=lVynu4bo1rY) and [official lecture slides at the archived course revision](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_03.pdf): topic sequence and lecture-time architectural comparisons.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745): foundational attention and a specific analysis of normalization placement.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) and [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): normalization and gated feed-forward definitions and experiments.
- [RoFormer](https://arxiv.org/abs/2104.09864): rotary positional encoding; [PaLM](https://arxiv.org/abs/2204.02311): a concrete large-model architecture with parallel branches.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): adaptive optimization and decoupled shrinkage; [GQA](https://arxiv.org/abs/2305.13245): grouped sharing of keys and values.

The numerical Atlas configuration, Jacobian worked checks, capped-confidence bound, cache-capacity calculation, CPU experiments, and exercise constructions are instructional additions. They should not be read as measurements reported by the cited papers.
