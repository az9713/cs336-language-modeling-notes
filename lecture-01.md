# Lecture 1 — Choosing the units of language-model computation

Atlas is a proposed language model for a small technical library. Its documents contain English explanations, Traditional Chinese notes, source code, equations written as text, and unfamiliar identifiers. The first prototype reports an encouraging validation loss, yet Chinese requests exhaust its context window sooner than English requests. A second tokenizer lowers the number of tokens but raises the loss per token. Which system improved? Before buying more computation, the team needs to know what its units measure.

A **language model** assigns probabilities to sequences of discrete symbols. A **tokenizer** converts the library's text into those symbols and converts model outputs back into text. Tokenization therefore determines both the prediction problem and part of its computational cost. The running Atlas case is an original teaching construction; the recording supplies the order of the engineering questions and the progression toward byte-pair encoding.

## 1. The system being optimized is larger than its neural network [03:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=180s)

### Define a design and an outcome before choosing an algorithm

Atlas has a fixed training budget and must answer requests on a smaller serving machine. Let a **design** be a tuple $z=(\tau,a,d,h,s)$: tokenizer $\tau$, model architecture $a$, data recipe $d$, optimization hyperparameters $h$, and systems implementation $s$. An architecture specifies the arrangement of parameterized operations. Hyperparameters control training without being learned as ordinary model weights; examples include learning rate and batch size. A data recipe specifies extraction, filtering, and sampling. The implementation determines how the resulting operations execute on hardware.

For design $z$, let $L(z)$ be a declared held-out prediction loss, $C(z)$ training arithmetic in floating-point operations, $M(z)$ peak device-memory occupancy in bytes, and $T(z)$ elapsed training time in seconds. **Held-out** means excluded from optimization and model-selection decisions that would invalidate the evaluation claim. A first design problem is

$$\min_z L(z)\quad\text{subject to}\quad
C(z)\le C_{\max},\quad M(z)\le M_{\max},\quad T(z)\le T_{\max}.$$

The subscript “max” denotes an externally specified limit. The objective is conditional on the chosen evaluation distribution. It does not claim that one number captures every useful capability. For Atlas, exact preservation of identifiers and separate English and Chinese task scores remain additional acceptance criteria.

Arithmetic, memory, and time are different resources. A parameter tensor occupies memory even when no operation is running. An operation can wait for data despite having few arithmetic instructions. Communication can delay a computation whose local memory fits comfortably. Replacing three constraints with a single vague “compute budget” can conceal the reason a design is infeasible.

**Worked example.** Suppose two implementations perform respectively 100 and 120 arbitrary arithmetic units while requiring respectively 80 and 30 milliseconds of memory service. At an arithmetic rate of one unit per millisecond, optimistic fully overlapped runtimes are $\max(100,80)=100$ and $\max(120,30)=120$ milliseconds. At ten units per millisecond they become $\max(10,80)=80$ and $\max(12,30)=30$ milliseconds. The ranking reverses without either algorithm changing. These are lower-bound accounting examples, not measured hardware results.

The consequence for Atlas is concrete. A tokenizer that saves arithmetic may still increase total time through a larger vocabulary projection or slower preprocessing. Measure the completed workload, then inspect its components to explain the result. Component measurements diagnose the system; they do not replace the system objective.

### What small experiments can establish

A small implementation is useful for checking **mechanics**: the precise transformation from inputs to outputs, including gradients and numerical conventions. It can also establish a method of experimentation: controlled comparisons, resource accounting, and reproducible checkpoints. It provides weaker evidence about which architecture or data mixture wins at a much larger scale.

One reason is a change in the fraction of work spent in each component. If a modification accelerates a component occupying fraction $f$ of runtime by factor $r$, while the remaining work is unchanged and executed serially, the new normalized runtime is $(1-f)+f/r$. Hence total speedup is

$$S=\frac{1}{(1-f)+f/r}.$$

Here $f$ lies between zero and one, and $r\ge1$. The derivation simply partitions the original unit runtime into affected and unaffected portions. If attention occupies 20% of a workload, making it infinitely fast leaves 80% of the original time, so speedup cannot exceed 1.25 under these assumptions. If attention occupies 80%, the corresponding limit is five. Overlap and bottleneck shifts require a richer model, but the example explains why an optimization's importance can change with scale.

For Atlas, preserve a testable statement such as “this tokenizer round-trips all accepted text” separately from an empirical statement such as “this tokenizer improves Chinese question answering.” The first follows from an implementation invariant plus input assumptions. The second requires data and measurement. Scaling a correct invariant does not automatically scale a quality advantage.

## 2. A probability model needs a specified representation [11:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=660s)

### From text to a conditional distribution

Let $x$ be an accepted Unicode string, and let $E_\tau(x)=(y_1,\ldots,y_n)$ be its encoding under tokenizer $\tau$. Each token identifier $y_t$ belongs to a finite vocabulary $\mathcal V=\{0,\ldots,V-1\}$ with $V$ entries. The integer $n$ is the encoded sequence length. A parameter vector $\theta$ determines conditional probabilities $p_\theta(y_t\mid y_{<t})$, where $y_{<t}$ denotes the tokens preceding position $t$.

The **autoregressive factorization** writes the probability of a fixed-length sequence as

$$p_\theta(y_{1:n})=\prod_{t=1}^{n}p_\theta(y_t\mid y_{<t}).$$

This is an application of the probability chain rule, not a special neural-network theorem. The modeling choice is how to compute each conditional probability. A variable-length generative model additionally needs a termination convention, often an end-of-sequence token. A training example's last observed token is not automatically a statement that all possible continuations terminate there.

Training commonly minimizes **negative log-likelihood**, abbreviated NLL. With natural logarithms, the loss of the sequence is $-\log p_\theta(y_{1:n})$. The logarithm transforms the product into a sum:

$$\operatorname{NLL}(y_{1:n})=-\sum_{t=1}^{n}\log p_\theta(y_t\mid y_{<t}).$$

The unit of this quantity is a **nat**, the information unit associated with natural logarithms. Mean token loss divides NLL by the number of evaluated tokens. **Perplexity** is the exponential of that mean. It can be interpreted as the effective number of equally likely choices per token for a hypothetical uniform predictor with the same mean log loss.

**Worked example.** Suppose the probabilities assigned to three observed targets are $0.5$, $0.25$, and $0.8$. Their joint probability is $0.1$, NLL is $\log 10\approx2.303$ nats, mean loss is approximately $0.768$, and perplexity is approximately $2.154$. The predictor is not literally choosing among 2.154 symbols. Perplexity expresses an average uncertainty on the tokenizer's units.

For Atlas, a Chinese word split into several tokens creates several conditional predictions where an English word might create one. Comparing their raw token perplexities confounds model uncertainty with segmentation. The appropriate comparison must state what text and what unit are held fixed.

### Representation is part of the experiment

A model checkpoint contains learned numerical parameters. An **open-weight** release makes those parameters available under stated terms. It does not necessarily provide the training documents, preprocessing, sampling schedule, optimizer state, or all implementation details needed to reproduce training. Reproducibility concerns the transformation from evidence and configuration to a result, not only access to the final tensor files.

The distinction affects small-model experiments. The [TinyStories study](https://arxiv.org/abs/2305.07759) investigates small language models in a deliberately restricted synthetic story domain. It provides evidence about what models can learn from that distribution. It does not establish that the same size is sufficient for Atlas's mixed technical library. A simplified domain is an instrument for studying a mechanism, and its simplifications must remain visible when interpreting the result.

Archive the tokenizer together with the checkpoint. If token 412 means one byte sequence during training and another at inference, the numerical embedding row has changed meaning even though the model file is intact. The vocabulary, merge order, text normalization, special-token policy, and serialization format are therefore part of model identity. A tokenizer version is not a cosmetic dependency.

## 3. Building a model is a sequence of coupled resource decisions [27:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1620s)

### The lifecycle preserves the same design variables

Atlas first tokenizes documents, then trains a network to predict their tokens. The optimizer updates parameters from prediction errors. Systems work changes how those same updates execute. Scaling experiments ask how to change model size and training exposure as available work grows. Data work changes the distribution that supplies the errors. Post-training changes which responses receive learning pressure. Each stage changes a component of $z$ rather than replacing the design problem with an unrelated one.

A **floating-point operation**, or FLOP, is one elementary arithmetic operation under a declared counting convention. Multiplication and addition are usually counted separately, even when hardware executes a fused instruction. Let $N$ denote the number of relevant dense model parameters and $D$ the number of training tokens. The approximation $C\approx6ND$ is a useful preliminary training-work estimate. Its common justification is roughly $2ND$ for parameter-associated forward matrix multiplication and twice that amount for its backward computations. Attention work, embeddings, sparsity, and implementation details can make it inaccurate; Lecture 2 derives its domain of use.

For an illustrative $N=10^8$ and $D=2\times10^9$, the estimate is $1.2\times10^{18}$ FLOPs. Doubling $N$ at fixed $D$ doubles this term. Doubling both quadruples it. A purportedly small hyperparameter change can therefore alter the resource requirement by orders of magnitude once it changes dimensions or training exposure.

A **scaling recipe** is a rule mapping a resource budget to a complete training configuration. Predictable behavior requires more than fitting a line through a few losses. The architecture, optimizer, data distribution, and stopping rule must change in controlled ways. A scale-dependent instability invalidates an otherwise attractive extrapolation. Atlas should reserve some larger pilot runs as genuine tests of the prediction, rather than fitting every observed result and calling the fit a forecast.

### Training cost and serving cost ask different questions

**Inference** uses a trained model to produce outputs. Its **prefill** phase processes an existing prompt; its **decode** phase produces successive output tokens. Repeated decoding can reuse stored intermediate states, but the states consume memory and must be accessed. A smaller model trained longer can be attractive when it will serve enough future requests to repay its extra training cost.

Let $C_{\rm train}$ be training work, $Q$ the number of lifetime requests, and $c_{\rm request}$ expected inference work per request under a specified length distribution. A simple lifetime-work model is $C_{\rm life}=C_{\rm train}+Qc_{\rm request}$. If design A uses more training work but saves $\Delta c>0$ per request, the extra training investment $\Delta C$ is repaid in this accounting model when $Q>\Delta C/\Delta c$. This is not a financial break-even theorem: hardware utilization, energy, latency requirements, and pricing remain separate quantities.

Atlas also needs two kinds of evaluation. A smooth development metric helps discriminate close training recipes. A deployment task measures whether the product performs the intended work. A lower prediction loss may coexist with poor identifier copying or unreliable technical answers. Preserve those outcomes separately until an explicit decision rule says how to trade them.

The data recipe belongs in that evaluation loop. Repeated documents consume training exposure that could have been spent on new examples. Aggressive filtering can remove rare technical styles the library actually needs. **Supervised fine-tuning** learns from selected target responses, while preference optimization and reinforcement learning use comparative judgments or rewards. They change the feedback signal; none rescues a silently corrupted text representation. This makes tokenization a suitable first implementation problem: its local correctness can be established before tackling uncertain learning outcomes.

## 4. A tokenizer must state exactly what it preserves [65:05](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3905s)

### Strings, code points, bytes, and tokens

Unicode identifies abstract characters using numerical **code points**. A displayed symbol can involve more than one code point; an accented character, for example, may have composed and decomposed representations. UTF-8 is a variable-length encoding of Unicode scalar values into bytes. A byte is an eight-bit quantity taking an integer value from 0 through 255. A token identifier is a vocabulary index, whose associated content may comprise one byte or several.

These distinctions matter operationally. Counting displayed symbols does not reliably predict bytes, and counting bytes does not determine token count for a learned tokenizer. A Traditional Chinese character commonly requires three UTF-8 bytes, while many familiar English letters require one. This is a statement about the encoding of those characters, not about their information content or linguistic complexity.

Let $U(x)$ be the UTF-8 byte sequence of an accepted string. Let $E_\tau$ be encoding into token identifiers and $D_\tau$ decoding into a string. A lossless tokenizer should satisfy

$$D_\tau(E_\tau(x))=x\qquad\text{for every accepted }x.$$

The accepted domain must be stated. Our reference implementation accepts strings encodable by Python's strict UTF-8 encoder; isolated surrogate code points are excluded. It performs no Unicode normalization. If a pipeline normalizes two distinct input strings to the same representation, its round-trip guarantee concerns normalized text, not the exact original string.

Round-trip preservation is a correctness property. It says nothing about whether tokenization makes learning efficient. Encoding every byte separately satisfies it with a vocabulary of 256 entries, but can leave a long prediction sequence. Assigning every complete training document its own identifier compresses the training corpus impressively while generalizing poorly and expanding the vocabulary drastically.

### The compression–vocabulary tradeoff

For a nonempty string with $B$ bytes and $n$ tokens, define compression ratio $\rho=B/n$ bytes per token. Larger $\rho$ means fewer token positions for that same byte sequence. It does not mean less information has to be modeled; the information is packaged into larger units.

Suppose Atlas has embedding width $d$, meaning each token maps to a vector with $d$ real entries. An input embedding table has $Vd$ parameters. An untied output projection has another $Vd$ parameters. **Weight tying** reuses the same parameter table for compatible input and output roles; it reduces distinct parameters but does not eliminate the output computation.

Increasing vocabulary from 32,000 to 64,000 at $d=768$ adds $2(32{,}000)(768)=49{,}152{,}000$ parameters when the tables are untied. Two-byte storage alone adds 98,304,000 bytes, approximately 93.75 MiB, where one MiB is $2^{20}$ bytes. Gradients and optimizer state increase training memory further. “Shorter sequences” must be assessed alongside this cost.

The quadratic size of a dense attention score matrix introduces another tradeoff. A sequence of $n$ positions has $n^2$ position pairs. If the same text requires half as many tokens, that pair count falls by a factor of four. Most projection work falls only linearly with sequence length. A larger vocabulary can increase output-layer cost at the same time. Atlas should compare measured end-to-end throughput at matched raw text, with per-language length distributions, rather than declaring the largest compression ratio the winner.

## 5. Why characters, bytes, and words fail in different ways [68:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4108s)

### Coverage and granularity are independent design choices

A character tokenizer represents code points, or a defined subset of them, directly. It provides recognizable units but allocates vocabulary entries to rare characters and can still produce long sequences. A byte tokenizer has a compact complete base alphabet for UTF-8 data. Its price is additional prediction positions and the need to learn relationships across byte boundaries.

A word tokenizer uses a segmentation rule, such as splitting on whitespace, and assigns one identifier per observed word. Its weakness is an **out-of-vocabulary** item: a segment with no assigned entry. Mapping all such items to one unknown symbol destroys distinctions. Two new variable names can become identical model inputs. That is unacceptable for Atlas even if its average prediction score looks reasonable.

The problem is not cured by storing more common words indefinitely. Technical libraries continually introduce identifiers, version strings, formulas, and compounds. Whitespace also fails to define equivalent linguistic units across languages. Coverage requires a fallback representation, while efficiency requires common sequences to be grouped. Subword methods attempt to satisfy both requirements.

| Representation | Guaranteed coverage under stated alphabet | Typical pressure on Atlas |
|---|---|---|
| Observed whole words | No, unless a lossless fallback is added | New identifiers become unknown or fragmented by ad hoc rules |
| Unicode code points | Yes only for the implemented code-point domain | Many rare entries and long sequences |
| UTF-8 bytes | Yes for valid UTF-8 strings | More prediction steps for multibyte text |
| Byte-based learned merges | Yes when all base bytes remain available | Vocabulary cost and training-distribution bias |

Each row describes a mechanism, not a leaderboard. A language or task with more tokens per byte spends more context positions on the same raw volume. Conversely, an unfamiliar byte pattern can remain representable without being understood. Coverage is a necessary interface property; capability still depends on training.

### Normalization can create a hidden failure

Suppose two source identifiers differ only in a Unicode representation that the tokenizer normalizes away. The text-processing pipeline may treat them as equivalent even though a downstream compiler or database distinguishes them. A round-trip test run only after normalization will pass. The original identifier has nevertheless been lost.

A correct Atlas ingestion test begins before normalization and checks the intended invariant. If canonicalization is an application decision, record the original bytes and the transformation. If exact reconstruction is required, keep normalization out of the reversible path. This is the first instance of a general rule used throughout the course: test the semantic property at the boundary where it matters, rather than testing an easier surrogate inside the pipeline.

## 6. Byte-pair encoding learns reusable chunks [72:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4320s)

### A merge changes symbols while preserving bytes

**Byte-pair encoding**, abbreviated BPE, repeatedly replaces frequent adjacent token pairs with new tokens. In our byte-based variant, the initial vocabulary contains all 256 single-byte values. Let $b(i)$ denote the byte string associated with token identifier $i$. Initially, $b(i)$ is the one-byte string containing $i$.

Training operates on a list of documents. For each document, count adjacent ordered pairs of current token identifiers. Counts may include overlapping occurrences, but replacements must not overlap. Choose the pair with the largest count, resolve ties deterministically, and create a fresh identifier $k$ with $b(k)=b(i)\Vert b(j)$, where $\Vert$ denotes byte-string concatenation. Replace occurrences of pair $(i,j)$ from left to right. Repeat until the merge budget is exhausted or no pair remains.

The choice of document boundary is part of the algorithm. Our trainer counts within documents and never learns a pair spanning two different documents. Concatenating a corpus without boundaries would permit artificial cross-document chunks. More elaborate tokenizers also use **pre-tokenization**, an initial split into regions within which merges are allowed. That can reduce work and control what kinds of chunks are learned, while constraining the reachable vocabulary.

The [Sennrich, Haddow, and Birch paper](https://arxiv.org/abs/1508.07909) establishes subword BPE as a method for handling rare words in neural machine translation. Its setting should not be confused with every detail of modern byte-level tokenizers. The [tiktoken repository](https://github.com/openai/tiktoken) supplies a primary implementation reference for learned merge ranks, byte-based content, and special-token handling. Our code isolates the merge mechanism and intentionally omits that production interface complexity.

### Work the overlapping case completely

For training document `ababab`, the initial identifiers are $(97,98,97,98,97,98)$. Pair $(97,98)$ occurs three times, and $(98,97)$ occurs twice. The first merge creates identifier 256 with content `ab`, producing $(256,256,256)$. The next pair $(256,256)$ has two overlapping occurrences. A left-to-right replacement creates identifier 257 with content `abab` and produces $(257,256)$.

The second replacement removes one pair, not two, because the middle token cannot participate in two replacements simultaneously. This is a useful distinction between a count used to select a pair and the reduction actually achieved by its application. The encoded document now has two tokens for six bytes, so its compression ratio is three bytes per token.

Decoding concatenates `abab` and `ab`, recovering the original bytes. The invariant can be proved by induction. Before a replacement, the relevant contribution to decoded content is $b(i)\Vert b(j)$. After it, the contribution is $b(k)$, defined to be exactly that concatenation. Other tokens remain unchanged. Thus each replacement preserves decoded content, and any finite sequence of replacements preserves it as well.

The proof establishes losslessness for encoding paths generated by these merges. It does not imply that every arbitrary token sequence decodes to valid Unicode, nor that encoding a decoded arbitrary token sequence returns the same identifiers. Different segmentations can concatenate to the same bytes. The round-trip guarantee goes from accepted text through the deterministic encoder and back to text.

### Executable reference and its contract

The cumulative [CPU reference module](cs336_lab/tokenization.py) implements this model. Training accepts a sequence of strings and a nonnegative integer merge count. It returns an immutable vocabulary and ordered merge list. It neither changes the input documents nor reads files or network resources. Invalid argument types or negative budgets raise exceptions. Encoding performs strict UTF-8 conversion; decoding checks identifier bounds, concatenates bytes, then performs strict UTF-8 decoding.

```python
from cs336_lab.tokenization import BytePairTokenizer

tokenizer = BytePairTokenizer.train(["ababab"], merge_count=2)
assert tokenizer.merges == ((97, 98), (256, 256))
assert tokenizer.encode("ababab") == [257, 256]
assert tokenizer.decode([257, 256]) == "ababab"
assert tokenizer.decode(tokenizer.encode("學習模型")) == "學習模型"
```

All assertions pass in the companion test run. The Chinese string is not in the training corpus. It remains encodable because the base bytes remain present. This checks coverage, not whether the resulting segmentation is efficient or the model understands the string.

The core replacement procedure exposes the nonoverlap rule directly:

```python
def replace_pair(tokens, pair, replacement):
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tuple(tokens[i:i + 2]) == pair:
            result.append(replacement)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


assert replace_pair([97] * 5, (97, 97), 256) == [256, 256, 97]
```

The function returns a new token list. Incrementing by two after a match prevents a consumed token from being reused. Incrementing by one otherwise guarantees progress even when no merge matches. A five-byte run has four overlapping adjacent pairs, yet only two nonoverlapping replacements occur.

Our trainer rescans the current corpus for every merge; our encoder tries the learned merges in training order. These choices make the reference inspectable but expensive. If $K$ merges are considered over $B$ initial bytes, a crude upper work bound is proportional to $KB$, before language-runtime overhead. Efficient implementations index active pairs, update local neighborhoods, and avoid reconsidering irrelevant merges. The reference provides an oracle for correctness comparisons, not a competitive tokenizer benchmark.

## 7. Researched extension — compare the same text in the same units

### A loss ranking can reverse when its denominator changes

Atlas's two tokenizers must be compared on an identical held-out byte corpus, with matching document boundaries and termination conventions. Let $B$ be its total byte count and $\mathcal N$ the summed NLL in nats under a declared canonical tokenization. Define the score

$$\operatorname{BPB}=\frac{\mathcal N}{B\log2},$$

where BPB means **bits per byte**. Division by $\log2$ converts nats to bits; division by $B$ expresses the result per raw byte. This is a representation-aware accounting convention. With multiple token sequences decoding to the same text, scoring one canonical encoding is not generally the same as summing the probability of all encodings of that text. State which quantity is measured.

**Worked example.** On 400 bytes, tokenizer A produces 100 tokens with mean loss 2.0 nats per token. Tokenizer B produces 80 with mean loss 2.3. Their summed losses are 200 and 184 nats. Their BPB scores are approximately 0.721 and 0.664. A has the lower token loss; B has the lower canonical-encoding loss on the fixed byte sample.

```python
from cs336_lab.tokenization import bits_per_byte

score_a = bits_per_byte(100 * 2.0, 400)
score_b = bits_per_byte(80 * 2.3, 400)
assert round(score_a, 3) == 0.721
assert round(score_b, 3) == 0.664
assert score_b < score_a
```

This check does not rank every property of the two models. Atlas still needs per-language task accuracy, context occupancy, identifier preservation, tokenizer throughput, and serving memory. A common loss unit removes one confounder; it does not create a complete product evaluation. A design is **Pareto-dominated** when another is no worse on every declared outcome and strictly better on at least one. When neither design dominates, choosing between them requires an explicit preference among the outcomes.

### Adaptive computation can move inside the model

The [Byte Latent Transformer paper](https://arxiv.org/abs/2412.09871) studies byte-level modeling with learned computation organized into patches rather than a fixed subword vocabulary. Its relevance is mechanistic: reducing tokenization dependence does not require processing every byte with the same expensive global computation. A system can group information adaptively elsewhere in the architecture.

For Atlas, this creates a research comparison rather than an automatic replacement. Fix the raw corpus, training-work budget, and evaluation protocol. Measure how an explicit tokenizer and an adaptive byte architecture allocate work across easy repetitive text and difficult rare strings. Also measure inference and preprocessing costs. A theoretical coverage advantage and a reported result in one study do not establish superiority on the technical library.

## 8. Limits, exercises, and fully worked solutions

The formal model separates representation, probability, and resource constraints. It does not predict which tokenizer minimizes downstream task error, how optimizer dynamics respond to segmentation, or which implementation runs fastest on a particular accelerator. The executable reference establishes deterministic merges and round-trip properties over its tested domain. It does not include production normalization, special tokens, regex pre-tokenization, streaming, or distributed training.

**Exercise 1 — derive an invariant.** Show that any sequence of valid BPE merges preserves the concatenated byte string. Explain why this does not guarantee that every sequence of token identifiers is valid UTF-8.

<details><summary>Worked solution</summary>

For one replacement, the old decoded substring is $b(i)\Vert b(j)$. The new token is defined to contain exactly that substring, so the complete document's decoded bytes are unchanged. The base case is the original byte sequence. Induction over the number of replacements establishes preservation after any finite merge sequence. An arbitrary token list need not arise from an accepted string's encoding: the singleton identifier 255 denotes byte `0xff`, which is not valid on its own in UTF-8. The implementation therefore guarantees text round-trip only along its accepted encoding path and raises a decoding error for invalid arbitrary bytes.

</details>

**Exercise 2 — compute a resource tradeoff.** Atlas doubles an untied vocabulary from 32,000 to 64,000 at width 768. Sequence length for a fixed corpus falls from 4096 to 3072. Compute added parameters and the fraction of dense attention pairs remaining. Is the change necessarily faster?

<details><summary>Worked solution</summary>

Each of the input and output tables gains $32{,}000\times768$ entries, giving $49{,}152{,}000$ additional parameters. The new sequence length is three quarters of the old length, so the attention-pair ratio is $(3/4)^2=9/16=0.5625$. This is a 43.75% reduction in the pair count. However, a full output projection scales roughly with the product of token count and vocabulary size at fixed width. Its relative work becomes $(3/4)\times2=1.5$, an increase of 50%. Total runtime depends on the original cost fractions, kernels, memory, and batching. The arithmetic comparison identifies competing effects and cannot settle the performance question alone.

</details>

**Exercise 3 — find a counterexample to a misleading metric.** Construct two models where lower token perplexity gives worse BPB on identical text. State which conventions must match.

<details><summary>Worked solution</summary>

Use the 400-byte example: A has 100 tokens at mean loss two and B has 80 at mean loss 2.3. Since the exponential function is increasing, A has smaller perplexity. Summed losses are 200 and 184 nats, so B has smaller BPB. The conclusion requires the same raw text, normalization, document boundaries, scored positions, and termination convention. If A excludes difficult tokens or B scores a normalized shorter corpus, the common denominator does not repair that mismatch. Report canonical-encoding likelihood explicitly rather than claiming a marginal probability over all tokenizations.

</details>

**Exercise 4 — diagnose an implementation.** A decoder converts every token's bytes to Unicode separately and then joins the resulting strings. Why can it pass English tests and fail multilingual tests? Give a principled repair.

<details><summary>Worked solution</summary>

ASCII bytes are individually valid UTF-8 encodings, so English-only cases may hide the mistake. A multibyte code point can be divided across token boundaries, leaving an individual token with an incomplete UTF-8 sequence. Decoding that token separately either raises an error or inserts a replacement character under a permissive policy. Concatenate token byte strings first, then decode the complete result once. For streaming, use an incremental UTF-8 decoder that preserves incomplete trailing bytes between chunks. Merely ignoring decoding errors violates exact text preservation.

</details>

**Exercise 5 — design an Atlas comparison.** Specify a small experiment that distinguishes lossless coverage, compression efficiency, and language-model quality without giving the larger tokenizer an undeclared compute advantage.

<details><summary>Worked solution</summary>

Train candidate tokenizers only on the training split and freeze their specifications. First test exact round-trip on unseen identifiers, empty text, accented forms, Traditional Chinese, and multibyte symbols. Then measure bytes per token and length tails separately by domain. For model quality, use identical held-out raw documents and a declared BPB convention, plus relevant task metrics. Match the resource constraint being claimed: equal tokens alone gives different raw-text exposure, while equal parameter counts can allocate different fractions to vocabulary tables. Record actual training FLOPs, wall time, peak memory, and bytes encountered. Finally compare serving cost at matched request text and output requirements. A Pareto comparison can reveal that neither candidate dominates; selecting one then requires an explicit application preference.

</details>

## Primary sources

- [Official executable Lecture 1](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_01.py): lecture sequence, resource framing, and tokenizer mechanisms.
- [Sennrich, Haddow, and Birch: Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): subword BPE and its original translation setting.
- [OpenAI: tiktoken](https://github.com/openai/tiktoken): a concrete tokenizer implementation and its interface conventions.
- [Eldan and Li: TinyStories](https://arxiv.org/abs/2305.07759): evidence from small models on a restricted synthetic distribution.
- [Pagnoni and colleagues: Byte Latent Transformer](https://arxiv.org/abs/2412.09871): a researched alternative placing adaptive grouping inside a byte-level architecture.

The [next chapter](lecture-02.html) turns Atlas's resource constraints into tensor-level operation and memory accounting.
