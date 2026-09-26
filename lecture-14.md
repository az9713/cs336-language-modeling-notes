# Lecture 14 — Filtering, deduplication, mixtures, and synthetic data

The second data lecture follows the transformations between raw material and a training distribution. It begins with extraction and quality filtering, derives MinHash locality-sensitive hashing, examines mixture weights and repeated exposure, and ends with synthetic reasoning and software-engineering data. Each operation changes what the model sees; none is merely administrative cleanup.

## 1. Extraction errors become training examples [00:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=0s)

HTML and PDF extraction must recover meaningful reading order and structure. Boilerplate, tables, equations, and code can require different treatment. A pipeline that handles ordinary prose well may scramble a scientific derivation or remove indentation that carries program semantics.

Define an extraction transformation $E(d)$ from source document $d$ to text. Subsequent filters only see $E(d)$, so an extraction error can masquerade as low source quality. Keep raw-source identifiers and sample outputs after each stage to distinguish these causes.

**Worked example.** A PDF page contains two columns, each with a complete paragraph. Reading horizontally across both columns interleaves sentences. A language-quality filter may reject the result, causing the pipeline to discard a useful paper because of an extractor defect. Inspecting only rejected text would conceal the original cause.

## 2. Quality filtering defines a target distribution [08:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=480s)

Let $R$ be a raw-data distribution and $T$ a target or reference distribution. A classifier can distinguish samples from $T$ and $R$, while a generative model can score how likely a document is under $T$. Either approach encodes a notion of desirable data through the reference set.

With class prior $\pi=P(y=1)$ and class-conditional densities $p_T(x)$ and $p_R(x)$, Bayes' rule gives

$$\frac{P(y=1\mid x)}{1-P(y=1\mid x)}
=\frac{\pi}{1-\pi}\frac{p_T(x)}{p_R(x)}.$$

A calibrated classifier's odds can therefore estimate a density ratio up to the prior factor. In practice, model misspecification and reference bias matter. Selecting “encyclopedia-like” documents is not identical to selecting all documents that improve a model.

**Worked example.** A classifier trained with equal priors assigns probability 0.8 to a document. Its estimated density ratio is 4. If the positive prior was 0.2 instead, the same posterior corresponds to ratio 16. Ignoring training priors changes the interpretation of scores.

Hard thresholds trade coverage for apparent quality. A small training budget may benefit from an aggressive filter, while a much larger budget may exhaust the retained pool and repeat it excessively. Threshold quality is therefore conditional on the intended run length.

## 3. Duplicates alter both efficiency and measurement [22:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1320s)

Exact duplicates can be found by hashing normalized content. Near duplicates require a similarity definition. Deduplication can operate at document, paragraph, sentence, or substring level; these choices remove different kinds of repetition.

Repeated boilerplate wastes token budget and changes sampling weights. Train–test overlap can inflate evaluation. But not all repeated text is undesirable: a common programming idiom, a mathematical definition, or a license has a reason to recur. The policy should distinguish the purpose of deduplication from the mechanism used to find similarity.

The primary [deduplication study](https://arxiv.org/abs/2107.06499) evaluates memorization and training effects of removing repeated text. Its results motivate testing deduplication, not blindly removing every repeated phrase.

## 4. MinHash estimates Jaccard similarity [31:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1860s)

Represent a document by a set of shingles, such as contiguous token $n$-grams. For sets $A$ and $B$, Jaccard similarity is

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

Choose a uniformly random ordering of the universe and let $h(A)$ be the minimum-ranked element of $A$. The minimum element of $A\cup B$ is equally likely to be any member of that union. The minima of $A$ and $B$ match exactly when that element belongs to their intersection. Therefore

$$P(h(A)=h(B))=J(A,B).$$

This is the ideal random-permutation statement. Implementations approximate it with suitable hash families, finite signatures, and collision handling.

**Worked example.** Let $A=\{a,b,c\}$ and $B=\{b,c,d\}$. The union has four elements and intersection two, so Jaccard similarity is $1/2$. If the first element in the random ordering is $b$ or $c$, minima match; if it is $a$ or $d$, they do not.

With $n$ independent hash functions, the fraction of matching signature entries estimates $J$. Its idealized variance is $J(1-J)/n$. At $J=0.5$ and $n=100$, standard deviation is 0.05. A short signature is a noisy estimator, not a perfect equality test.

## 5. Banding turns signatures into candidate retrieval [38:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2280s)

Comparing every document pair is quadratic in corpus size. Locality-sensitive hashing splits a signature into $b$ bands of $r$ rows each. Documents become candidate matches if every row matches in at least one band.

If signature comparisons behave independently and similarity is $s$, one band matches with probability $s^r$. No band matches with probability $(1-s^r)^b$. Thus

$$P(\text{candidate}\mid s)=1-(1-s^r)^b.$$

Increasing $r$ makes a band harder to match; increasing $b$ creates more chances. The familiar threshold shape is probabilistic, not a hard cutoff.

**Worked example.** With $b=20$ and $r=5$, a pair at similarity 0.8 becomes a candidate with probability $1-(1-0.8^5)^{20}\approx0.9996$. At similarity 0.2, the probability is about 0.0064. Candidates can then receive a more precise similarity check before a removal decision.

Cluster policy matters after pair detection. If A resembles B and B resembles C, A need not resemble C above the same threshold. Removing all but one document per connected component can erase a chain of gradually changing documents. Preserve representative-selection rules and inspect large clusters.

## 6. Mixture weights determine effective epochs [50:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3000s)

Let source $i$ contain $U_i$ unique eligible tokens, receive fraction $w_i$ of a run, and let total token presentations be $D$. Its average exposure is

$$e_i=\frac{w_iD}{U_i}.$$

This is an expected number of passes under uniform sampling. A small high-quality source can be repeated many times even when its mixture weight looks modest.

**Worked example.** A high-quality pool has 10 billion tokens and a broad pool has 990 billion. A one-trillion-token run with equal mixture weights makes 50 passes over the first pool and about 0.505 passes over the second. “Half high-quality data” conceals substantial repetition.

Mixture optimization can train small proxy models, fit a response surface, and choose weights that improve selected evaluations. This is valuable but can overfit the evaluation suite. If only code tests define quality, the optimizer will favor code. The objective must represent the intended model, and the final choice should be tested on held-out tasks.

The [DoReMi paper](https://arxiv.org/abs/2305.10429) is a primary example of learning domain weights with a proxy. Transfer to larger models and longer runs is an empirical assumption to test, especially when unique-data limits differ.

## 7. Synthetic data requires generation, selection, and a learning target [73:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4380s)

The recording examines synthetic reasoning examples and software-engineering tasks. A teacher can generate candidate solutions, a verifier or judge can filter them, and a student can learn from the retained examples. Quality depends on the prompt distribution, teacher capability, verification, diversity, and the student's starting point.

For repository tasks, generating a plausible issue is not enough. The environment must install, the task must have a meaningful target, and tests should distinguish a real solution from shortcuts. Synthetic edits that are always trivial teach a different skill from repairing naturally occurring bugs.

**Worked example.** A generator mutates an arithmetic operator in a function and asks the model to restore passing tests. This creates verifiable tasks cheaply, but a dataset consisting only of such mutations can teach local pattern repair without teaching requirements interpretation or multi-file design. Evaluate on independently sourced tasks to measure transfer.

Teacher output is not automatically ground truth. Filtering for final-answer correctness can still retain flawed reasoning, and filtering with a model judge can preserve the judge's biases. Keep the verification method and its limitations in the dataset record.

## 8. Researched extension — filtering and weighting are two versions of selection

A hard filter uses acceptance function $a(x)\in\{0,1\}$; a soft sampler uses $a(x)\ge0$. In either case, the resulting distribution is

$$P_a(x)=\frac{a(x)P_R(x)}{\mathbb E_{P_R}[a(X)]}.$$

This independent formulation connects quality filtering, source mixing, and deduplication. A removed duplicate receives zero weight; a retained rare domain can receive extra weight. Thinking in distributions exposes what every rule changes and suggests diagnostics: domain shares, language shares, document-length distributions, and effective sample counts before and after processing.

Hard filtering also destroys optionality. Keeping compact provenance and scores permits later resampling without repeating expensive extraction or annotation. That can be valuable when the target model or token budget changes.

## 9. Exercises and worked solutions

**Exercise 1.** For $b=10$, $r=2$, and similarity $s=0.5$, compute candidate probability.

<details><summary>Solution</summary>

$1-(1-0.25)^{10}=1-0.75^{10}\approx0.9437$. This is high despite only moderate similarity. Banding parameters should be chosen for the desired recall and candidate volume.

</details>

**Exercise 2.** A 2-billion-token source receives 5% of a 400-billion-token run. What is its expected exposure?

<details><summary>Solution</summary>

$0.05\cdot400/2=10$ passes. If documents are sampled nonuniformly, some may be repeated much more often than this mean.

</details>

**Exercise 3.** Why should a deduplication pipeline inspect both false positives and false negatives?

<details><summary>Solution</summary>

False negatives leave unwanted repetition or contamination. False positives discard distinct useful content, perhaps disproportionately in formulaic genres or smaller languages. The acceptable balance depends on whether the goal is efficiency, memorization reduction, or evaluation integrity.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=5sxHosTLPF8) and [official executable lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_14.py): MinHash derivation and implementation, filtering, mixtures, and synthetic-data cases.
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499).
- [FineWeb](https://arxiv.org/abs/2406.17557): filtering and deduplication ablations.
- [DoReMi](https://arxiv.org/abs/2305.10429): learning data-mixture weights.
