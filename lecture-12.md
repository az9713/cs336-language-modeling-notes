# Lecture 12 — Evaluation measures a model under a protocol

The lecture asks what makes a model good after its architecture, training, scaling, and inference machinery are in place. It moves from perplexity to exams, human preference, model judges, agents, reasoning, safety, economic tasks, and contamination. A score is a property of a model–task–protocol combination, not an intrinsic scalar measure of intelligence.

## 1. Start with the intended use and observable outcome [00:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=0s)

Let $x\sim D$ be a task from the evaluation distribution, let $A$ be the complete system including model, prompt, tools, and decoding policy, and let $S(A,x)$ be a score. The target quantity is

$$\mu=\mathbb E_{x\sim D}[S(A,x)].$$

An observed benchmark estimates this expectation on a finite sample. Changing the prompt, tool budget, scoring rule, or task distribution changes the estimand. A comparison that changes several components at once may be valid as a system comparison but does not isolate model quality.

**Worked example.** A coding model with ten attempts and execution feedback solves 70% of tasks; another with one attempt and no execution solves 60%. Those are two system results. They do not establish that the first base model is stronger under equal resources. Report the budget and interface with the score.

The [HELM paper](https://arxiv.org/abs/2211.09110) formalizes a broad scenario-and-metric approach. Its important methodological idea is to measure multiple dimensions rather than collapse accuracy, robustness, calibration, and efficiency into one unexplained rank.

## 2. Perplexity is precise but distributes importance over all tokens [05:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=300s)

For held-out tokens $x_{1:T}$, mean negative log likelihood is $L=-T^{-1}\sum_t\log p(x_t\mid x_{<t})$, and perplexity is $e^L$. It gives dense feedback and can be measured with low variance on large corpora. But it weights the selected tokens according to the loss definition, not according to their practical importance to a user.

Predicting a common function word and predicting the crucial date in a factual answer both contribute to the sum. Small improvements across abundant easy tokens can dominate rare consequential errors. Perplexity also depends on tokenization, so comparisons across tokenizers need a common normalization such as bits per byte.

**Worked example.** A model improves 990 routine tokens by 0.01 nats each but worsens ten critical tokens by 0.5 nats each. Total loss improves by $9.9-5=4.9$ nats, even though performance on the selected critical subset becomes worse. Whether that is desirable depends on the task.

Multiple-choice likelihood scoring introduces further choices: sum or mean log probability of answer strings, prompt formatting, answer ordering, and whether to score full text or labels. The protocol must be fixed before comparing models.

## 3. Exams trade control for ecological coverage [18:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1080s)

Exam-style benchmarks provide questions, answer keys, and a controllable subject/difficulty distribution. They are convenient for reproducible measurement but cover a limited interaction pattern. A model may excel at difficult multiple-choice science while struggling to clarify an ambiguous real request or complete a workflow.

For binary correctness indicators $X_i$, estimated accuracy is $\widehat p=n^{-1}\sum_iX_i$. Under independent identically distributed Bernoulli trials, its approximate standard error is

$$\operatorname{SE}(\widehat p)=\sqrt{\widehat p(1-\widehat p)/n}.$$

**Worked example.** At 80% accuracy on 100 independent questions, standard error is 0.04, giving a rough normal 95% interval of about $0.80\pm0.078$. The interval is wide. Near boundaries or with small samples, use a more appropriate interval such as Wilson's; correlated or clustered questions reduce the effective sample size.

Harder exams do not automatically solve evaluation. Expert-written questions can be ambiguous, derived from public sources, or uneven across fields. Benchmark saturation and contamination motivate new tasks, but a new name alone does not establish validity.

## 4. Preference evaluation measures a population's choices [31:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1860s)

Pairwise preference asks a judge to choose between responses. A common model assigns each system a latent score $s_i$ and predicts

$$P(i\succ j)=\sigma(s_i-s_j),$$

where $\sigma$ is the logistic function. Only score differences matter; adding a common constant leaves probabilities unchanged. Rankings depend on which prompts and judges generated the comparisons.

**Worked example.** A score difference of $\log3$ implies a 75% predicted win probability in the simple model. It does not mean the winner is “three times as intelligent.” It is an odds relationship under the fitted preference model.

User populations can prefer different styles, verbosity, or risk tolerance. Self-selected users are not a random sample of all potential users. Prompt distributions can favor some capabilities. Report the population and collection process, and avoid interpreting a preference leaderboard as a universal ordering.

## 5. Model judges need validation and structured criteria [37:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2220s)

An LLM judge is itself a measurement instrument. It may favor longer responses, particular formatting, familiar model styles, or the first answer shown. Prompt-specific checklists can make criteria more explicit than a single “which is better?” question, but the checklist can omit an essential requirement or encode a mistaken standard.

A useful validation set contains expert-reviewed examples, including factual errors hidden in polished prose, concise correct answers, conflicting constraints, and adversarially persuasive answers. Swap response order and compare judgments. Measure agreement with the intended human rubric rather than assuming the judge's confidence establishes correctness.

**Worked example.** Suppose a judge chooses the longer response 80% of the time on pairs where humans find no quality difference. A model that learns verbosity can gain judge score without improving usefulness. Length control or regression adjustment can reveal the confound, but the appropriate remedy depends on whether extra length is sometimes legitimately useful.

## 6. Agents require evaluating actions and environment state [45:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2700s)

An agent combines a model with tools, memory, an execution loop, and an environment. Its final prose can claim success without producing the requested state. Evaluation should inspect the resulting artifact, test execution, or environment state where possible.

Let a trajectory be $\tau=(o_0,a_0,o_1,a_1,\ldots,o_T)$ and let $V(\tau)$ be a verifier. Success depends on both the policy and environment dynamics. Tool versions, timeouts, network access, hidden tests, and resource budgets can change the result.

**Worked example.** For a repository repair task, a patch that passes existing tests may still break an untested contract. A stronger evaluation combines hidden behavioral tests, regression tests, and inspection for shortcuts such as deleting the failing test. The score should distinguish actual repair from gaming the measurement.

Trajectory inspection complements aggregate metrics. A hundred failures sharing one tool-schema bug suggest a different intervention from a hundred unrelated reasoning errors. The [SWE-bench paper](https://arxiv.org/abs/2310.06770) provides a primary example of repository-based evaluation; the recording also surveys terminal and machine-learning engineering environments.

## 7. Reasoning, safety, and economic value need different criteria [54:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=3240s)

Abstract puzzles aim to reduce dependence on factual knowledge, but the representation, instructions, and prior training still matter. A benchmark should state what kind of generalization it tests rather than claiming to isolate all reasoning ability.

Safety evaluation is contextual: a response's acceptability can depend on intent, information content, and the surrounding workflow. A refusal rate alone cannot distinguish appropriate refusal from excessive refusal. Evaluate helpfulness on benign requests as well as failure on harmful or disallowed ones, under clearly defined criteria.

Economic-task evaluation asks whether outputs satisfy professional requirements. Artifact quality, revision burden, time saved, and integration into a workflow may matter more than exam accuracy. The recording's examples illustrate this change of unit: evaluate completed work rather than only answers to questions. Numerical leaderboard claims in the recording are historical and are not presented here as current rankings.

## 8. Contamination has more routes than exact string overlap [69:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=4140s)

Exact duplication of test questions in training is one contamination route. Others include paraphrases, solutions in discussion forums, source documents from which questions were derived, and repeated development against the benchmark. Temporal splits help but do not guarantee novelty; a newly posted task can reuse an old problem.

Private tasks, procedurally generated variants, freshness checks, and source-level deduplication each address part of the problem. They introduce tradeoffs in cost, reproducibility, and representativeness. A transparent report records known overlap checks and their limits.

## 9. Researched extension — paired comparisons use information that separate error bars discard

When two systems answer the same questions, analyze per-question differences. Let $D_i=X_i^{(A)}-X_i^{(B)}$. Then $\widehat\Delta=n^{-1}\sum_iD_i$ estimates the accuracy difference, and its standard error can be estimated from the sample variance of $D_i$ divided by $n$.

This uses the correlation induced by shared tasks. If both systems fail the same difficult questions, separate independent-sample calculations waste information. For stochastic agents, repeat runs and distinguish variation across tasks from variation across attempts. Bootstrap at the task level when multiple attempts belong to one task; treating every attempt as an independent new task understates uncertainty.

## 10. Exercises and worked solutions

**Exercise 1.** A system solves 81 of 100 questions, another 79. Is the two-point difference compelling by itself?

<details><summary>Solution</summary>

No. The sample is small enough that uncertainty can exceed two points. Use paired outcomes to examine disagreements, appropriate intervals or tests, and repeat evaluation if decoding is stochastic. Statistical evidence also does not establish practical importance.

</details>

**Exercise 2.** Design an evaluation for a literature-review agent.

<details><summary>Solution guidance</summary>

Define a task distribution and time/source-access budget. Check citation existence, claim–source support, coverage of relevant evidence, treatment of disagreement, and usefulness of the synthesis. Include hidden source omissions and misleading abstracts. Inspect the finished report and retrieval trace, and record human revision time.

</details>

**Exercise 3.** Why can an aggregate score improve while a deployment becomes worse?

<details><summary>Solution</summary>

The benchmark weighting may differ from deployment. Improvements on common easy cases can outweigh regressions on rare costly cases. Averages can also conceal increased latency, reduced reliability, or failures concentrated in a subgroup. Report slices chosen for the intended use.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=JpAxdTWQJxM) and [official lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_12.py).
- [HELM](https://arxiv.org/abs/2211.09110): scenario and multi-metric evaluation.
- [SWE-bench](https://arxiv.org/abs/2310.06770): evaluating repository issue resolution.
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300): exam-style evaluation.
