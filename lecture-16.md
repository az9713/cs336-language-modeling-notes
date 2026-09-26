# Lecture 16 — Reinforcement learning with verifiable rewards

The recording develops the algorithms and practical recipe behind RLVR: PPO, group-relative advantages, normalization effects, DeepSeek-R1 and later reasoning pipelines, task difficulty, verifiers, and combined training/inference infrastructure. The central change from preference optimization is often the reward's source and reliability, rather than a completely new mathematical learning principle.

## 1. A verifier changes the feedback channel [00:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=0s)

Let $x$ be a prompt, $y\sim\pi_\theta(\cdot\mid x)$ a sampled response, and $R(x,y)$ a reward. The objective is

$$J(\theta)=\mathbb E_x\mathbb E_{y\sim\pi_\theta}[R(x,y)].$$

In RLVR, reward can come from a final-answer checker, program tests, or an environment outcome. “Verifiable” identifies a mechanism; it does not prove the mechanism captures the entire intended task. A unit-test suite can miss bugs, a parser can reject equivalent answers, and an agent can exploit an environment defect.

**Worked example.** A mathematics checker compares the string `0.5` with expected `1/2`. Exact string matching rejects a correct value. A permissive parser that accepts any response containing `1/2` may accept contradictory answers. A robust verifier needs an explicit accepted-answer language and tests for both false positives and false negatives.

## 2. Mathematical primer — derive the sequence policy gradient

For a fixed prompt and a reward independent of parameters except through sampled output,

$$\nabla J=\sum_yR(y)\nabla\pi_\theta(y)
=\mathbb E_y[R(y)\nabla\log\pi_\theta(y)].$$

Autoregressive factorization gives

$$\nabla\log\pi_\theta(y\mid x)=\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t}).$$

Thus a terminal reward can reinforce all sampled token decisions. This is an unbiased score-function estimator under the stated sampling and regularity assumptions, but variance can be high and credit assignment crude.

A baseline $b(x)$ independent of the sampled response can be subtracted without changing the expectation because

$$\mathbb E_y[b(x)\nabla\log\pi_\theta(y\mid x)]
=b(x)\nabla\sum_y\pi_\theta(y\mid x)=0.$$

The difference $R-b$ is an advantage-like signal: better or worse than a reference expectation. Its sign determines whether a sampled behavior is encouraged or discouraged in the simplest update.

## 3. PPO constrains reuse of samples from an older policy [06:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=360s)

Let the old rollout policy be $\pi_{\rm old}$ and define a token-level probability ratio $r_t(\theta)=\pi_\theta(y_t\mid h_t)/\pi_{\rm old}(y_t\mid h_t)$. A common PPO surrogate is

$$\mathbb E\min\left(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right).$$

Here $h_t$ is the token history and $A_t$ an estimated advantage. Clipping limits incentives for some large changes on the sampled actions. It is not an exact hard bound on the full policy's KL divergence.

The value model in a conventional actor–critic setup estimates expected future reward and supports advantage estimation. For a language model it can be costly in memory and computation. The lecture emphasizes that implementation details—masking, normalization, value targets, KL terms, and batching—make practical PPO more complicated than its outer loop suggests.

**Worked example.** With positive advantage one and $\epsilon=0.2$, increasing the sampled action ratio from 1.2 to 1.5 does not improve the clipped surrogate beyond 1.2. With a negative advantage, the direction of the minimum matters; the expression must be implemented as written rather than clipping the gradient by intuition.

## 4. Group-relative methods replace a learned critic with comparisons [13:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=780s)

Sample $G$ responses to the same prompt, with rewards $R_1,\ldots,R_G$. A common group-relative advantage is

$$A_i=\frac{R_i-\overline R}{s_R+\varepsilon},\qquad
\overline R=G^{-1}\sum_iR_i.$$

Here $s_R$ is a chosen group standard deviation and $\varepsilon$ protects numerical division. Implementations differ in variance convention and normalization. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) introduces GRPO as a way to avoid a separate value model in this setting.

**Worked example.** For rewards $(1,0,0,0)$, the mean is 0.25 and population standard deviation is $\sqrt{0.1875}\approx0.433$. The normalized advantages are approximately $(1.732,-0.577,-0.577,-0.577)$. If every response has the same reward, centered advantages vanish, so that group supplies no relative reward signal.

There is a subtle statistical point. The group mean includes the current sample's reward, so it is not independent of that response. Without standard-deviation normalization, the expected estimator using the inclusive mean is scaled by $(G-1)/G$ relative to the ordinary policy gradient for independent samples. A leave-one-out baseline avoids that particular dependence. Standard-deviation normalization introduces further reward-dependent weighting. These methods are useful surrogates, not automatically the same unbiased estimator as an action-independent baseline.

## 5. Length normalization and difficulty normalization change incentives [20:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1200s)

A sequence policy gradient naturally contains a sum over token log probabilities. Dividing a response's contribution by its own length changes how sequences are weighted. It can create length-dependent effects, especially for negative-advantage responses. Similarly, dividing by a prompt's reward standard deviation changes how prompts of different success probabilities contribute.

For binary reward with success probability $p$, variance is $p(1-p)$. It is small near zero and one, so normalization can strongly reweight such groups, subject to finite-sample effects and zero-variance handling.

**Worked example.** Two failed responses have the same negative advantage, but one is ten tokens and the other one hundred. Under a mean-token loss, each token's coefficient in the longer response is ten times smaller. This does not by itself prove the model will always choose longer failures, because token gradients differ, but it exposes an incentive that an unnormalized sequence objective does not share.

The [Dr. GRPO analysis](https://arxiv.org/abs/2503.20783) examines such biases. Read objective formulas and denominator conventions when reproducing a result; algorithm names alone are insufficient.

## 6. Successful RL needs tasks with learnable reward signal [27:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1620s)

The [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948) distinguishes an RL-centered experimental route from a fuller pipeline involving cold-start data and later stages. The recording treats these as recipes built on a capable pretrained model, not learning reasoning from an empty initialization.

For a task with independent per-sample success probability $p$, the probability of at least one success among $G$ samples is

$$1-(1-p)^G.$$

**Worked example.** At $p=0.01$ and $G=8$, this is about 7.7%. Most groups have no success. At $p=0.2$, it is about 83.2%. A curriculum or data filter can increase the frequency of useful contrasts, but excluding all hard tasks can narrow the learned capability.

Longer generated reasoning is an observable behavior, not proof of a particular internal mechanism. Length can increase through improved search, objective bias, or both. Evaluate correctness against computation spent and inspect successful and failed trajectories separately.

## 7. Verifier robustness determines how far optimization can go [49:40](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2980s)

The recording stresses the difficulty of making the “verifiable” part reliable. A checker should recognize valid equivalent outputs, reject invalid shortcuts, and remain robust as the policy changes. More optimization can discover weaknesses absent from the initial training samples.

For code tasks, isolate the environment, protect the scoring mechanism, and test actual behavior. For mathematical answers, separate parsing from semantic correctness. For agents, distinguish producing an artifact from altering the evaluator or merely claiming completion. These are evaluation-design requirements, not optional postprocessing.

**Worked example.** If reward is based on a file containing `PASS`, an agent can write that file directly. If reward comes from independently executed tests whose results the agent cannot forge, that shortcut disappears, though incomplete tests can still be exploited. The reward boundary defines what behavior optimization can select.

## 8. Researched extension — rollout infrastructure is part of the algorithm [51:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3060s)

RL alternates generation and parameter updates. A rollout service wants large efficient decode batches; training wants suitable batches and fresh-enough policy data. Variable response lengths create stragglers. Sharing devices between training and inference can improve utilization while complicating scheduling and state transfer.

Let rollout collection take $T_r$, scoring $T_s$, and updating $T_u$. A synchronous iteration takes roughly their sum plus transfer overhead. An asynchronous system can overlap stages, but then samples may come from older policies. Policy-version tracking and probability ratios become algorithmic correctness information.

An independently useful diagnostic reports reward by policy version, task family, response length, and verifier outcome category, alongside tokens/s and update time. A rising average reward with collapsing task coverage or growing checker anomalies is not convincing evidence of broader improvement.

## 9. Exercises and worked solutions

**Exercise 1.** Why does a constant reward produce zero expected ordinary policy gradient?

<details><summary>Solution</summary>

For constant $c$, $\mathbb E[c\nabla\log\pi(y)]=c\nabla\sum_y\pi(y)=0$. Finite samples can still have noise. A centered group with identical observed rewards has exactly zero reward-advantage contribution in that batch.

</details>

**Exercise 2.** A task succeeds with probability 0.1. How many independent samples are needed for at least a 95% chance of one success?

<details><summary>Solution</summary>

Solve $1-0.9^G\ge0.95$, giving $G\ge\log(0.05)/\log(0.9)\approx28.43$. Thus 29 samples suffice under independence. Correlated samples can reduce the practical benefit.

</details>

**Exercise 3.** What would distinguish verifier exploitation from a real capability gain?

<details><summary>Solution guidance</summary>

Use independent held-out verification, manually inspect high-reward failures, perturb formatting, test equivalent tasks with different checkers, and evaluate unseen task families. Reward increase alone is evidence of optimizing the implemented reward, not necessarily the intended task.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=dIFAi87Ws4E) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_16.pdf).
- [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and [Dr. GRPO analysis](https://arxiv.org/abs/2503.20783).
