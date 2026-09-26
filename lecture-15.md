# Lecture 15 — Mid-training, supervised fine-tuning, and preference optimization

The lecture begins with a mismatch: a capable next-token predictor is not automatically a useful assistant. It then examines demonstrations, conversational data, style and factual behavior, mid-training, human and model preferences, reward models, and alternatives to PPO. This chapter derives the relationship between KL-regularized reward optimization and DPO while keeping the assumptions visible.

## 1. Demonstrations define which behavior is imitated [03:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=180s)

For prompt $x$ and demonstrated response $y=(y_1,\ldots,y_T)$, supervised fine-tuning minimizes

$$\mathcal L_{\rm SFT}(\theta)=-\mathbb E_{(x,y)\sim D}\sum_{t=1}^T
\log\pi_\theta(y_t\mid x,y_{<t}).$$

The policy $\pi_\theta$ is the model's conditional response distribution. Whether the loss sums or averages tokens changes weighting across response lengths. In chat training, user, system, and tool tokens can appear in context while only selected assistant tokens receive loss.

**Worked example.** One training example contains a 500-token prompt and 20-token answer. If all 520 tokens receive equal loss, most optimization effort imitates prompt text rather than the assistant response. Masking changes the objective to the behavior intended for supervision. Neither choice is universally wrong, but they train different distributions.

Data collection therefore matters as much as the loss function. Demonstrations must cover the desired task, interaction style, uncertainty behavior, and tool conventions. A perfectly implemented loss cannot recover a behavior absent from the selected examples.

## 2. Conversational quality and underlying capability can move separately [09:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=540s)

The recording contrasts older task-formatted instruction datasets with more natural conversation and synthetic instruction data. A model can become easier to talk to through relatively limited post-training while relying on knowledge and representations developed during pretraining.

This creates an evaluation trap. Increased verbosity, friendliness, or familiar formatting can improve preference scores without improving factual accuracy or reasoning. Conversely, a concise correct answer may be preferred less by a judge trained to reward elaborate explanations.

**Worked example.** Two models solve the same mathematics problems at the same rate. One adds fluent introductions and detailed-looking intermediate text. If judges prefer it, the improvement is real under that preference measure, but it is not evidence of increased mathematical success. Measure style and task correctness separately when their distinction matters.

Training on assertions the base model cannot reliably support may also encourage confident imitation. The lecture discusses uncertainty and abstention as behaviors that require appropriate signals. A model's internal uncertainty is not automatically expressed honestly in its output.

## 3. Mid-training changes the boundary between generic and task-specific data [34:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2040s)

“Mid-training” describes intermediate adaptation such as higher-quality domain data, long-context material, or instruction-like examples before final post-training. The term does not name a unique mathematical objective. Often the underlying next-token loss remains the same while the mixture and schedule change.

Let source weights vary with training progress: $P_t=\sum_iw_i(t)P_i$. Late-stage emphasis on high-quality data changes the distribution seen closest to deployment, potentially at a lower learning rate. That can be helpful, but the outcome depends on forgetting, data coverage, and the remaining optimization budget.

**Worked example.** A model trained mostly on broad prose is adapted on repository trajectories. It may learn tool-call syntax and long-range code relationships before SFT teaches the desired assistant protocol. If adaptation excludes general prose entirely for too long, broader abilities can regress. Evaluate both the new target and retained capabilities.

## 4. Preferences provide a different signal from demonstrations [42:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2520s)

Humans can sometimes recognize a better response more easily than write it from scratch. Preference data records a winner $y_w$ and loser $y_l$ for prompt $x$. A scalar reward model $r_\phi(x,y)$ can be trained with the Bradley–Terry model

$$P(y_w\succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).$$

The negative log of that probability is a binary classification loss. Only reward differences are identifiable from pairwise comparisons; adding a prompt-dependent constant changes no preferences.

Annotator instructions, expertise, demographics, and task distribution affect the labels. A preference model learns those observed judgments, not an objective universal utility function. Model-generated preferences add another layer: the judge's failure modes become training signals.

**Worked example.** If a reward model assigns scores 3 and 1, the predicted win probability is $\sigma(2)\approx0.881$. Scores 103 and 101 give the same probability. The absolute reward scale and offset require careful interpretation when the model is later optimized.

## 5. KL regularization anchors reward optimization [64:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3840s)

Let $\pi_{\rm ref}$ be a reference policy. For one prompt, a common objective is

$$J(\pi)=\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\rm ref}(y)},\qquad\beta>0.$$

The second term is $\beta D_{\rm KL}(\pi\|\pi_{\rm ref})$. It penalizes departure from the reference and can limit exploitation of reward-model errors. Assume the relevant reference probabilities are positive and the normalization below is finite.

Introduce a multiplier for $\sum_y\pi(y)=1$. Differentiating with respect to $\pi(y)$ yields $r(y)-\beta[\log(\pi(y)/\pi_{\rm ref}(y))+1]+\lambda=0$. Rearranging and normalizing gives

$$\pi^*(y)=\frac{\pi_{\rm ref}(y)e^{r(y)/\beta}}{Z},\qquad
Z=\sum_y\pi_{\rm ref}(y)e^{r(y)/\beta}.$$

Thus reward exponentially tilts the reference distribution. Larger $\beta$ keeps the solution closer to the reference; smaller $\beta$ favors high rewards more aggressively.

**Worked example.** For two equally likely reference responses with rewards 1 and 0 and $\beta=1$, the optimal probabilities are $e/(e+1)\approx0.731$ and 0.269. At $\beta=0.1$, the first receives about 0.99995. Reward scale and KL strength must be interpreted together.

## 6. DPO substitutes a policy ratio for a reward difference [70:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4200s)

From the optimal-policy relation,

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x).$$

For winner and loser under the same prompt, $\log Z(x)$ cancels. Substituting this difference into the preference likelihood motivates the DPO loss

$$\mathcal L_{\rm DPO}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\right]\right).$$

Sequence log probability is the sum of token log probabilities under the response prefix. DPO directly fits preferences through policy likelihood ratios, avoiding an explicit learned reward model and rollout-based policy optimization in its basic offline training loop.

The derivation does not say every offline dataset recovers the true reward optimum. It relies on the preference model, coverage, optimization, and policy expressiveness. Online refresh, iterative data collection, and length-normalized variants change the procedure or objective and should be named separately.

**Worked example.** Suppose a preferred response's probability ratio to reference is 2 and the rejected response's ratio is $1/2$. Their log-ratio margin is $\log4$. With $\beta=1$, the implied preference probability is 0.8 and the pair loss is $-\log0.8\approx0.223$. This measures relative change from reference, not simply which response currently has larger absolute probability.

## 7. Researched extension — stronger optimization amplifies proxy errors

The [InstructGPT](https://arxiv.org/abs/2203.02155) and [helpful/harmless RLHF](https://arxiv.org/abs/2204.05862) papers establish concrete demonstration–preference–optimization pipelines. The [DPO paper](https://arxiv.org/abs/2305.18290) provides the direct preference formulation. A common issue across them is that the reward or preference dataset is a proxy for deployment quality.

Suppose learned reward is $\widehat r(y)=r_{\rm true}(y)+\epsilon(y)$. Optimization selects responses partly for positive error $\epsilon$, not only true quality. As search or policy optimization becomes stronger, it can increasingly exploit regions where the proxy is wrong. KL control, fresh human evaluation, adversarial tests, and data refresh can mitigate this, but none makes an imperfect reward infallible.

This is a statistical selection effect, not a claim that optimization is inherently harmful. The engineering question is how far the proxy remains reliable as the policy moves beyond the data that trained it.

## 8. Exercises and worked solutions

**Exercise 1.** Why does adding a prompt-dependent constant to reward leave the KL-regularized optimal policy unchanged?

<details><summary>Solution</summary>

The numerator and normalization both gain the same factor $e^{c(x)/\beta}$, which cancels. The expected reward objective also shifts by a constant independent of the policy for that prompt.

</details>

**Exercise 2.** At the reference policy, what is the DPO loss for any pair in the basic formulation?

<details><summary>Solution</summary>

Both log ratios are zero, so the margin is zero and the loss is $-\log(1/2)=\log2$. The gradient depends on the winner/loser likelihood derivatives, so the loss value alone does not imply no learning signal.

</details>

**Exercise 3.** How would you determine whether preference gains reflect style or capability?

<details><summary>Solution guidance</summary>

Evaluate objective task outcomes separately, use length- and format-controlled comparisons where appropriate, inspect factual support, and test with judges whose rubric explicitly distinguishes presentation from correctness. Report both results rather than assuming one explains the other.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=2oH6PWPrYFo) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_15.pdf).
- [InstructGPT](https://arxiv.org/abs/2203.02155), [Helpful and Harmless RLHF](https://arxiv.org/abs/2204.05862), and [DPO](https://arxiv.org/abs/2305.18290).
