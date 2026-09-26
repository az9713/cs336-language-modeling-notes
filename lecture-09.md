# Lecture 9 — Scaling laws and compute-optimal training

Scaling laws are empirical forecasting tools. They turn many affordable experiments into a prediction about a more expensive run. The lecture develops power-law behavior, its statistical interpretation, architecture and batch-size studies, and the allocation of compute between model size and training data. Its recurring warning is that a clean curve can conceal fragile accounting and experimental assumptions.

## 1. A power law is a model of a regime [03:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=180s)

Let $x$ denote a resource such as training tokens and $L(x)$ a loss. A common model is

$$L(x)=L_\infty+Ax^{-\alpha},\qquad A>0,\ \alpha>0.$$

The asymptote $L_\infty$ is the fitted irreducible or limiting loss under the model. Subtracting it and taking logs yields $\log(L-L_\infty)=\log A-\alpha\log x$. Thus a power law appears linear on log–log axes only after handling the offset correctly.

**Worked example.** If $\alpha=0.1$, doubling resources multiplies excess loss by $2^{-0.1}\approx0.933$: about a 6.7% reduction. A tenfold increase multiplies it by about 0.794. Predictable improvement can therefore coexist with sharply increasing cost per unit of progress.

A straight segment on a narrow log plot does not establish a universal law. Several smooth functions look nearly linear over a short interval. The resource range, number of runs, seed variability, and residuals all matter. Extrapolation should extend a tested regime, not assume every architectural or data transition preserves it.

## 2. Why statistical rates offer intuition, not a complete explanation [15:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=900s)

In elementary estimation, averaging independent observations reduces variance as $1/n$, while standard error scales as $n^{-1/2}$. Nonparametric estimation can have slower rates depending on smoothness and effective dimension. Such examples make power laws plausible without deriving a language model's observed exponent from first principles.

For a simple bias–variance illustration, let approximation error decrease with model complexity $k$ as $Ak^{-a}$ and estimation error increase as $Bk/n$. Balancing the two gives $k\propto n^{1/(a+1)}$ and total error proportional to $n^{-a/(a+1)}$. A slow exponent can arise from a difficult approximation problem, but this toy model is not a validated theory of transformer scaling.

The distinction is useful: a statistical analogy suggests hypotheses; an empirical fit estimates a relationship; a mechanistic theory explains why it should hold. Do not treat the first as the third.

## 3. Intercepts, exponents, and architectural crossings [22:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1320s)

An intervention can lower loss at every tested scale without changing the exponent. If two methods share $L_\infty$ and $\alpha$, changing $A$ changes the intercept in log excess loss. This can be expressed as an effective resource multiplier.

If method B has coefficient $A_B<A_A$, matching the same loss requires resources satisfying $A_Ax_A^{-\alpha}=A_Bx_B^{-\alpha}$. Therefore

$$\frac{x_A}{x_B}=\left(\frac{A_A}{A_B}\right)^{1/\alpha}.$$

**Worked example.** At $\alpha=0.1$, reducing the coefficient by 10% gives an effective multiplier $(1/0.9)^{10}\approx2.87$. A modest-looking vertical improvement can correspond to a large compute saving when the exponent is shallow.

Different exponents can make rankings cross. A method that wins at small scale may lose later, or vice versa. Compare methods over multiple scales with matched accounting. In MoE models, total and active parameters are different variables; fitting a dense-model law to total MoE size without qualification confuses storage capacity with executed work.

## 4. Critical batch size links optimization to parallelism [40:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2400s)

A larger batch reduces gradient noise but usually gives diminishing returns in the number of optimizer steps needed to reach a target. Let $S(B)$ be steps needed at batch $B$, so examples consumed are $E(B)=BS(B)$. A simple saturation model is

$$S(B)\approx S_\infty+\frac{E_0}{B}.$$

At small $B$, examples consumed remain near $E_0$, and increasing batch reduces steps almost proportionally. At large $B$, steps approach $S_\infty$ while examples consumed grow. The crossover is $B_{\rm crit}\approx E_0/S_\infty$.

This is an explanatory model of the tradeoff, not an exact optimizer identity. A noise-scale heuristic compares gradient covariance with squared mean gradient, $\operatorname{tr}(\Sigma)/\|g\|^2$. Its interpretation depends on the objective, parameterization, and sampling scheme.

**Worked example.** Suppose $S_\infty=1000$ and $E_0=10^6$. Then $B_{\rm crit}=1000$. At batch 100, the model predicts 11,000 steps and 1.1 million examples; at batch 10,000, 1100 steps but 11 million examples. A cluster may process the larger batch efficiently while the learning algorithm uses far more data to reach the target.

## 5. Derive the model–data allocation at fixed compute [56:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3360s)

Let $N$ be parameter count and $D$ training tokens. Assume the fitted loss is

$$L(N,D)=L_\infty+AN^{-\alpha}+BD^{-\beta},$$

and training cost is $C=\kappa ND$ with $\kappa\approx6$ for the dense approximation from Lecture 2. Fixing $C$ gives $D=C/(\kappa N)$. Substitute:

$$L(N)=L_\infty+AN^{-\alpha}+B(\kappa N/C)^\beta.$$

Differentiate and set the derivative to zero:

$$\alpha AN^{-\alpha}=\beta B(\kappa N/C)^\beta.$$

Therefore

$$N_* =\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\left(\frac C\kappa\right)^{\beta/(\alpha+\beta)},$$

$$D_*\propto C^{\alpha/(\alpha+\beta)}.$$

When $\alpha\approx\beta$, both scale approximately as $\sqrt C$. This is the structural basis for balanced scaling under this particular model. The constants depend on data, architecture, optimization, and counting conventions.

**Worked example.** Suppose a fitted optimum at one budget is a one-billion-parameter model trained on 20 billion tokens. With equal exponents and sixteen times the budget, the same law predicts four billion parameters and 80 billion tokens. Holding tokens per parameter constant follows here from the assumed equal exponents and unchanged coefficients; it is not a law of nature.

## 6. Why Chinchilla and earlier estimates differed [60:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3600s)

The lecture compares earlier model-heavy allocations with the more balanced allocation in [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). It discusses lower-envelope analysis, fixed-compute sweeps, and fitting a joint loss surface.

An **IsoFLOP curve** fixes compute and varies model size, adjusting token count accordingly. The optimum is the bottom of the curve. Repeating at several budgets estimates how the optimum moves. A joint fit instead assumes a functional form over all $(N,D)$ observations and optimizes that fitted surface.

These methods need not agree exactly. Their fitted regions, schedules, noise models, and parameter accounting differ. A learning-rate schedule optimized for a long run can make early checkpoints look worse than a run intentionally trained to end at that budget. Omitting the output projection can distort compute comparisons for small models where vocabulary work is substantial.

The methodological improvement is to match each run's training horizon, count all significant operations, and reserve extrapolation tests. The [Kaplan et al. paper](https://arxiv.org/abs/2001.08361) remains valuable for its empirical studies even where later allocation estimates differ.

## 7. Researched extension — optimize lifetime cost, not training cost alone [73:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4380s)

The recording closes with a practical distinction: a model can rationally train beyond the token count that minimizes pretraining compute for its size. A smaller model trained longer may be cheaper to serve many times.

An independent simplified lifetime-cost model is

$$C_{\rm life}=\kappa ND+\lambda NQ,$$

where $Q$ is expected inference-token volume and $\lambda$ converts parameter-token inference work into comparable cost. The target loss imposes a relation between $N$ and $D$. As $Q$ grows, reducing $N$ becomes more valuable, potentially justifying a larger $D$.

This model omits cache traffic, context length, batching, and hardware pricing, so it is not a deployment cost calculator. It clarifies why “compute-optimal” must name the objective. Training-optimal, latency-optimal, memory-optimal, and lifetime-cost-optimal designs can differ.

## 8. Exercises and worked solutions

**Exercise 1.** If $\alpha=0.3$ and $\beta=0.2$, how do optimal model size and data scale with compute in the two-term model?

<details><summary>Solution</summary>

$N_*\propto C^{0.2/0.5}=C^{0.4}$ and $D_*\propto C^{0.3/0.5}=C^{0.6}$. Data grows faster than model size. The result depends on the assumed separable loss surface and product compute model.

</details>

**Exercise 2.** A power law fits well over a factor-of-two compute range. Why is a thousandfold forecast questionable?

<details><summary>Solution guidance</summary>

The observed interval may not identify the exponent or asymptote reliably. Optimization, architecture, data reuse, and hardware regimes can change. Fit uncertainty and alternative functional forms can produce widely different distant predictions despite similar local residuals.

</details>

**Exercise 3.** Design a useful small scaling study with twelve runs.

<details><summary>Solution guidance</summary>

One option is three compute budgets and three model sizes per budget, with three additional runs reserved for seed checks or an extrapolation test. Use matched horizon-aware schedules, fixed data processing, consistent FLOP accounting, and a held-out evaluation set. Predeclare the fit and test prediction before observing the largest result. This is a design exercise; the optimal allocation depends on noise and prior knowledge.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=Q15rhEWZPQ4) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_09.pdf).
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162): noise scale and batch efficiency.
