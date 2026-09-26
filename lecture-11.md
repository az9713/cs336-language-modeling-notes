# Lecture 11 — Scaling recipes, optimizers, and hyperparameter transfer

A scaling law is only as reliable as the recipe used to generate its measurements. This lecture examines MiniCPM and DeepSeek case studies, learning-rate and batch-size scaling, optimizer comparisons, and maximal-update parameterization. Its important shift is from fitting a loss curve to designing a family of training runs whose behavior remains comparable as size changes.

## 1. A scaling recipe includes the optimizer and schedule [03:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=180s)

Write a training configuration at model width $d$, depth $L$, and token budget $D$ as a collection

$$\mathcal R(d,L,D)=\{\text{initialization},\eta(t),B(t),\text{normalization},\text{decay},\text{optimizer},\ldots\}.$$

Here $\eta(t)$ is learning rate and $B(t)$ batch size at training time $t$. A scaling experiment changes model size while choosing this recipe. If it leaves a larger model poorly tuned, the observed curve measures a mixture of representational scaling and optimization failure.

The lecture compares two broad strategies. One uses parameterization rules intended to transfer hyperparameters across width. The other empirically sweeps learning rate and batch size at several scales, then fits rules for their optima. Both require evidence; neither licenses arbitrary extrapolation.

**Worked example.** Suppose architecture A reaches loss 2.5 with a well-tuned learning rate while architecture B reaches 2.6 using A's learning rate. If B's own optimum yields 2.4, the original comparison ranked tuning quality rather than architectural potential. A fair study either tunes each method comparably or explicitly studies performance under a shared fixed recipe.

## 2. Warmup–stable–decay makes training horizons reusable [11:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=660s)

A warmup–stable–decay (WSD) schedule first raises learning rate, maintains a stable region, and decays near the end. It can support multiple candidate stopping budgets by branching from checkpoints in the stable region and applying a decay phase for each endpoint.

This differs from treating every intermediate checkpoint of one long cosine schedule as an optimally trained short run. The short checkpoint has not received the same end-of-run schedule as a run designed to stop there.

**Worked example.** To compare 10, 20, and 40 billion-token budgets, a stable trajectory can provide starting checkpoints for appropriately budgeted decay branches. Count the tokens and compute in each branch carefully. Reusing a trajectory reduces experimental cost, but the resulting observations are statistically related and should not be treated as fully independent replications.

WSD does not eliminate all horizon dependence. Warmup duration, decay length, optimizer state, and data order can matter. The advantage is experimental flexibility, not a universal optimality theorem.

## 3. Optimal hyperparameters can depend on different scale variables [16:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=960s)

One can hypothesize rules such as $\eta_*\propto N^{-a}D^b$ or $B_*\propto D^c$, where $N$ is parameter count. Log transforms turn these into regression models. But correlations among $N$, $D$, and compute make causal interpretation difficult when the experiment varies them together.

**Worked example.** If all runs satisfy $D=20N$, then a fit $B\propto D^{1/2}$ is numerically indistinguishable from $B\propto N^{1/2}$ up to a constant. The data cannot determine which variable governs the relationship. To separate them, vary $N$ and $D$ independently over at least part of the study.

The recording notes disagreements among empirical prescriptions. That is a reason to inspect the training region and schedule, not to average the exponents into a supposedly universal rule. Hyperparameter optima can be broad in one regime and sharp in another. Report the loss surface around the optimum, not only its minimizing coordinate.

## 4. Optimizer comparisons require two axes of scale [39:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2340s)

A method can work well on small models trained briefly and behave differently on larger models or longer token budgets. Model scale and training duration are distinct axes. A short-run speed record establishes something valuable, but it does not establish long-horizon stability.

For a gradient $g_t$, Adam maintains first and second moment estimates:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

After bias correction, the update divides the first moment coordinatewise by $\sqrt{v_t}+\epsilon$. This adapts scaling by coordinate; it does not directly exploit the singular-value structure of a weight matrix.

A controlled comparison should tune learning rates for both optimizers, match data and architecture, count optimizer computation and communication, and evaluate stability over the intended horizon. Reporting only loss per step can hide a slower step. Reporting only time to a very early loss can hide later divergence.

## 5. Muon modifies the spectrum of a matrix update [50:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3000s)

For a matrix-shaped momentum update $M\in\mathbb R^{m\times n}$, write a singular value decomposition $M=U\Sigma V^\top$. A polar-style update replaces the nonzero singular values with a common scale, using a direction proportional to $UV^\top$. Practical Muon implementations approximate this transformation with iterative matrix operations and apply shape-dependent scaling and other conventions.

The distinction from Adam is geometric. Coordinatewise rescaling treats entries individually. A spectral transformation changes the matrix as a linear map, emphasizing directions differently according to its singular structure.

**Worked example.** For $M=\operatorname{diag}(100,1)$, the polar factor is the identity matrix. The first direction no longer receives an update 100 times the second solely because of those singular values. The final step magnitude still depends on the optimizer's learning rate and scaling rules. This illustrates the operation, not a proof that equalizing singular values always improves learning.

The lecture discusses evidence and failures as scale changes. A successful large training run establishes feasibility under that recipe; proving superiority over Adam requires a controlled comparison. The primary [Muon implementation](https://github.com/KellerJordan/Muon) is a better source for exact current update details than a short prose summary.

## 6. Width changes the scale of activations and updates [58:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3480s)

Consider $y=Wx$ with $n$ input coordinates. If independent zero-mean inputs have variance one and entries of $W$ have variance $\sigma^2$, then $\operatorname{Var}(y_i)=n\sigma^2$. Choosing $\sigma^2\propto1/n$ keeps activation variance of order one at initialization.

Training introduces another requirement: updates should have a comparable effect on representations as width changes. Expanding the perturbed layer gives

$$\Delta y=(W+\Delta W)(x+\Delta x)-Wx
=W\Delta x+\Delta W x+\Delta W\Delta x.$$

Initialization alone controls neither the direct update term $\Delta W x$ nor the propagated change $W\Delta x$. Learning rates and output scaling may need width-dependent rules.

**Worked example.** If every term in a sum contributes an independent update with fixed variance, widening the sum can enlarge its total variance. If the terms are correlated, the scaling can be even different. This is why a naive “divide everything by square root of width” rule is insufficient for all parameter classes.

## 7. Maximal-update parameterization aims at hyperparameter transfer [63:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3780s)

Maximal-update parameterization, usually written $\mu$P, chooses initialization, learning-rate, and output scaling rules so that useful feature updates remain appropriately scaled in a large-width limit. Input, hidden, and output parameter classes can have different rules.

The primary [Tensor Programs V paper](https://arxiv.org/abs/2203.03466) studies tuning small proxy models and transferring hyperparameters to larger ones. The claim is conditional: the architecture and parameterization must satisfy the relevant setup. Width transfer does not automatically establish depth transfer, token-budget transfer, or invariance under every weight-decay scheme.

The lecture's spectral perspective gives useful intuition: constrain the scale of the linear map and its update, rather than looking only at individual entries. But it should not be used as an improvised substitute for the full parameterization rules. Different optimizer families and tensor shapes require explicit treatment.

**Worked example.** A practical diagnostic plots validation loss against learning rate for several widths. If the minimizing learning rate stays near the same value under the chosen parameterization, transfer is empirically supported in that range. If the minimum shifts systematically, either the rules, implementation, or assumptions need investigation. A flat minimum can make transfer robust even without perfect coincidence.

## 8. Researched extension — preregister a scaling forecast and its failure tests

An independent study protocol follows naturally from the lecture. Choose a target width and token budget, tune only on smaller runs, freeze the scaling rules, and write down the predicted loss and plausible interval before the target run. Also specify failure diagnostics: nonfinite values, gradient spikes, changes in update-to-weight ratios, and a sustained forecast miss.

This turns an attractive retrospective curve into a falsifiable forecast. Record all failed runs, not only the successful branch. Failure evidence can reveal a stability boundary that a smooth fit through survivors hides. A solo researcher can apply this discipline on small models; the scientific value comes from controlled prediction, not merely from expensive hardware.

## 9. Exercises and worked solutions

**Exercise 1.** Derive the variance-preserving initialization for a linear layer under independent unit-variance inputs.

<details><summary>Solution</summary>

$y_i=\sum_jW_{ij}x_j$. With zero means and independence, cross terms vanish and variance is $n\sigma^2$. Set $\sigma^2=1/n$ for unit output variance. Nonlinearities and correlations alter the appropriate constant and assumptions.

</details>

**Exercise 2.** Why does a successful width-transfer experiment not prove that the same learning rate works at twice the depth?

<details><summary>Solution</summary>

Depth changes the number of composed transformations and residual updates. The scaling limit and parameter classes used for width transfer do not automatically control accumulation through more layers. Depth needs its own analysis or experiment.

</details>

**Exercise 3.** A new optimizer takes 20% fewer steps but each step is 30% slower. Does it reduce training time?

<details><summary>Solution</summary>

The relative time is $0.8\times1.3=1.04$: 4% slower at the same target, before other overheads. Quality per step and quality per second are different comparisons.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=vTfEyOyzV9E) and [official slides](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_11.pdf): case studies, schedules, optimizer scaling, and spectral intuition.
- [Tensor Programs V](https://arxiv.org/abs/2203.03466): maximal-update parameterization and hyperparameter transfer.
- [Muon](https://github.com/KellerJordan/Muon): primary algorithm implementation.
- [Adam](https://arxiv.org/abs/1412.6980): adaptive moment estimation.
