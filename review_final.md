# Reviewer QbaF

We thank reviewer QbaF for recognizing the structured modeling of expert residuals, the tractability of the closed-form updates and the feasibility of the approach.

> While the proposed model structure is interesting, the paper provides limited discussion on statistical guarantees. So it is difficult to compare the proposed method with existing approaches.

Our goal was to study the full routing problem --- non-stationary latent regimes, partial feedback, heterogeneous experts, time-varying availability --- all at once. A meaningful regret analysis in this setting is substantially more delicate than in standard stationary contextual bandits: a simplified theorem would require dropping one or more of these ingredients, which changes the problem.

The paper does prove structural results specific to our formulation: **cross-expert information transfer** (Proposition 1), **pruning invariance** (Proposition 2), and **coupling at birth** (Proposition 3). To complement these, we expanded the empirical comparison with broader baselines, a new benchmark, and component ablations (see Reviewer iNRb for ablation tables).

> The computational cost of the proposed method is not clearly analyzed. Can you provide a clear discussion of the computational complexity of the proposed method?

The per-step cost is $T_t^{\mathrm{online}} = \widetilde O\big(C_\theta(x_t) + M d_g^3 + M |\mathcal K_t| d_\alpha^3+ |\mathcal E_t|\, C_{\mathrm{score}}(M,d_g,d_\alpha,S)\big)$ with memory $\mathrm{Mem}_t^{\mathrm{online}} = O\big(M d_g^2 + M |\mathcal K_t| d_\alpha^2\big)$, where $M$ is the number of regimes, $d_g$/$d_\alpha$ the shared/idiosyncratic dimensions, $\mathcal K_t$ the maintained registry, and $S$ the Monte Carlo budget.

The key point: both scale with the **maintained registry** $|\mathcal K_t| \le |\mathcal E_t| + \Delta_{\max}$, **not the cumulative catalog**. Baselines like per-expert LinUCB store one ridge matrix per expert ever seen --- $O(K_t^{\mathrm{cat}} d^2)$ --- growing unboundedly. Proposition 3 shows pruning preserves the retained marginals.

To validate this, we ran a **registry-churn experiment** where four experts are active but the cumulative catalog grows from $R=4$ to $R=64$ through periodic replacement. We report each method's maintained online state (total scalar entries in sufficient statistics, averaged over time).

**Maintained online state** (average scalar entries, $R=4$ vs $R=64$):

| Method | $R=4$ | $R=64$ | Growth |
| --- | ---: | ---: | ---: |
| L2D-SLDS | 30.0 | 49.2 | 1.6x |
| SharedLinUCB | 90 | 16,770 | 186x |
| LinTS | 90 | 16,770 | 186x |
| Ensemble | 522 | 22,962 | 44x |
| NeuralUCB | 1,120 | 17,440 | 16x |

**Routing cost under churn** (same experiment):

| $R$ | L2D-SLDS | SharedLinUCB | LinTS | Ensemble | NeuralUCB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 13.25 | 22.96 | 22.97 | 23.77 | 18.94 |
| 8 | 13.88 | 20.69 | 24.92 | 23.89 | 20.73 |
| 16 | 12.88 | 21.82 | 22.84 | 20.68 | 20.14 |
| 32 | 14.13 | 21.11 | 24.46 | 24.18 | 26.76 |
| 64 | 12.42 | 17.46 | 19.19 | 20.55 | 19.22 |

The SLDS state grows **less than 2x** while SharedLinUCB/LinTS grow **~200x**, without sacrificing quality --- L2D-SLDS is **best at every catalog size**. At $R=64$, runtime is 1.89 ms/step vs 4.98 (SharedLinUCB), 2.93 (LinTS), 2.61 (Ensemble), 0.75 (NeuralUCB).

> The experimental comparison is limited to LinUCB and NeuralUCB. It is unclear why the method is not compared with other exploration strategies such as Thompson sampling or ensemble sampling (Lu and Van Roy, 2017).

> In the LinUCB setting, the paper uses a separate parameter for each expert. [...] It would be helpful to understand the motivation for this choice, and whether comparisons with a shared-parameter LinUCB would change the results.

We added Shared-Parameter LinUCB, Linear Thompson Sampling, and Ensemble Sampling on Jena (new benchmark) and 
Melbourne (see below)

> Also, it would be helpful to understand how the runtime compares to the baselines used in the experiments.

**Jena**

| Method | Mean cost | Runtime (ms/step) |
| --- | ---: | ---: |
| L2D-SLDS | 3.6100 | 2.295 |
| L2D-SLDS w/o $g_t$ | 3.6633 | 1.564 |
| Shared-Parameter LinUCB | 3.9220 | 5.805 |
| NeuralUCB | 4.1537 | 0.658 |
| Linear Thompson Sampling | 4.0733 | 3.519 |
| Ensemble Sampling | 3.9789 | 3.480 |
| LinUCB | 5.1433 | 0.216 |

**Melbourne**

| Method | Mean cost | Runtime (ms/step) |
| --- | ---: | ---: |
| L2D-SLDS | 5.6959 | 0.745 |
| L2D-SLDS w/o $g_t$ | 5.7540 | 0.643 |
| Shared-Parameter LinUCB | 5.9582 | 4.873 |
| NeuralUCB | 5.8780 | 0.426 |
| Linear Thompson Sampling | 5.9267 | 2.242 |
| Ensemble Sampling | 5.9336 | 2.319 |
| LinUCB | 6.0998 | 0.178 |

The proposed method is **best on both benchmarks** and **faster than the three stronger baselines** on both datasets. Component ablations (regime count, EM, transitions) are in our response to Reviewer iNRb.


# Reviewer W5VY

We thank reviewer W5VY for highlighting the novelty of the time-series perspective for expert modeling and the shared-factor / individual-factor decomposition as a relatively unexplored direction.

> I found the paper difficult to read and hard to follow. [...] For example, the abstract includes abbreviations such as L2D-SLDS and IDS without spelling out their full names. [...] the shared factor and individual states are introduced at the beginning of Section 4.2, but their connection to the problem only becomes clear much later in Equation (13).

In the revision we will: (1) spell out L2D-SLDS and IDS at first use in both the abstract and introduction; (2) 
restructure Section 4.2 to lead with the emission model, then introduce $g_t$, $u_{t,k}$, $z_t$ as successive motivated extensions.

> My second concern relates to the evaluation. [...] The synthetic experiment is relatively simple, relying on a linear AR(1) model, and the comparisons are restricted to classical bandit approaches such as LinUCB and NeuralUCB. Overall, neither the theoretical results nor the empirical evaluation fully justify the proposed methodology.

We address the empirical and theoretical sides in turn.

**Synthetic experiment.** The AR(1) setting is simple by design --- it isolates a single question: does modeling cross-expert dependencies help? The bandit baselines fail to model this cross-expert correlation, while our shared factor $g_t$ captures it explicitly --- the gap on synthetic reflects that difference. A more complex data-generating process would conflate multiple effects; the simple setting keeps the comparison interpretable.

**Broader baselines and new dataset.** We added Shared-Parameter LinUCB, Linear Thompson Sampling, and Ensemble 
Sampling, and introduced a second real-world benchmark on **Jena Climate** (T=6000, 13-dim context). The method is **best on both datasets** (3.6100 vs 3.9220 for the strongest baseline on Jena; 5.6959 vs 5.8780 on Melbourne). See our response to Reviewer QbaF for full tables and runtime numbers.

**Component ablations.** We ran three ablations to validate each piece of the model (see detailed tables in our response to Reviewer iNRb):

- *Regime count* ($M \in \{1,2,4\}$, real data, five-seed average): $M=1$ is not best on either dataset --- regimes help, and a small $M$ often suffices.
- *EM estimation* (synthetic, misspecified init): EM cuts parameter error by **more than half** (RMSE 0.0809 → 0.0363) and improves routing cost, even with a conservative budget of 8 iterations.
- *Context-dependent transitions* (synthetic): transition accuracy jumps from **8% to 60%** with the learned model.

**Runtime and scalability.** Our method runs at 0.7--2.3 ms/step, faster than the shared-parameter baselines that are its closest competitors in routing cost. We also ran a registry-churn experiment (see our response to Reviewer QbaF) where the expert catalog grows from 4 to 64 identities under churn: the SLDS maintained state grows **less than 2x** while baseline state grows **nearly 200x**.

**Theory.** The setting combines non-stationary regimes, partial feedback, expert heterogeneity, and time-varying availability --- standard regret techniques do not apply without dropping one of these. The paper provides structural results (Propositions 1--3: information transfer, pruning invariance, coupling at birth) rather than regret bounds. We see regret analysis for this full setting as an open direction.

> While $C_k(x, y)$ is explicitly defined in Equation (1), $C_I(x, y)$ is not clearly defined when $I$ is a set. [...] What is the definition of $C_I$ on Line 81, Page 2?

$I$ is not a set --- it denotes the selected expert index as defined L-80. The intended meaning is the scalar cost $C_{I_t}(x_t, y_t)$ with $I_t \in \mathcal{E}_t$. We will make this more explicit.

> Would it be possible to establish an information-theoretic lower bound for the cumulative cost under the proposed setting, and to determine whether the regret of the proposed method matches this lower bound?

A clean lower bound for the full setting would require simplifying away one or more of: latent non-stationarity, censoring, expert heterogeneity, time-varying availability. Standard tools like Fano's inequality assume a fixed parameter class, which does not apply when latent regimes shift over time. We see this as an open direction but not one that can be resolved within the current framework without changing the problem.

> I might have missed something, but I did not find adequate discussion of the limitations of the proposed methodology in the paper.

We will add a limitations paragraph. The paper does not provide regret-style or minimax guarantees; the theoretical contribution is structural (information transfer, pruning invariance), not regret-theoretic. Routing quality depends on the quality of the SLDS parameters, whether set manually or via EM.


# Reviewer iNRb

We thank reviewer iNRb for recognizing that the SLDS construction for belief modeling over correlated expert costs is well conceived, and for the detailed breakdown of the four model components.

> The whole setup covers all bases of various modeling needs that this problem domain might need. Since the paper presents a new method without theoretical analysis, a more comprehensive experimental validation would have made a better case.

We expanded the evaluation with a second real-data benchmark (**Jena Climate**), reran **Melbourne** with the full baseline set, and added Shared-Parameter LinUCB, Linear Thompson Sampling, and Ensemble Sampling (with runtime analysis).
We also ran a registry-churn scalability experiment showing that the maintained SLDS state stays bounded even as 
the expert catalog grows 16-fold, while baseline state grows by orders of magnitude (see the full table in our response to Reviewer QbaF).

> In practice, the predicted costs require model parameters such as , etc., and these would need to be estimated from data. The main body of the paper does not talk about estimation at all. It is presented in Algorithm 3 in the appendix.

The paper separates **parameter estimation** from **online routing** by design. The structural results --- cross-expert transfer (Proposition 1), pruning invariance (Proposition 2), coupling at birth (Proposition 3) --- hold for any parameter setting; the routing rule is not contingent on EM. EM is one way to obtain the parameters, but the online mechanism is self-contained. We will make this separation clearer in the main text.

> Once you factor in an estimation phase, the sample complexity will depend crucially on the size of the model. With that in mind, the whole idea of regimes and its soft-max-based model appears somewhat of an overkill. A discussion to that effect should be included.

We ran three ablations to isolate each component.

**1. Regime count (real data, five-seed average).** $M \in \{1,2,4\}$, everything else held fixed:

| Dataset | $M=1$ | $M=2$ | $M=4$ | Best |
| --- | ---: | ---: | ---: | ---: |
| Jena | 3.6572 | 3.6749 | 3.6100 | $M=4$ |
| Melbourne | 5.7181 | 5.6959 | 5.7093 | $M=2$ |

$M=1$ is **not best on either dataset** --- regimes do help. But the best $M$ is small and dataset-dependent, and the online cost scales linearly in $M$. Going from $M=1$ to $M=2$ adds one extra mode-conditioned filter per expert --- a modest overhead.

**2. Transition model (synthetic with context-dependent switching).** To validate the context-dependent transition mechanism, we ran a synthetic where the true transition depends on context sign:

| Variant | Transition NLL | Transition acc. | Regime acc. | Transition-matrix error |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 0.6981 | 0.0782 | 0.4810 | 0.7583 |
| Attention-learned | 0.5830 | 0.6012 | 0.5190 | 0.6383 |

Transition accuracy goes from **8% to 60%**, and the learned transition matrix is substantially closer to the ground truth.

**3. EM (synthetic, misspecified init, five seeds).** We deliberately perturb the SLDS parameters away from their true values and measure whether EM can recover them.

| Variant | Parameter RMSE |
| --- | ---: |
| Matched (ground truth) | 0.0000 |
| Misspecified, no EM | 0.0809 |
| Misspecified + EM (8 iterations) | 0.0363 ± 0.0027 |

EM reduces parameter error by **more than half** with only 8 iterations on a short warmup prefix, and the recovered parameters also improve downstream routing cost. A larger EM budget or longer prefix would push recovery further.

**Putting it together:** **none of the three components is redundant** --- removing regimes hurts on both real datasets, removing context-dependent transitions loses most of the switching signal, and removing EM leaves misspecified parameters unrecovered. Each is validated on the setting where it matters most. The model is a hierarchy rather than a monolith: practitioners select the complexity their setting requires, and simpler operating points ($M=1$, uniform transitions, no EM) are special cases. We will add a discussion of this tradeoff in the paper.

> The paper spends a lot of real estate in ``developing'' the model, but it does not quite provide a straightforward set of algorithms to implement it.

The full implementation is specified in the appendix: Algorithm 1 (routing loop), Algorithm 2 (Kalman update + mode reweighting), Algorithm 3 (Monte Carlo EM). We will add clearer forward references from the main text.

> It would have been easier if the linear-Gaussian model was presented way before and then it was explained.

Good suggestion. We will restructure Section 4.2 to lead with the linear-Gaussian model, then explain each component.
