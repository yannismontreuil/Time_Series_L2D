# Rebuttal Objective & Action Plan

## A. Paper Understanding

### Problem
Online Learning-to-Defer for non-stationary time series. At each round t, a router sees context x_t and an available expert set E_t, picks one expert I_t, and only observes that expert's prediction (bandit/partial feedback). The data can be non-stationary (regime shifts), and experts come and go.

### Model (L2D-SLDS)
A factorized switching linear dynamical system over signed expert residuals e_{t,k} = y_hat_{t,k} - y_t. Three-level latent hierarchy:

1. **Discrete regime z_t in {1,...,M}** with context-dependent transitions Pi_theta(x_t) parameterized via scaled-attention (low-rank QK^T).
2. **Shared global factor g_t in R^{d_g}** — linear-Gaussian dynamics per regime. This is the key innovation: it couples all experts so querying one expert updates beliefs for all.
3. **Per-expert idiosyncratic state u_{t,k} in R^{d_alpha}** — captures expert-specific drift. Dynamics params shared across experts.

**Emission:** alpha_{t,k} = B_k g_t + u_{t,k}, then e_{t,k} ~ N(Phi(x_t)^T alpha_{t,k}, R_{m,k}).

**Inference:** IMM-style filtering — maintain M mode-conditioned beliefs, Kalman update on joint (g_t, u_{t,I_t}), project to factorized form (discard cross-covariance for scalability).

**Routing (IDS):** Minimize Delta_t(k)^2 / IG_t(k) where:
- Delta_t(k) = predicted cost gap vs. best expert
- IG_t(k) = I((z_t, g_t); e_{t,k} | F_t) = mode-identification (MC) + shared-factor refinement (closed-form)

**Registry management:** Prune stale experts (unavailable + not queried for > Delta_max steps). Proposition 2 proves this is exact marginalization. New experts get initialized and immediately couple to others via B_k g_t (Proposition 3).

**Theory:** Three propositions: (1) information transfer iff predictive cross-covariance != 0, (2) pruning invariance, (3) coupling at birth. No regret bounds.

**Experiments:** Synthetic (M=2, 4 experts, block regime switching, correlated noise), Melbourne (5 experts, daily temp), Jena (5 experts, 4-hourly temp), FRED DGS10 (4 experts, daily yield).

---

## B. Codebase Structure

- **`models/factorized_slds.py`** (~3000 lines): Core `FactorizedSLDS` class inheriting from `SLDSIMMRouter`. Implements the full model: init, registry management (`manage_registry`, `_birth_expert`), time updates, Kalman updates (`_update_from_predicted`), predictive stats (`_compute_predictive_stats`), scoring (`_score_experts`), IDS selection (`_select_expert_from_stats`), EM learning, and exploration (modes: `g`, `g_z`, `ucb`, `sampling`).
- **`models/shared_linear_bandits.py`**: New baselines (SharedLinUCB, LinearThompsonSampling, LinearEnsembleSampling) using joint (context, expert) features.
- **`models/linucb_baseline.py`**, **`models/neuralucb_baseline.py`**: Original baselines.
- **`environment/synthetic_env.py`**: Synthetic environment with regime-dependent AR(1) target and correlated expert noise.
- **`environment/etth1_env.py`**: Real-data environments (Melbourne, Jena, FRED).
- **`slds_imm_router.py`**: Main entry point, loads config, builds routers, runs experiments.
- **`router_eval.py`**: Evaluation loop, runs select/update cycle.
- **`scripts/registry_complexity_experiment.py`**: NEW — scalability experiment showing cost vs. growing catalog size.
- **`scripts/measure_jena_runtime.py`**, **`scripts/measure_melbourne_runtime.py`**: Runtime measurement scripts.
- **Configs**: YAML files per experiment with full hyperparams.

---

## C. Reviews — What Each Reviewer Wants

### Reviewer QbaF (Rating 4: Weak Accept, Confidence 1/low)
- W1: No statistical guarantees — hard to compare
- W2: Computational cost unclear
- W3: Only LinUCB/NeuralUCB baselines; no Thompson/Ensemble/shared-param LinUCB
- **Status**: Most sympathetic reviewer. W2-W3 addressed (runtime tables, 3 new baselines). W1 handled with honest scoping. This reviewer can likely be moved to accept.

### Reviewer iNRb (Rating 3: Weak Reject, Confidence 3/moderate)
- No theory — needs more comprehensive experiments
- Model parameters need estimation but paper ignores it in main body; regimes seem "overkill"
- Presentation: too much model development, no clear algorithms
- **Status**: Most dangerous reviewer (highest confidence among rejectors). Partially addressed: new experiments added but response to "regimes overkill" is still unfinished. Presentation restructuring promised but not done.

### Reviewer W5VY (Rating 3: Weak Reject, Confidence 2/low)
- Presentation: hard to read, abbreviations unexplained, motivation unclear
- Evaluation: needs stronger theory OR more experiments; synthetic too simple, baselines limited
- C_I notation unclear; no limitations discussion
- **Status**: Baselines/experiments addressed. Presentation and limitations discussion promised but not done.

---

## D. What's Already Done for Rebuttal

1. Added 3 new baselines: Shared LinUCB, Thompson Sampling, Ensemble Sampling
2. Added Jena Climate experiment (real-world, T=6000)
3. Runtime tables for Jena and Melbourne (ms/step comparisons)
4. Computational complexity analysis (O-notation + registry bound)
5. Tuned operating-point sweeps (showing runtime can be halved with minimal cost increase)
6. Started `registry_complexity_experiment.py` for scalability under growing catalog
7. Drafted `review_final.md` with most responses
8. Drafted `complexity.tex` with formal complexity analysis

---

## E. What's Still Missing — Prioritized Action Plan

Current scores: **4, 3, 3** (Weak Accept, Weak Reject, Weak Reject). Need to flip at least one reject.

### PRIORITY 1: Complete the Reviewer iNRb response (biggest risk)

**1a. Answer the "regimes overkill" point (currently empty)**
This is iNRb's core concern and the response is missing. Need:
- An **ablation on M** (number of regimes): run Melbourne/Jena with M=1 (no switching, just a single LDS) vs M=2 vs M=4. Show that M>1 actually helps on real data (or if M=1 is competitive, argue the extra cost is small and the flexibility is worth it).
- Argue that regimes are cheap: complexity scales linearly in M (not exponentially), and the tuned settings show M=2 with d_g=1 gets similar performance at ~1.7ms/step.
- Point to the synthetic experiment where M=2 regime switching is the ground truth and the method correctly identifies it.

**1b. Finish the scalability/registry experiment**
iNRb wants "more comprehensive experiments." The `registry_complexity_experiment.py` script is ready. Run it and include results showing that as catalog size grows (4 -> 8 -> 16 -> 32 -> 64), L2D-SLDS runtime stays flat (because |K_t| is bounded by |E_t| + Delta_max) while baseline memory/compute grows with the full catalog.

**1c. Respond to "estimation = overkill" more concretely**
Point out that in the Jena and Melbourne experiments, **no EM is used at all** — parameters are fixed at initialization. The model works without the estimation phase on real data. This directly counters the "sample complexity of estimation" concern.

### PRIORITY 2: Strengthen the paper text (presentation fixes promised to W5VY and iNRb)

**2a. Expand abbreviations in abstract**
L2D-SLDS -> "Learning-to-Defer Switching Linear Dynamical System (L2D-SLDS)"; IDS -> "Information-Directed Sampling (IDS)"

**2b. Restructure Section 4.2**
As promised: start with the linear-Gaussian emission model (Definition 1), then introduce each component (g_t, u_{t,k}, z_t) as motivated extensions.

**2c. Add a Limitations paragraph**
In the conclusion or after it, add explicit limitations: no regret bounds, dependence on model specification quality, the factorized approximation discards cross-covariance.

**2d. Fix the C_I notation** (W5VY's minor point) — clarify it's C_{I_t}, a scalar for the selected expert.

### PRIORITY 3: Additional empirical evidence to push beyond "weak" territory

**3a. Ablation table: exploration rule**
Already have `g` vs `g_z` in the code. Show a table comparing: greedy (no exploration), g-only, g_z, UCB exploration modes. This shows the IDS routing matters.

**3b. If time permits: one more diverse dataset**
The FRED experiment already exists in the appendix. Make sure the rebuttal tables reference it. Alternatively, run on a multivariate or fundamentally different domain.

### PRIORITY 4: Polish review_final.md into camera-ready rebuttal format

- Fill in the "blabla" placeholder for iNRb's "regimes overkill" concern
- Add the registry scalability results
- Add the M-ablation results
- Make sure the general comment section highlights all new additions concisely

---

## Summary

The strongest card is the empirical results — L2D-SLDS consistently beats all baselines including the 3 new ones requested by reviewers. The critical gaps are:
1. The unanswered "regimes overkill" point for iNRb
2. The missing scalability experiment
3. The promised presentation fixes
