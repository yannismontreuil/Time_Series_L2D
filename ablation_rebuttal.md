# EM Ablation for Rebuttal

## What this ablation is meant to show

Objective: show that the EM component is useful when the SLDS is misspecified, by improving both parameter recovery and post-prefix routing quality on a controlled synthetic setting where misspecification can be introduced deliberately.

This ablation is the cleanest way to justify the EM component without overclaiming.

The point is not that the full estimation stack is always needed on every dataset.  
The point is narrower and stronger:

- when the SLDS parameters are already well specified, no EM is needed;
- when the SLDS parameters are misspecified, EM moves them back toward the true synthetic model and improves post-prefix routing performance.

This supports the intended narrative:

- on **synthetic**, where the latent structure is controlled and misspecification can be introduced deliberately, EM has a real role;
- on **real data**, simpler fixed-parameter operating points can already be enough.

## Setup

We use the paper synthetic with the `tri_cycle_corr` setting and evaluate only the **post-prefix tail**.

- Environment: same synthetic data across runs
- Warmup prefix for offline EM: `em_tk = 2000`
- EM setting: `n_em = 8`, `n_samples = 10`, `burn_in = 20`
- Misspecification profile: `mild`
- Seeds: `11,12,13,14,15`

We compare:

- `Matched, no EM`: parameters initialized at the synthetic ground-truth setting
- `Misspecified, no EM`: parameters deliberately perturbed, then kept fixed
- `Misspecified + EM`: same misspecified initialization, then offline EM on the warmup prefix

## Main table

| Variant | Post-prefix mean cost | Post-prefix cumulative cost | Parameter RMSE after fit |
| --- | ---: | ---: | ---: |
| Matched, no EM | `12.3256` | `12313.28` | `0.0000` |
| Misspecified, no EM | `15.6740` | `15658.32` | `0.0809` |
| Misspecified + EM | `15.5173 ± 0.0465` | `15501.82 ± 46.47` | `0.0363 ± 0.0027` |

## Key readout

- Deliberate misspecification materially hurts performance: mean post-prefix cost rises from `12.3256` to `15.6740`.
- Starting from that same misspecified initialization, offline EM improves the post-prefix tail cost to `15.5173 ± 0.0465`.
- This is a recovery of `0.1567 ± 0.0465` mean-cost units on the tail, i.e. about `156.5 ± 46.5` cumulative cost units over the post-prefix horizon.
- EM also cuts the parameter error substantially:
  - before EM: `0.0809`
  - after EM: `0.0363 ± 0.0027`

## Interpretation

This is the right conclusion to draw:

- EM is **not** needed when the model is already well specified.
- EM is **useful** when the parameters are misspecified: it improves both parameter recovery and downstream routing cost.
- The result is intentionally conservative: EM does not fully recover the matched model, but it moves the system in the right direction on both parameter fit and decision quality.

That is exactly the role EM should play in the rebuttal. It justifies why the estimation machinery exists, without forcing the stronger and less credible claim that the richest configuration must always be used in practice.

## Suggested rebuttal wording

> To clarify the role of EM, we ran a synthetic misspecification ablation using the paper's `tri_cycle_corr` setting. We compare (i) a matched no-EM model, (ii) a deliberately misspecified no-EM model, and (iii) the same misspecified initialization followed by offline EM on a warmup prefix (`em_tk=2000`, `n_em=8`, `n_samples=10`, `burn_in=20`, five seeds). Misspecification increases the post-prefix mean cost from `12.3256` to `15.6740`. Starting from that same misspecified initialization, offline EM reduces the post-prefix mean cost to `15.5173 ± 0.0465`, corresponding to a gain of `0.1567 ± 0.0465` mean-cost units, or about `156.5 ± 46.5` cumulative cost units on the tail. At the parameter level, EM reduces the RMSE from `0.0809` to `0.0363 ± 0.0027`. The intended reading is therefore not that EM is always necessary, but that it is useful when the SLDS parameters are misspecified, while simpler fixed-parameter operating points can already suffice on the real-data benchmarks.

## Internal note

Cluster source for the five-seed confirmation:

- Slurm job: `491364`
- Config selected from one-seed search:
  - `misspec-profile = mild`
  - `em_tk = 2000`
  - `em_n = 8`
  - `em_samples = 10`
  - `em_burn_in = 20`

---

# Transition Ablation for Rebuttal

## Objective

Objective: show that the learned context-dependent transition model is useful when regime switching truly depends on context, by improving transition prediction and regime tracking relative to a uniform transition model.

## What this ablation is meant to show

This experiment is meant to justify the transition component itself.

The intended claim is not that the learned transition model must always reduce routing cost on every dataset.  
The intended claim is narrower:

- when the true regime transitions depend on context, a learned context-dependent transition model should fit those switches better than a uniform transition model;
- this should appear first in transition-focused metrics, such as transition likelihood, transition prediction accuracy, and regime tracking.

That is exactly how this experiment should be read: as a component-validation result for the transition layer.

## Setup

We use a dedicated synthetic setting (`context_gate_corr`) in which the true regime transition matrix depends on the sign of the observed context.

- Two regimes
- Context-dependent true transition:
  - one transition matrix when `x_t < 0`
  - a different transition matrix when `x_t >= 0`
- Same routing model and same synthetic environment in both variants
- Comparison:
  - `Uniform transitions`: fixed uniform transition model
  - `Attention-learned transitions`: learned context-dependent transition model

Best run selected from the Slurm search:

- `em_tk = 2000`
- `n_em = 8`
- `n_samples = 12`
- `burn_in = 20`
- `theta_lr = 0.05`
- `theta_steps = 120`

## Main table

| Variant | Transition NLL | Transition accuracy | Regime accuracy | Transition-matrix error |
| --- | ---: | ---: | ---: | ---: |
| Uniform transitions | `0.6981` | `0.0782` | `0.4810` | `0.7583` |
| Attention-learned transitions | `0.5830` | `0.6012` | `0.5190` | `0.6383` |

Here:

- `Transition NLL` is the held-out negative log-likelihood of the true regime switch under the model transition probabilities;
- `Transition accuracy` is the accuracy of predicting the next regime from the transition model;
- `Regime accuracy` is the filtering accuracy of the inferred regime;
- `Transition-matrix error` is the Frobenius error between the learned transition matrix and the true context-dependent transition matrix.

## Key readout

- The learned transition model substantially improves transition prediction:
  - NLL drops from `0.6981` to `0.5830`
  - transition accuracy rises from `0.0782` to `0.6012`
- Regime tracking also improves:
  - regime accuracy rises from `0.4810` to `0.5190`
- The learned transition model is also closer to the true context-dependent dynamics:
  - transition-matrix error drops from `0.7583` to `0.6383`

## Interpretation

This is the right conclusion to draw:

- when the true switching mechanism is context-dependent, the learned transition model captures that dependence better than a uniform transition model;
- this is visible in the transition metrics and in regime tracking;
- in this particular synthetic setup, the downstream routing cost is essentially unchanged, so this experiment should be read as a validation of the transition component itself rather than as a new routing benchmark.

## Suggested rebuttal wording

> To justify the context-dependent transition model, we ran a dedicated synthetic experiment in which the true regime transition matrix depends on the sign of the observed context. We compare a fixed uniform transition model against the learned attention-based transition model. The learned transition model gives substantially better transition metrics: the held-out transition NLL drops from `0.6981` to `0.5830`, transition accuracy rises from `0.0782` to `0.6012`, regime accuracy rises from `0.4810` to `0.5190`, and the transition-matrix error drops from `0.7583` to `0.6383`. The routing cost is essentially unchanged in this setup, so the correct reading of this experiment is that the learned transition layer better captures context-dependent switching, rather than that it serves as a standalone performance benchmark.

## Internal note

Cluster source:

- Slurm search job: `491805`
- Best run:
  - `tk2000_n8_s12_b20_lr0.05_ts120`

---

# Regime-Count Ablation for Rebuttal

## Objective

Objective: show whether multiple regimes are actually useful on the real-data benchmarks, and whether the best operating point requires the largest regime count or a smaller one.

## What this ablation is meant to show

This experiment addresses the “overkill” criticism directly.

The point is not that larger `M` is always better.  
The point is:

- `M=1` tests whether the regime mechanism can be removed entirely;
- `M=2` tests whether a small switching model is already sufficient;
- `M=4` tests whether the richer switching model is needed.

This gives the right kind of answer to the reviewer: whether regimes help at all, and whether the full regime count is actually necessary on each real dataset.

## Setup

We run the proposed method only, keeping the dataset protocol fixed and changing only the regime count.

- Jena:
  - shared factor kept at `d_g = 1`
  - exploration fixed to `g`
- Melbourne:
  - shared factor kept at `d_g = 1`
  - exploration fixed to `g_z`
  - Monte Carlo budget fixed to `12`

Seeds: `11,12,13,14,15`

## Main table

| Dataset | `M=1` | `M=2` | `M=4` | Best |
| --- | ---: | ---: | ---: | ---: |
| Jena | `3.6572` | `3.6749` | `3.6100` | `M=4` |
| Melbourne | `5.7181` | `5.6959` | `5.7093` | `M=2` |

## Key readout

- `M=1` is not best on either dataset.
- On **Jena**, the richer switching model helps:
  - `M=4` is best at `3.6100`
  - `M=1` is worse at `3.6572`
  - `M=2` is worse still at `3.6749`
- On **Melbourne**, the best operating point is smaller:
  - `M=2` is best at `5.6959`
  - `M=4` is slightly worse at `5.7093`
  - `M=1` is worse at `5.7181`

## Interpretation

This is the right conclusion to draw:

- the regime mechanism is useful, since `M=1` is not best on either real dataset;
- at the same time, the largest regime count is not always needed;
- the practical operating point is dataset-dependent:
  - Jena benefits from the richer `M=4` setting,
  - Melbourne is best with the simpler `M=2` setting.

So the fairest summary is not “the full model is always necessary,” but rather:

- the regime component matters,
- and the proposed framework supports simpler operating points when they are sufficient.

## Suggested rebuttal wording

> We also ran a regime-count ablation on the two real-data benchmarks, comparing `M=1,2,4` while keeping the rest of the routing protocol fixed. The key result is that `M=1` is not best on either dataset, so removing the regime mechanism altogether hurts performance. At the same time, the best regime count is dataset-dependent rather than always maximal: Jena is best with `M=4` (`3.6100` vs `3.6572` for `M=1`), whereas Melbourne is best with `M=2` (`5.6959` vs `5.7181` for `M=1`, and `5.7093` for `M=4`). This supports the intended reading of the model as a flexible hierarchy rather than as a claim that the richest setting must always be used.

## Internal note

Cluster source:

- Slurm job: `493033`

---

# Registry-Churn Complexity Check for Rebuttal

## Objective

Objective: show that under expert churn, the maintained SLDS registry stays small even when the cumulative catalog of expert identities grows, so the online state remains controlled.

## What this check is meant to show

This is a complexity-focused supporting experiment, not a new routing benchmark.

The intended point is narrow:

- in our setting, experts may enter and leave over time, so the cumulative catalog
  $K_t^{\mathrm{cat}} := \left|\bigcup_{s=1}^t \mathcal E_s\right|$
  can grow even when only a few experts are active at each round;
- our method stores expert-specific latent state only for the maintained registry $\mathcal K_t$;
- shared baselines that keep catalog-dependent state do not enjoy that same built-in pruning mechanism.

We report all of the main rebuttal baselines:

- `L2D-SLDS`
- `SharedLinUCB`
- `LinTS`
- `Ensemble Sampling`
- `NeuralUCB`

The main scaling table focuses on the quantities that matter most for the pruning argument: the maintained registry size and the maintained online state.

## Setup

We use the paper synthetic with a fixed active set of four experts and a growing cumulative catalog.

- Active experts at any round: `4`
- Total catalog size: `R in {4, 8, 16, 32, 64}`
- Every `30` rounds, one active expert is replaced by a fresh identity from the same latent family
- No EM
- We measure:
  - mean routing cost,
  - average maintained registry size for L2D-SLDS,
  - average online state size, counted as stored scalar array entries

## Main table

| `R` | L2D-SLDS mean cost | $\overline{|\mathcal K_t|}$ | L2D-SLDS avg. state | SharedLinUCB avg. state | LinTS avg. state | Ensemble avg. state | NeuralUCB avg. state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `13.2496` | `4.00` | `30.0` | `90` | `90` | `522` | `1120` |
| `8` | `13.8762` | `4.32` | `31.3` | `306` | `306` | `1122` | `2208` |
| `16` | `12.8760` | `4.96` | `33.8` | `1122` | `1122` | `2706` | `4384` |
| `32` | `14.1303` | `6.23` | `38.9` | `4290` | `4290` | `7410` | `8736` |
| `64` | `12.4214` | `8.80` | `49.2` | `16770` | `16770` | `22962` | `17440` |

Here `Avg. state` means the time-average number of stored scalar entries in the method's maintained online state during the run.

## Largest-catalog snapshot

To show that the state-scaling story is not hiding the routing outcomes, we also report the full `R=64` snapshot below.

| Method | Mean cost at `R=64` | Runtime at `R=64` (ms/step) | Avg. state at `R=64` |
| --- | ---: | ---: | ---: |
| `L2D-SLDS` | `12.4214` | `1.8907` | `49.2` |
| `SharedLinUCB` | `17.4646` | `4.9789` | `16770` |
| `LinTS` | `19.1910` | `2.9312` | `16770` |
| `Ensemble Sampling` | `20.5454` | `2.6110` | `22962` |
| `NeuralUCB` | `19.2210` | `0.7496` | `17440` |

## Full per-catalog snapshot

For completeness, we also report the measured mean cost and average maintained state for every evaluated catalog size.

### Mean cost by catalog size

| `R` | `L2D-SLDS` | `SharedLinUCB` | `LinTS` | `Ensemble Sampling` | `NeuralUCB` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4` | `13.2496` | `22.9579` | `22.9736` | `23.7692` | `18.9416` |
| `8` | `13.8762` | `20.6917` | `24.9229` | `23.8949` | `20.7307` |
| `16` | `12.8760` | `21.8244` | `22.8350` | `20.6846` | `20.1360` |
| `32` | `14.1303` | `21.1061` | `24.4606` | `24.1827` | `26.7560` |
| `64` | `12.4214` | `17.4646` | `19.1910` | `20.5454` | `19.2210` |

### Average maintained state by catalog size

| `R` | `L2D-SLDS` | `SharedLinUCB` | `LinTS` | `Ensemble Sampling` | `NeuralUCB` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4` | `30.0` | `90` | `90` | `522` | `1120` |
| `8` | `31.3` | `306` | `306` | `1122` | `2208` |
| `16` | `33.8` | `1122` | `1122` | `2706` | `4384` |
| `32` | `38.9` | `4290` | `4290` | `7410` | `8736` |
| `64` | `49.2` | `16770` | `16770` | `22962` | `17440` |

## Key readout

- As the cumulative catalog grows from `4` to `64`, the maintained SLDS registry remains small:
  - $\overline{|\mathcal K_t|}$ grows only from `4.00` to `8.80`
- The maintained online SLDS state therefore stays small as well:
  - from `30.0` to `49.2` average entries
- By contrast, the catalog-dependent baselines grow by orders of magnitude:
  - `SharedLinUCB`: from `90` to `16770`
  - `LinTS`: from `90` to `16770`
  - `Ensemble Sampling`: from `522` to `22962`
  - `NeuralUCB`: from `1120` to `17440`
- This is not masking a cost collapse at large catalog size:
  - at `R=64`, `L2D-SLDS` is also the lowest-cost method at `12.4214`
  - the strongest baseline there is `SharedLinUCB` at `17.4646`
- At the same largest catalog size, the runtime picture remains practical:
  - `L2D-SLDS`: `1.8907` ms/step
  - `SharedLinUCB`: `4.9789`
  - `LinTS`: `2.9312`
  - `Ensemble Sampling`: `2.6110`
  - `NeuralUCB`: `0.7496`

## Interpretation

This is the right conclusion to draw:

- the registry is not just bookkeeping; it keeps the maintained SLDS state controlled under churn;
- the relevant comparison here is state scaling with a growing catalog, not raw milliseconds alone;
- the experiment supports the complexity claim that our online state depends on the maintained registry $\mathcal K_t$, not on the full cumulative catalog.

We deliberately do not use this experiment as the main runtime comparison in the rebuttal. Runtime is reported on Jena and Melbourne, where it is more directly comparable across methods.

## Suggested rebuttal wording

> As a supporting complexity check, we also ran a bounded-active-set, growing-catalog synthetic experiment. The active set is always four experts, but the cumulative catalog grows from `R=4` to `R=64` under expert churn. The key result is that the maintained SLDS registry stays small: the average registry size grows only from `4.00` to `8.80`, and the maintained online state grows only from `30.0` to `49.2` average scalar entries. By contrast, the catalog-dependent baselines grow by orders of magnitude over the same sweep (`SharedLinUCB`: `90` to `16770`; `LinTS`: `90` to `16770`; `Ensemble Sampling`: `522` to `22962`; `NeuralUCB`: `1120` to `17440`). This supports the intended complexity claim that the proposed method's maintained online state scales with the registry $\mathcal K_t$, not the full cumulative catalog of expert identities.

> To show that this is not only a state-size effect, we also report the largest-catalog snapshot at `R=64`. There, the proposed method remains the lowest-cost method (`12.4214`) while keeping a maintained online state of only `49.2` average entries, versus `16770` for `SharedLinUCB`, `16770` for `LinTS`, `22962` for `Ensemble Sampling`, and `17440` for `NeuralUCB`. The corresponding runtime remains practical as well: `1.8907` ms/step for the proposed method, compared with `4.9789` for `SharedLinUCB`, `2.9312` for `LinTS`, `2.6110` for `Ensemble Sampling`, and `0.7496` for `NeuralUCB`.

## Internal note

Source:

- cluster Slurm job: `503357`
- cluster output:
  - `~/scratch/Time_Series_L2D/out/complexity_registry_allbaselines_full/registry_complexity_summary.csv`

## Additional reviewer-requested non-stationary bandit baselines

Objective: answer the reviewer request for comparisons to the cited non-stationary bandit papers, while being explicit about which methods are direct baselines and which ones require contextual adaptation to fit the expert-routing setup.

Compatibility:

- `1909.09146v2` (`D-LinUCB`) is a direct non-stationary linear contextual bandit baseline. We implement it directly as discounted shared linear regression over expert-conditioned features.
- `1711.03539v2` (`CUSUM-UCB`) is a piecewise-stationary non-contextual MAB algorithm. In our setting, the closest compatible analogue is a contextual LinUCB backbone with per-expert CUSUM-based local resets driven by bounded prediction-error magnitudes.
- `1908.10402v4` (`GLR-CUCB`) is a piecewise-stationary combinatorial semi-bandit algorithm. In our setting, the closest compatible analogue is a contextual LinUCB backbone with GLR-based global restart and forced exploration, again driven by bounded prediction-error magnitudes.

We therefore treat:
- `D-LinUCB` as a direct baseline;
- `CUSUM-LinUCB` and `GLR-LinUCB` as careful contextual adaptations of the change-detection ideas in the other two papers.

We ran all three on Slurm on the same tuned Jena and Melbourne environments already used in the rebuttal. For `D-LinUCB`, we additionally ran a small Slurm sweep over the discount factor and kept the best value for each dataset:
- Jena: `gamma = 0.95`
- Melbourne: `gamma = 0.98`

| Method | Jena mean cost | Jena runtime (ms/step) | Melbourne mean cost | Melbourne runtime (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| `L2D-SLDS` | `3.6100` | `1.479` | `5.6959` | `1.688` |
| `NeuralUCB` | `3.9755` | `0.658` | `5.9279` | `0.426` |
| `D-LinUCB` (tuned gamma) | `5.0490` | `1.011` | `5.9889` | `0.677` |
| `CUSUM-LinUCB` | `4.6077` | `0.350` | `6.3191` | `0.324` |
| `GLR-LinUCB` | `5.8584` | `0.453` | `6.2238` | `0.410` |

## Readout

- These additional baselines do not change the ranking on either real dataset.
- On Jena, the strongest of the three is `CUSUM-LinUCB`, but it remains clearly above both `NeuralUCB` and `L2D-SLDS`.
- On Melbourne, the strongest of the three is tuned `D-LinUCB`, but it remains above both `NeuralUCB` and `L2D-SLDS`.
- Runtime remains practical for all three, but none of them closes the performance gap to the tuned `L2D-SLDS`.

## Internal note

Sources:

- cluster Slurm job: `553112`
  - `out/paper_testing/jena_dlinucb.csv`
  - `out/paper_testing/jena_cusum_linucb.csv`
  - `out/paper_testing/jena_glr_linucb.csv`
  - `out/paper_testing/melbourne_dlinucb.csv`
  - `out/paper_testing/melbourne_cusum_linucb.csv`
  - `out/paper_testing/melbourne_glr_linucb.csv`
- cluster Slurm job: `553123`
  - `out/paper_testing_dlin_gamma/*.csv`
