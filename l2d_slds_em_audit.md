# L2D-SLDS and EM Audit (Code-Math Mapping)

This note documents the exact model implemented by the base L2D-SLDS
router (SLDSIMMRouter), the approximate EM update added in
SLDSIMMRouter_EM, and where the approximation can fail. It maps each
mathematical object to concrete code locations so the implementation
can be audited line-by-line.

----------------------------------------------------------------------
1) Base L2D-SLDS model (independent experts)
----------------------------------------------------------------------

Latents and parameters
  - Discrete regime: z_t in {1,...,M}
  - Expert states:  alpha_{j,t} in R^d
  - Features:        phi_t = phi(x_t) in R^d
  - Dynamics (per regime k):
        alpha_{j,t} | z_t=k ~ N(A_k alpha_{j,t-1}, Q_k)
  - Observation (per expert j, regime k):
        ell_{j,t} = phi_t^T alpha_{j,t} + v_{j,t},
        v_{j,t} ~ N(0, R_{k,j})
  - Regime transition: P(z_t=k | z_{t-1}=i) = Pi[i,k]

Code mapping
  - Model definition and IMM filter:
      models/router_model.py
      * A, Q, R, Pi loaded and checked in __init__ (lines ~44-90)
      * IMM mixing + time update:
          _interaction_and_time_update_state (lines ~126-213)
      * Predictive loss mean/var:
          _predict_loss_distribution (lines ~265-309)
      * Selection rule:
          select_expert (lines ~308-356)
      * Kalman update + regime posterior:
          update_beliefs (lines ~374-478)
  - Observation source (losses vs residuals):
      router_eval._get_router_observation (router_eval.py:173-235)

Correctness (proof-sketch)
  - IMM time update uses b_{t|t-1} = b_{t-1} Pi, then moment-matches
    per-regime Gaussian mixtures, which is the standard IMM
    approximation for switching linear systems.
  - Given z_t=k, the observation model is linear Gaussian, so the
    Kalman update in update_beliefs is exact for each (k, j).
  - Regime posterior b_t(k) is updated by Bayes rule using the
    (possibly partial) observation likelihood, which is correct under
    conditional independence of experts given z_t.

Thus the base SLDS filter is mathematically correct for the stated
linear-Gaussian loss model. It is not guaranteed to be optimal for
real losses (squared errors) which are non-Gaussian and non-negative.

----------------------------------------------------------------------
2) Approximate EM for L2D-SLDS (SLDSIMMRouter_EM)
----------------------------------------------------------------------

Class
  - models/router_model_em.py

Goal
  - Estimate A_k, Q_k, R_{k,j}, and optionally Pi from data
    on an initial window t = 1..em_tk.

E-step approximation
  - Use filtered posterior b_t(k), m_{k,j,t}, P_{k,j,t} from IMM.
  - Ignore cross-time covariances:
        E[alpha_t alpha_{t-1}^T | z_t=k] approx m_t m_{t-1}^T
  - Use available experts only (full feedback required).

Sufficient statistics (accumulated over t and observed experts j)
  Let w_tk = b_t(k).
  - S_xx[k]     = sum w_tk * (P_t + m_t m_t^T)
  - S_xprev[k]  = sum w_tk * (P_{t-1} + m_{t-1} m_{t-1}^T)
  - S_x_xprev[k]= sum w_tk * (m_t m_{t-1}^T)
  - R stats (per k,j):
        sum w_tk * ( (ell_t - phi^T m_t)^2 + phi^T P_t phi )
  - Transition counts (approx):
        xi_t(i,k) = b_t(k) * b_{t-1}(i) * Pi[i,k] / b_pred(k)

M-step updates (approximate)
  - A_k = S_x_xprev[k] * (S_xprev[k] + lambda I)^(-1)
  - Q_k = (S_xx[k] - A_k S_x_xprev[k]^T - S_x_xprev[k] A_k^T
           + A_k S_xprev[k] A_k^T) / sum_w
  - R_{k,j} = sum_r / sum_w  (with floor)
  - Pi rows normalized from accumulated xi counts

Code mapping
  - Accumulation and M-step:
      models/router_model_em.py
      * _em_accumulate (approx E-step)
      * _em_run_m_step (approx M-step)
      * update_beliefs override wires EM into filter loop

Correctness (proof-sketch)
  - If we had exact smoothing expectations and cross-time covariances,
    these updates are the standard EM M-step for linear Gaussian state
    space models with switching regimes.
  - This implementation replaces smoothing moments with filtered
    moments and drops cross-time covariances, so it is an approximation.

Therefore:
  - EM "works" as a reasonable approximate maximum-likelihood update,
    but monotonic likelihood improvement is not guaranteed.
  - This is the same approximation used in router_model_corr_em.py.

----------------------------------------------------------------------
3) Conditions where it can fail
----------------------------------------------------------------------

Model mismatch
  - Observations are squared errors, not Gaussian. The filter and EM
    assume Gaussian noise on ell_{j,t}.
  - Expert losses are correlated, violating conditional independence.

EM approximation error
  - Using filtered (not smoothed) moments can bias A/Q estimates.
  - Ignoring cross-time covariances can under-estimate Q.

Practical failure modes
  - Small windows (em_tk) => ill-conditioned S_xprev => unstable A.
  - If losses are missing or NaN, R estimates can collapse without
    safeguards.

----------------------------------------------------------------------
4) How to enable in config
----------------------------------------------------------------------

Add to config:
  routers:
    slds_imm_em:
      enabled: true
      em_tk: 5000
      em_min_weight: 1.0e-6
      em_verbose: false
      em_update_pi: true
      em_update_AQ: true
      em_update_R: true
      em_lambda_A: 1.0e-6
      em_r_floor: 1.0e-8
      apply_to_partial: false

The EM update is applied to the full-feedback base router only by
default. When em_tk is set, evaluate_routers_and_baselines will use the
training split protocol (run_router_on_env_em_split).

