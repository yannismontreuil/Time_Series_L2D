# Settings Documentation Description

| Setting  | Data | Regime Switching | Observation Noise | expert availability |
|----------|------|------------------|-------------------|---------------------|
| `config.yaml` | synthetic data | 0 → 1 → 2 → … → M-1 → 0 | scale: 0.6 | expert 4 only appears in early intervals, expert 1 is periodically unavailable |

Selection distribution (oracle):
  expert 0: count=552, freq=0.276
  expert 1: count=634, freq=0.317
  expert 2: count=154, freq=0.077
  expert 3: count=522, freq=0.261
  expert 4: count=137, freq=0.069

Selection distribution (partial_corr_rec):
  expert 0: count=592, freq=0.296
  expert 1: count=233, freq=0.117
  expert 3: count=826, freq=0.413
  expert 4: count=348, freq=0.174

Always using expert 0:       5.2546
Always using expert 1:       1.0302
Always using expert 2:       3.0568
Always using expert 3:       1.1273
Always using expert 4:       0.8423

Router (partial feedback):      1.0365
Router (full feedback):         0.9511
Router r-SLDS (partial fb):    1.2192
Router r-SLDS (full fb):       0.9500
Router Corr (partial feedback): 1.0763
Router Corr (full feedback):    0.9440
Router Corr+Rec (partial fb): 1.0210
Router Corr+Rec (full fb):    0.9451

L2D baseline:                  0.9484
L2D_SW baseline:               0.9615
LinUCB (partial feedback):     1.0586
LinUCB (full feedback):        0.9994
NeuralUCB (partial feedback):  1.0494
NeuralUCB (full feedback):     0.9720
Random baseline:               2.5837
Oracle baseline:               0.8265