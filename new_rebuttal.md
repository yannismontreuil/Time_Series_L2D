We thank the reviewer for this suggestion. We added the requested non-stationary comparisons and ran them on the same tuned Jena and Melbourne settings used in the rebuttal.

There is one compatibility point to make explicit. The three cited papers are not all written for the same problem class as ours:

- 1909.09146 (D-LinUCB) is a direct non-stationary linear contextual bandit baseline, so we implemented it directly as discounted shared linear regression over expert-conditioned features.
- 1711.03539 (CUSUM-UCB) is a piecewise-stationary non-contextual MAB method. In our contextual expert-routing setting, the closest compatible analogue is a contextual LinUCB backbone with per-expert CUSUM-based local resets driven by bounded prediction-error magnitudes.
- 1908.10402 (GLR-CUCB) is a piecewise-stationary combinatorial semi-bandit method. In our setting, the closest compatible analogue is a contextual LinUCB backbone with GLR-based global restart and forced exploration, again driven by bounded prediction-error magnitudes.

So the comparison below contains one direct baseline (D-LinUCB) and two careful contextual adaptations of the change-detection ideas in the other two papers.

We ran all three on Slurm. For D-LinUCB, we also ran a small discount-factor sweep and kept the best value for each 
dataset ($\gamma=0.95$ on Jena, $\gamma=0.98$ on Melbourne).

| Method | Jena mean cost | Jena runtime (ms/step) | Melbourne mean cost | Melbourne runtime (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| L2D-SLDS | 3.6100 | 2.295 | 5.6959 | 0.745 |
| NeuralUCB | 4.1537 | 0.658 | 5.8780 | 0.426 |
| D-LinUCB (tuned) | 5.0490 | 1.011 | 5.9889 | 0.677 |
| CUSUM-LinUCB | 4.6077 | 0.350 | 6.3191 | 0.324 |
| GLR-LinUCB | 5.8584 | 0.453 | 6.2238 | 0.410 |

These additions do not change the empirical ranking on either dataset.

- On Jena, the strongest of the three is CUSUM-LinUCB, but it remains above both NeuralUCB and L2D-SLDS.
- On Melbourne, the strongest of the three is tuned D-LinUCB, but it remains above both NeuralUCB and L2D-SLDS.

We also implemented a sliding-window variant of NeuralUCB (SW-NeuralUCB), which refits the neural-linear model on only the last $W$ observations at each step:

| Window | Jena mean cost | Jena runtime (ms/step) | Melbourne mean cost | Melbourne runtime (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| W=30 | 5.3597 | 4.066 | 6.1692 | 2.100 |
| W=90 | 4.6183 | 10.217 | 6.0168 | 5.388 |
| W=180 | 4.3777 | 19.263 | 6.0670 | 10.370 |

SW-NeuralUCB is **worse than plain NeuralUCB** on both datasets (4.38 vs 4.15 on Jena; 6.02 vs 5.88 on Melbourne), while being 6--30x slower. Naively discarding old observations hurts rather than helps: the non-stationarity in these benchmarks is better handled by structured modeling (latent regimes) than by brute-force forgetting.

After adding a direct non-stationary linear baseline (D-LinUCB), two change-detection alternatives (CUSUM-LinUCB, GLR-LinUCB), and a sliding-window neural baseline (SW-NeuralUCB), the ranking on both datasets remains unchanged: L2D-SLDS is still the best-performing method.
