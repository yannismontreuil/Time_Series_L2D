import numpy as np


def phi(x):
    """Simple feature map: identity on R^d."""
    return np.asarray(x, dtype=float)


# Basic SLDS/IMM toy configuration used in the examples.
M = 2  # regimes
N = 3  # experts
d = 2  # state / feature dimension

A = np.stack(
    [
        np.eye(d),         # regime 0 dynamics
        0.9 * np.eye(d),   # regime 1 dynamics
    ]
)
Q = np.stack(
    [
        0.01 * np.eye(d),
        0.05 * np.eye(d),
    ]
)
R = 0.1 * np.ones((M, N))  # observation noise variances R_{k,j}
Pi = np.array(
    [
        [0.95, 0.05],
        [0.10, 0.90],
    ]
)  # regime transition matrix
beta = np.array([0.0, 0.1, 0.2])  # consultation costs for each expert

# Shared RNG for reproducible examples.
rng = np.random.default_rng(0)

