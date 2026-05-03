#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation for Anti-Collapse experiments.
================================================

Synthetic long-memory regression task from the companion learnability paper:
  y_t = Σ_k c_k u^T x_{t-ℓ_k} + ε_t

Default parameters match those in the learnability paper (Section 5.1):
  T=1024, D=16, lags={32,64,128,192,256}, coeffs={0.6,0.5,0.4,0.32,0.26}, σ_noise=0.3
"""

from typing import List, Optional, Tuple

import numpy as np
import torch


def sample_heavy_tailed_lags(
    K: int,
    lag_min: int,
    lag_max: int,
    alpha_task: float,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Draw K target lags from a truncated Pareto distribution on [lag_min, lag_max]
    with tail index alpha_task > 0. Smaller alpha_task => heavier tail (more mass
    near lag_max). Returns a sorted list of unique integer lags.

    Sampling uses inverse-CDF of the truncated Pareto density
        p(l) ∝ l^{-(alpha_task + 1)}, l ∈ [lag_min, lag_max].

    If unique sampling cannot fill K slots (e.g. K close to lag_max-lag_min),
    the remaining slots are filled by a monotone integer grid on [lag_min, lag_max].
    """
    if rng is None:
        rng = np.random.default_rng()
    assert lag_min >= 1 and lag_max > lag_min and K >= 1
    assert alpha_task > 0.0

    a = float(lag_min)
    b = float(lag_max)
    alpha = float(alpha_task)

    # Inverse CDF of truncated Pareto with density ∝ l^{-(alpha+1)} on [a, b]:
    # F(l) = (a^{-alpha} - l^{-alpha}) / (a^{-alpha} - b^{-alpha})
    # => l = (a^{-alpha} - U * (a^{-alpha} - b^{-alpha}))^{-1/alpha}
    lags_set = set()
    max_tries = 50 * K
    tries = 0
    while len(lags_set) < K and tries < max_tries:
        U = rng.uniform(size=max(1, K - len(lags_set)))
        inv = a ** (-alpha) - U * (a ** (-alpha) - b ** (-alpha))
        samples = inv ** (-1.0 / alpha)
        for s in samples:
            l = int(round(float(s)))
            l = max(lag_min, min(lag_max, l))
            lags_set.add(l)
            if len(lags_set) >= K:
                break
        tries += 1

    if len(lags_set) < K:
        fill = np.linspace(lag_min, lag_max, K, dtype=int).tolist()
        for l in fill:
            lags_set.add(int(l))
            if len(lags_set) >= K:
                break

    return sorted(lags_set)[:K]


def build_task_coeffs(
    K: int,
    coeff_base: float = 0.6,
    coeff_decay: float = 0.85,
) -> List[float]:
    """
    Geometrically decaying coefficient schedule c_k = coeff_base * coeff_decay^{k}
    for k=0,...,K-1. This matches the spirit of the fixed-lag default
    {0.6, 0.5, 0.4, 0.32, 0.26} (ratios ~0.83).
    """
    return [float(coeff_base * (coeff_decay ** k)) for k in range(K)]


def make_dataset_cpu(
    Nseq: int,
    T: int,
    D: int,
    task_lags: List[int],
    task_coeffs: List[float],
    noise_std: float,
    u_vec: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate synthetic regression dataset on CPU.

    Args:
        Nseq: number of sequences
        T: sequence length
        D: input dimension
        task_lags: list of target lags ℓ_k
        task_coeffs: list of mixing coefficients c_k
        noise_std: standard deviation of additive Gaussian noise
        u_vec: optional fixed projection direction (D,). If None, random unit vector.

    Returns:
        X: (Nseq, T, D)  input sequences (float32 CPU tensor)
        Y: (Nseq, T, 1)  target sequences (float32 CPU tensor)
        u: (D,) projection direction used
    """
    if u_vec is None:
        u = np.random.randn(D).astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        u = u_vec.astype(np.float32)

    X = np.random.randn(Nseq, T, D).astype(np.float32)
    Y = np.zeros((Nseq, T, 1), dtype=np.float32)

    for k, lag in enumerate(task_lags):
        c = float(task_coeffs[k])
        if lag < T:
            proj = np.einsum("ntd,d->nt", X[:, :T - lag, :], u)
            Y[:, lag:, 0] += c * proj

    Y += noise_std * np.random.randn(Nseq, T, 1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y), u
