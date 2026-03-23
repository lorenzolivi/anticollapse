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
