#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alpha-stable parameter estimation utilities
=============================================

Shared module for estimating the tail index (α) and scale (σ) of symmetric
α-stable distributions from 1-D samples.

Two estimators:
  1. McCulloch (1986) quantile-ratio method
  2. Koutrouvelis (1980) ECF regression method

Plus a unified interface with reliability guards.

References:
  - McCulloch, J.H. (1986). "Simple Consistent Estimators of Stable Distribution
    Parameters." Communications in Statistics—Simulation and Computation, 15(4), 1109–1136.
  - Koutrouvelis, I.A. (1980). "Regression-Type Estimation of the Parameters of
    Stable Laws." JASA, 75(372), 918–928.
"""

import math
from typing import Tuple

import numpy as np

# ============================================================
# Minimum sample count for a reliable α̂ estimate
# ============================================================
_MIN_SAMPLES_ALPHA = 500

# ============================================================
# McCulloch quantile-ratio cache
# ============================================================

class _StableQuantileCache:
    """Pre-computed quantile-ratio lookup table for McCulloch method."""

    def __init__(self):
        self._have_scipy = False
        self.levy_stable = None
        try:
            from scipy.stats import levy_stable  # type: ignore
            self._have_scipy = True
            self.levy_stable = levy_stable
        except Exception:
            self._have_scipy = False
            self.levy_stable = None

        self.cache_q = {}
        self._grid_ready = False
        self._R_SORT = None
        self._A_SORT = None
        self._IQR_SORT = None

    def theo_quantiles(self, alpha: float):
        a = float(np.clip(alpha, 1.0, 2.0))
        key = round(a, 6)
        if key in self.cache_q:
            return self.cache_q[key]
        if self._have_scipy and self.levy_stable is not None:
            q = self.levy_stable.ppf([0.05, 0.25, 0.5, 0.75, 0.95], a, 0.0, loc=0.0, scale=1.0)
            out = tuple(float(x) for x in q)
        else:
            out = (-1.0, -0.3, 0.0, 0.3, 1.0)
        self.cache_q[key] = out
        return out

    def ensure_grid(self, n_grid: int = 201):
        if self._grid_ready:
            return
        alpha_grid = np.linspace(1.0, 2.0, int(n_grid))
        r_grid = np.empty_like(alpha_grid)
        iqr_grid = np.empty_like(alpha_grid)
        for i, a in enumerate(alpha_grid):
            q05, q25, _, q75, q95 = self.theo_quantiles(float(a))
            denom = (q75 - q25) + 1e-12
            r_grid[i] = (q95 - q05) / denom
            iqr_grid[i] = (q75 - q25)
        order = np.argsort(r_grid)
        self._R_SORT = r_grid[order]
        self._A_SORT = alpha_grid[order]
        self._IQR_SORT = iqr_grid[order]
        self._grid_ready = True


_STABLE_CACHE = _StableQuantileCache()
_STABLE_CACHE.ensure_grid(201)


# ============================================================
# McCulloch estimator
# ============================================================

def estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
    q05: float, q25: float, q75: float, q95: float
) -> Tuple[float, float]:
    """
    McCulloch quantile-ratio method for symmetric α-stable.
    Takes pre-computed quantiles (5th, 25th, 75th, 95th percentiles).

    Returns:
        (alpha_hat, sigma_hat) both in [1,2] × [0,∞)
    """
    iqr = float(q75 - q25)
    if (not np.isfinite(iqr)) or (iqr <= 1e-12):
        return 2.0, 0.0

    r_hat = float((q95 - q05) / (iqr + 1e-12))

    R = _STABLE_CACHE._R_SORT
    A = _STABLE_CACHE._A_SORT
    IQR = _STABLE_CACHE._IQR_SORT
    assert R is not None and A is not None and IQR is not None

    r_hat_clamped = float(np.clip(r_hat, float(np.min(R)), float(np.max(R))))
    alpha_hat = float(np.interp(r_hat_clamped, R, A))
    iqr_theory = float(np.interp(r_hat_clamped, R, IQR))
    sigma_hat = float(iqr / (iqr_theory + 1e-12))
    return float(np.clip(alpha_hat, 1.0, 2.0)), float(max(0.0, sigma_hat))


def estimate_alpha_sigma_mcculloch_symmetric_from_samples(
    samples: np.ndarray
) -> Tuple[float, float]:
    """
    McCulloch quantile-ratio method for symmetric α-stable.
    Convenience wrapper that computes quantiles from samples.

    Returns:
        (alpha_hat, sigma_hat) both in [1,2] × [0,∞)
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size < 32:
        return 2.0, 0.0
    q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
    return estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95)


# ============================================================
# ECF (Empirical Characteristic Function) estimator
# — Koutrouvelis (1980) regression for symmetric α-stable
#
# For the symmetric stable (β=0, μ=0) case, the CF is:
#   φ(t) = exp(-σ^α |t|^α)
#
# Taking logs:
#   log(-log|φ̂(t)|²) = log(2σ^α) + α·log|t|
#
# This is a simple linear regression Y = c + α·X where
#   Y_k = log(-log|φ̂(t_k)|²), X_k = log|t_k|
# and the slope directly gives α̂.
#
# The grid of t-values is chosen in the "informative region"
# to avoid:
#   - t ≈ 0  where φ ≈ 1 and log(−log(·)) is numerically unstable
#   - t >> 1 where φ ≈ 0 and |φ̂|² is dominated by sampling noise
# ============================================================

def _ecf_at_t(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Compute |φ̂(t)|² for each t in t_grid from real-valued samples.

    For real symmetric distributions:
        φ̂(t) = (1/n) Σ_j exp(i·t·x_j)
        |φ̂(t)|² = [(1/n)Σ cos(t·x)]² + [(1/n)Σ sin(t·x)]²

    Uses chunked computation to avoid O(n_samples × n_grid) memory.

    Returns: 1-D array of |φ̂(t)|² values, shape (len(t_grid),).
    """
    n = samples.size
    if n == 0:
        return np.zeros_like(t_grid)

    chunk_size = min(n, 50000)  # keep memory bounded
    total_cos = np.zeros(len(t_grid), dtype=np.float64)
    total_sin = np.zeros(len(t_grid), dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = samples[start:end]
        # outer product: (len(t_grid), chunk_size)
        tx = np.outer(t_grid, x_chunk)
        total_cos += np.cos(tx).sum(axis=1)
        total_sin += np.sin(tx).sum(axis=1)

    phi2 = (total_cos / n) ** 2 + (total_sin / n) ** 2
    return phi2


def _choose_ecf_grid(samples: np.ndarray, n_points: int = 50) -> np.ndarray:
    """
    Choose a grid of t-values in the informative region for ECF regression.

    Strategy: t should be in a range where |φ(t)|² is between ~0.01 and ~0.95.
    For a symmetric stable with scale σ:
        |φ(t)|² = exp(-2σ^α |t|^α)
    So |φ|² ≈ 0.95 when t ≈ (0.025/σ^α)^{1/α}
    and |φ|² ≈ 0.01 when t ≈ (2.3/σ^α)^{1/α}

    We use the IQR as a robust scale estimate to set the range.
    """
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-12:
        iqr = float(np.std(samples)) * 1.349  # Gaussian IQR from std
    if iqr <= 1e-12:
        return np.linspace(0.1, 2.0, n_points)

    # Rough scale: for Gaussian, IQR ≈ 1.349σ, so σ ≈ IQR/1.349
    scale_est = iqr / 1.349
    # t range: from ~0.05/scale to ~3/scale (covers the informative region)
    t_lo = 0.05 / scale_est
    t_hi = 3.0 / scale_est
    return np.linspace(t_lo, t_hi, n_points)


def estimate_alpha_sigma_ecf_symmetric(
    samples: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate (α̂, σ̂) for a symmetric α-stable distribution using the ECF
    regression method (Koutrouvelis 1980, simplified for β=0).

    For SαS: log(-log|φ̂(t)|²) = log(2σ^α) + α·log|t|
    The slope of the regression gives α̂; the intercept gives σ̂.

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter (≥ 0).
    """
    n = samples.size
    if n < 50:  # absolute minimum for meaningful ECF regression
        return 2.0, 0.0

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)

    # Filter: keep only points where |φ̂|² is in a usable range
    # Too close to 1 → log(-log(·)) is unstable; too close to 0 → noise-dominated
    mask = (phi2 > 0.01) & (phi2 < 0.95)
    if mask.sum() < 5:
        # Relax bounds
        mask = (phi2 > 1e-4) & (phi2 < 0.999)
    if mask.sum() < 3:
        # Fall back to McCulloch
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        return estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95)

    t_use = t_grid[mask]
    phi2_use = phi2[mask]

    # Regression: Y = log(-log(|φ̂(t)|²)),  X = log(|t|)
    # Clamp phi2 to (0, 1) to prevent NaN from finite-sample noise
    phi2_use = np.clip(phi2_use, 1e-12, 1.0 - 1e-12)
    Y = np.log(-np.log(phi2_use))
    X = np.log(t_use)

    # Weighted least squares: points near |φ̂|² ≈ 0.5 are most informative
    # Weight = exp(-2*(log|φ̂|² + 0.7)²)  peaks near |φ̂|² ≈ 0.5
    w = np.exp(-2.0 * (np.log(phi2_use) + 0.7) ** 2)
    w /= w.sum() + 1e-12

    # WLS: α̂ = Σw·(X-X̄)(Y-Ȳ) / Σw·(X-X̄)²
    Xbar = np.average(X, weights=w)
    Ybar = np.average(Y, weights=w)
    dx = X - Xbar
    dy = Y - Ybar
    alpha_hat = float(np.sum(w * dx * dy) / (np.sum(w * dx ** 2) + 1e-12))
    intercept = Ybar - alpha_hat * Xbar

    # σ̂ from intercept: log(2σ^α) = intercept → σ = (exp(intercept)/2)^{1/α}
    alpha_hat = float(np.clip(alpha_hat, 1.0, 2.0))
    if alpha_hat > 0:
        sigma_hat = float((np.exp(intercept) / 2.0) ** (1.0 / alpha_hat))
    else:
        sigma_hat = 0.0

    return alpha_hat, float(max(0.0, sigma_hat))


# ============================================================
# Unified interface with reliability guards
# ============================================================

def estimate_alpha_sigma(
    samples: np.ndarray,
    method: str = "mcculloch",
    n_samples_for_ecf: int = 100000,
    min_samples: int = None,
) -> Tuple[float, float, bool]:
    """
    Unified interface for α̂ estimation with reliability checking.

    Args:
        samples: 1-D array of gradient projection values (float64).
        method: "mcculloch" or "ecf".
        n_samples_for_ecf: subsample limit for ECF (controls speed).
        min_samples: minimum samples for reliable estimate (default: _MIN_SAMPLES_ALPHA=500).

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter.
        reliable: True if the estimate passes quality checks.
    """
    n = samples.size
    _min_n = min_samples if min_samples is not None else _MIN_SAMPLES_ALPHA

    # ── reliability check: too few samples ──
    if n < _min_n:
        return 2.0, 0.0, False

    # ── compute estimate ──
    if method == "ecf":
        # Subsample if very large (ECF is O(n·K))
        if n > n_samples_for_ecf:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, n_samples_for_ecf, replace=False)
            sub = np.asarray(samples[idx], dtype=np.float64)
        else:
            sub = np.asarray(samples, dtype=np.float64)
        alpha_hat, sigma_hat = estimate_alpha_sigma_ecf_symmetric(sub)
    else:
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        alpha_hat, sigma_hat = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
            q05, q25, q75, q95
        )

    # ── reliability checks ──
    reliable = True

    # Check 1: σ̂ should be positive
    if sigma_hat <= 1e-12:
        reliable = False

    # Check 2: α̂ at boundary is suspicious with few samples
    if (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliable = False

    # Check 3: IQR near zero → degenerate distribution
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-10:
        reliable = False

    return alpha_hat, sigma_hat, reliable
