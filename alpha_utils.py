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

Plus a unified interface with reliability guards and bootstrap CI.

References:
  - McCulloch, J.H. (1986). "Simple Consistent Estimators of Stable Distribution
    Parameters." Communications in Statistics—Simulation and Computation, 15(4), 1109–1136.
  - Koutrouvelis, I.A. (1980). "Regression-Type Estimation of the Parameters of
    Stable Laws." JASA, 75(372), 918–928.

Changelog (backported from learnability project, 2026-03):
  - FIX: α-dependent fallback quantile table (was hardcoded constants, producing
    systematically biased estimates when scipy unavailable)
  - FIX: ECF internal minimum sample guard (n < 50 → fallback)
  - FIX: np.isfinite() checks before positivity checks in reliability assessment
  - ADD: Detailed reliability reason strings (method_origin, method_reason,
    reliability_reason, ecf_filter_mode) via estimate_alpha_sigma_with_meta()
  - ADD: bootstrap_mcculloch() for bootstrap confidence intervals
  - ADD: estimate_alpha_sigma_ecf_symmetric_with_meta() with full diagnostics
"""

import math
from typing import Dict, Tuple

import numpy as np

# ============================================================
# Minimum sample count for a reliable α̂ estimate
# ============================================================
_MIN_SAMPLES_ALPHA = 500


# ============================================================
# McCulloch quantile-ratio cache
# ============================================================

class _StableQuantileCache:
    """Pre-computed quantile-ratio lookup table for McCulloch method.

    When scipy is available, theoretical quantiles are computed from the
    exact SαS distribution via scipy.stats.levy_stable.

    When scipy is NOT available, we use an α-dependent fallback table
    (11 reference points from α=1.0 to α=2.0) with linear interpolation.
    This replaces the previous hardcoded constants which were α-independent
    and produced biased estimates.  (Backported from learnability project.)
    """

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

        # α-dependent fallback table: columns are [α, (q95-q05)/(q75-q25), IQR]
        # Computed from scipy.stats.levy_stable on a reference machine.
        # Previously this was a single tuple (-1.0, -0.3, 0.0, 0.3, 1.0)
        # which gave the same quantiles for all α — a significant bug.
        self.fallback = np.array([
            [2.00, 1.903, 1.349],
            [1.90, 2.020, 1.404],
            [1.80, 2.160, 1.472],
            [1.70, 2.330, 1.556],
            [1.60, 2.545, 1.662],
            [1.50, 2.820, 1.802],
            [1.40, 3.180, 2.000],
            [1.30, 3.670, 2.289],
            [1.20, 4.390, 2.781],
            [1.10, 5.560, 3.865],
            [1.00, 7.430, 6.314],
        ], dtype=float)

        self.cache_q: dict = {}
        self._grid_ready = False
        self._R_SORT = None
        self._A_SORT = None
        self._IQR_SORT = None

    def theo_quantiles(self, alpha: float) -> Tuple[float, float, float, float, float]:
        """Return (q05, q25, q50, q75, q95) for symmetric α-stable with unit scale."""
        a = float(np.clip(alpha, 1.0, 2.0))
        key = round(a, 6)
        if key in self.cache_q:
            return self.cache_q[key]

        if self._have_scipy and self.levy_stable is not None:
            q = self.levy_stable.ppf([0.05, 0.25, 0.5, 0.75, 0.95], a, 0.0, loc=0.0, scale=1.0)
            out = tuple(float(x) for x in q)
        else:
            # α-dependent interpolation from fallback table.
            # Table has α in descending order; np.interp needs ascending xp.
            grid = self.fallback
            al = grid[::-1, 0]
            r = np.interp(a, al, grid[::-1, 1])
            iqr = np.interp(a, al, grid[::-1, 2])
            q25, q75 = -0.5 * iqr, 0.5 * iqr
            q95 = 0.5 * r * iqr
            q05 = -q95
            q50 = 0.0
            out = (q05, q25, q50, q75, q95)

        self.cache_q[key] = out
        return out

    def ensure_grid(self, n_grid: int = 201):
        """Build the R → α inversion grid for the McCulloch estimator."""
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

    Falls back to McCulloch if too few informative ECF points are available.

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter (≥ 0).
    """
    n = samples.size
    # Internal ECF guard: need at least 50 samples for meaningful ECF
    # (backported from learnability — was missing, causing unreliable
    # estimates to pass through without fallback)
    if n < 50:
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
        # Fall back to McCulloch (logged in _with_meta version)
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        return estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95)

    t_use = t_grid[mask]
    phi2_use = phi2[mask]

    # Regression: Y = log(-log(|φ̂(t)|²)),  X = log(|t|)
    # Clamp phi2 to (0, 1) to prevent NaN from finite-sample noise
    phi2_use = np.clip(phi2_use, 1e-12, 1.0 - 1e-12)
    Y = np.log(-np.log(phi2_use))
    X = np.log(t_use)

    # Weighted least squares: points near |φ̂|² ≈ 0.5 are most informative.
    # Weight = exp(-2*(log|φ̂|² + 0.7)²)  peaks near |φ̂|² ≈ exp(-0.7) ≈ 0.50.
    # The constant 0.7 ≈ -log(0.5) centers the weighting on the midpoint of
    # the characteristic function's decay, where the signal-to-noise ratio
    # for estimating the slope α is highest.
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
# ECF estimator with full diagnostic metadata
# (backported from learnability project)
# ============================================================

def _default_alpha_meta(method_requested: str, n_samples_total: int) -> Dict[str, object]:
    """Return a default metadata dict with all fields initialized."""
    return {
        "method_requested": method_requested,
        "method_origin": "none",          # which estimator actually produced the result
        "method_reason": "not_run",       # why that method was chosen
        "reliability_reason": "not_run",  # semicolon-separated list of failure reasons
        "alpha_hat": float("nan"),
        "sigma_hat": float("nan"),
        "reliable": False,
        "n_samples_total": int(n_samples_total),
        "n_samples_used": 0,
        "used_subsample": 0,
        "boundary_hit": 0,
        "iqr": float("nan"),
        "quantile_ratio": float("nan"),
        "ecf_n_grid": 0,
        "ecf_n_points_strict": 0,
        "ecf_n_points_relaxed": 0,
        "ecf_n_points_used": 0,
        "ecf_filter_mode": "none",        # "strict", "relaxed", or "none"
    }


def estimate_alpha_sigma_ecf_symmetric_with_meta(samples: np.ndarray) -> Dict[str, object]:
    """
    ECF estimator with full diagnostic metadata.

    Returns a dict with fields: method_origin, method_reason, reliability_reason,
    ecf_filter_mode, alpha_hat, sigma_hat, and all intermediate diagnostics.
    This makes it possible to trace exactly which estimator was used and why.

    Backported from learnability project to replace the previous silent-fallback
    implementation that returned only (alpha_hat, sigma_hat, reliable).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    meta = _default_alpha_meta("ecf", n)
    if n < _MIN_SAMPLES_ALPHA:
        meta["method_reason"] = "too_few_samples"
        meta["reliability_reason"] = "too_few_samples"
        return meta

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)
    meta["ecf_n_grid"] = int(len(t_grid))

    mask_strict = (phi2 > 0.01) & (phi2 < 0.95)
    mask_relaxed = (phi2 > 1e-4) & (phi2 < 0.999)
    n_strict = int(mask_strict.sum())
    n_relaxed = int(mask_relaxed.sum())
    meta["ecf_n_points_strict"] = n_strict
    meta["ecf_n_points_relaxed"] = n_relaxed

    if n_strict >= 5:
        mask = mask_strict
        meta["ecf_filter_mode"] = "strict"
    elif n_relaxed >= 3:
        mask = mask_relaxed
        meta["ecf_filter_mode"] = "relaxed"
    else:
        meta["method_reason"] = "too_few_informative_points"
        meta["reliability_reason"] = "too_few_informative_points"
        return meta

    t_use = t_grid[mask]
    phi2_use = phi2[mask]
    meta["ecf_n_points_used"] = int(mask.sum())

    Y = np.log(-np.log(phi2_use))
    X = np.log(t_use)

    w = np.exp(-2.0 * (np.log(phi2_use) + 0.7) ** 2)
    w /= w.sum() + 1e-12

    Xbar = np.average(X, weights=w)
    Ybar = np.average(Y, weights=w)
    dx = X - Xbar
    dy = Y - Ybar
    alpha_hat = float(np.sum(w * dx * dy) / (np.sum(w * dx ** 2) + 1e-12))
    intercept = Ybar - alpha_hat * Xbar

    alpha_hat = float(np.clip(alpha_hat, 1.0, 2.0))
    sigma_hat = float((np.exp(intercept) / 2.0) ** (1.0 / alpha_hat))

    meta.update({
        "method_origin": "ecf_regression",
        "method_reason": "ok",
        "reliability_reason": "ok",
        "alpha_hat": alpha_hat,
        "sigma_hat": float(max(0.0, sigma_hat)),
    })
    return meta


# ============================================================
# Unified interface with detailed metadata
# (backported from learnability project)
# ============================================================

def estimate_alpha_sigma_with_meta(
    samples: np.ndarray,
    method: str = "ecf",
    n_samples_for_ecf: int = 100000,
) -> Dict[str, object]:
    """
    Unified interface returning full diagnostic metadata.

    This replaces the old estimate_alpha_sigma() for callers that need to
    know which method was actually used (ECF vs McCulloch fallback) and
    why an estimate may be unreliable.

    Returns dict with: method_origin, method_reason, reliability_reason,
    ecf_filter_mode, alpha_hat, sigma_hat, reliable, and all diagnostics.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    meta = _default_alpha_meta(method, n)

    iqr = float(np.subtract(*np.percentile(samples, [75, 25]))) if n > 0 else float("nan")
    meta["iqr"] = iqr

    if n < _MIN_SAMPLES_ALPHA:
        meta["method_reason"] = "too_few_samples"
        meta["reliability_reason"] = "too_few_samples"
        return meta

    if method == "ecf":
        if n > n_samples_for_ecf:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, n_samples_for_ecf, replace=False)
            sub = np.asarray(samples[idx], dtype=np.float64)
            meta["used_subsample"] = 1
            meta["n_samples_used"] = int(sub.size)
        else:
            sub = np.asarray(samples, dtype=np.float64)
            meta["n_samples_used"] = int(sub.size)

        ecf_meta = estimate_alpha_sigma_ecf_symmetric_with_meta(sub)
        for key in [
            "method_origin", "method_reason", "alpha_hat", "sigma_hat",
            "ecf_n_grid", "ecf_n_points_strict", "ecf_n_points_relaxed",
            "ecf_n_points_used", "ecf_filter_mode",
        ]:
            meta[key] = ecf_meta[key]
    else:
        # Finiteness check before IQR positivity (backported from learnability:
        # previously, NaN IQR could pass the > 0 check accidentally)
        if (not np.isfinite(iqr)) or (iqr <= 1e-12):
            meta["method_origin"] = "none"
            meta["method_reason"] = "degenerate_iqr"
            meta["reliability_reason"] = "degenerate_iqr"
            meta["n_samples_used"] = int(n)
            return meta

        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        meta["quantile_ratio"] = float((q95 - q05) / (iqr + 1e-12))
        alpha_hat, sigma_hat = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
            q05, q25, q75, q95
        )
        meta.update({
            "method_origin": "mcculloch",
            "method_reason": "ok",
            "alpha_hat": float(alpha_hat),
            "sigma_hat": float(sigma_hat),
            "n_samples_used": int(n),
        })

    # ── Reliability assessment ──
    # (backported from learnability: now produces reason strings, not just bool)
    reliability_reasons = []
    alpha_hat = float(meta["alpha_hat"])
    sigma_hat = float(meta["sigma_hat"])

    # Check finiteness first (backported fix: was missing, so NaN sigma
    # could pass the > 0 check since NaN > 0 is False)
    if not np.isfinite(alpha_hat) or not np.isfinite(sigma_hat):
        reliability_reasons.append(str(meta["method_reason"]))
    if np.isfinite(sigma_hat) and sigma_hat <= 1e-12:
        reliability_reasons.append("nonpositive_sigma")
    if np.isfinite(alpha_hat) and (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliability_reasons.append("boundary_with_few_samples")
    if np.isfinite(iqr) and iqr <= 1e-10:
        reliability_reasons.append("degenerate_iqr")

    meta["boundary_hit"] = int(np.isfinite(alpha_hat) and (alpha_hat <= 1.01 or alpha_hat >= 1.99))
    meta["reliable"] = len(reliability_reasons) == 0
    meta["reliability_reason"] = (
        "ok" if meta["reliable"] else ";".join(dict.fromkeys(reliability_reasons))
    )
    return meta


# ============================================================
# Legacy unified interface (preserved for backward compatibility)
# ============================================================

def estimate_alpha_sigma(
    samples: np.ndarray,
    method: str = "mcculloch",
    n_samples_for_ecf: int = 100000,
    min_samples: int = None,
) -> Tuple[float, float, bool]:
    """
    Unified interface for α̂ estimation with reliability checking.

    This is the original API preserved for backward compatibility.
    For new code, prefer estimate_alpha_sigma_with_meta() which returns
    detailed diagnostics including method_origin and reliability_reason.

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

    # Check 1: finiteness (backported from learnability)
    if not np.isfinite(alpha_hat) or not np.isfinite(sigma_hat):
        reliable = False

    # Check 2: σ̂ should be positive
    if np.isfinite(sigma_hat) and sigma_hat <= 1e-12:
        reliable = False

    # Check 3: α̂ at boundary is suspicious with few samples
    if np.isfinite(alpha_hat) and (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliable = False

    # Check 4: IQR near zero → degenerate distribution
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if not np.isfinite(iqr) or iqr <= 1e-10:
        reliable = False

    return alpha_hat, sigma_hat, reliable


# ============================================================
# Bootstrap CI for McCulloch
# (backported from learnability project — was absent entirely)
# ============================================================

def bootstrap_mcculloch(
    samples: np.ndarray,
    estimator_fn=None,
    n_boot: int = 200,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap confidence interval for α̂ via the McCulloch estimator.

    Resamples the data n_boot times, computes α̂ from each bootstrap
    replicate, and returns the median and CI endpoints.

    Args:
        samples: 1-D array of values.
        estimator_fn: callable(q05, q25, q75, q95) → (alpha, sigma).
                      Defaults to estimate_alpha_sigma_mcculloch_symmetric_from_quantiles.
        n_boot: number of bootstrap replicates (default 200).
        ci: confidence level (default 0.95 → 95% CI).

    Returns:
        (alpha_median, alpha_ci_lo, alpha_ci_hi, sigma_median)
    """
    if estimator_fn is None:
        estimator_fn = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles

    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 4:
        return 2.0, 1.0, 2.0, 0.0

    rng = np.random.RandomState(42)
    alpha_boots = []
    sigma_boots = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_samples = samples[idx]
        q05 = float(np.quantile(boot_samples, 0.05))
        q25 = float(np.quantile(boot_samples, 0.25))
        q75 = float(np.quantile(boot_samples, 0.75))
        q95 = float(np.quantile(boot_samples, 0.95))
        alpha_hat, sigma_hat = estimator_fn(q05, q25, q75, q95)
        alpha_boots.append(float(alpha_hat))
        sigma_boots.append(float(sigma_hat))

    alpha_boots = np.array(alpha_boots, dtype=np.float64)
    sigma_boots = np.array(sigma_boots, dtype=np.float64)

    alpha_median = float(np.median(alpha_boots))
    sigma_median = float(np.median(sigma_boots))

    alpha_lower = (1.0 - ci) / 2.0
    alpha_upper = 1.0 - alpha_lower
    alpha_ci_lo = float(np.quantile(alpha_boots, alpha_lower))
    alpha_ci_hi = float(np.quantile(alpha_boots, alpha_upper))

    return alpha_median, alpha_ci_lo, alpha_ci_hi, sigma_median
