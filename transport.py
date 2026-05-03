#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transport factor computation for Anti-Collapse experiments.
=============================================================

Implements the first-order diagonal expansion of the Jacobian product
and model-specific mu_tl (effective learning rate proxy) computations.

The key objects:
  - precompute_prefix_sums: O(T) prefix sums of log(leak) and rdiag/leak
  - mu_diag_product_first_order: first-order corrected diagonal product (mu0 + mu1)
  - prod_from_prefix: pure diagonal product (no first-order correction)
  - compute_mu_tl_for_lag: model-specific mu_tl for envelope computation
  - compute_mu_tl_for_matched_stat: model-specific mu_tl for matched-stat computation

Notation (from the companion papers):
  leak[t]  = zeroth-order diagonal factor (carry/forget factor)
  rdiag[t] = first-order correction term
  mu0      = Π_p leak[p]  (product over window)
  mu1      = mu0 * Σ_p (rdiag[p] / leak[p])  (first-order correction)
  mu_tl    = mu0 + mu1  (first-order expansion of diagonal Jacobian product)

For GRU:  mu_tl = gamma + rho0 + eta0
  gamma = first-order product of leak=(1-z)
  rho0  = product of r (reset gate)
  eta0  = product of (1-z)*r

For LSTM: mu_tl = base * e_end
  base  = first-order product of leak=f (forget gate)
  e_end = o * (1 - tanh(c)^2)  at the end of the window

Window semantics (backported from learnability project, 2026-03):
  - Envelope kernel: *unshifted* window [0, ℓ), guard ℓ > T.
    Used for f̂(ℓ) = mean |μ(ℓ)| over (batch, time, units).
  - Matched-stat kernel: *shifted* window [t-ℓ+1, t], guard ℓ ≥ T.
    Used for ψ_t(ℓ) = Σ_q μ δ v connecting h_t back to h_{t-ℓ}.
  The two kernels differ in their prefix-sum index offsets.

Decomposition (backported from learnability project, 2026-03):
  All functions now return (mu0, mu1, mu_tl) instead of just mu_tl,
  exposing the zero-order and first-order contributions separately.
  This enables the shape-correction diagnostic R(ℓ) = f(ℓ)/f_gates(ℓ).
"""

import torch
from typing import Tuple


# ============================================================
# Prefix sum computations
# ============================================================

@torch.no_grad()
def precompute_prefix_sums(
    leak: torch.Tensor,
    rdiag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prefix sums for efficient sliding-window diagonal products.

    Args:
        leak:  (B, T, H) — positive diagonal factors (clipped to [1e-12, 1.0])
        rdiag: (B, T, H) — first-order correction terms

    Returns:
        cs_log:   (B, T+1, H) — cumulative sum of log(leak), with cs_log[:,0,:] = 0
        cs_ratio: (B, T+1, H) — cumulative sum of rdiag/leak, with cs_ratio[:,0,:] = 0
    """
    B, T, H = leak.shape
    device = leak.device

    leak64 = torch.clamp(leak.double(), 1e-12, 1.0)
    log_leak = torch.log(leak64)
    cs_log = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
    cs_log[:, 1:, :] = torch.cumsum(log_leak, dim=1)

    ratio = (rdiag.double() / leak64).to(torch.float64)
    cs_ratio = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
    cs_ratio[:, 1:, :] = torch.cumsum(ratio, dim=1)
    return cs_log, cs_ratio


# ============================================================
# Diagonal product operators — envelope (unshifted) window
# ============================================================

@torch.no_grad()
def mu_diag_product_first_order(
    cs_log: torch.Tensor,
    cs_ratio: torch.Tensor,
    leak: torch.Tensor,
    rdiag: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    First-order corrected diagonal product over all valid *unshifted*
    windows of length ell (envelope semantics).

    mu0[b,t,q] = Π_{p=t}^{t+ell-1} leak[b,p,q]
    mu1[b,t,q] = mu0 * Σ_{p=t}^{t+ell-1} (rdiag[b,p,q] / leak[b,p,q])
    mu_tl      = mu0 + mu1

    Returns:
        (mu0, mu1, mu_tl) each of shape (B, T-ell+1, H)

    Note: previously returned only mu_tl. Now returns the decomposition
    to enable shape-correction diagnostics (backported from learnability).
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        z = torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
        return z, z, z

    if ell == 1:
        mu0 = leak.to(out_dtype)
        mu1 = rdiag.to(out_dtype)
        return mu0, mu1, mu0 + mu1

    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]

    # Compute mu0 and mu1 entirely in float64 before casting to prevent precision loss
    mu0_64 = torch.exp(log_prod)
    sum_ratio_window = cs_ratio[:, ell:(T + 1), :] - cs_ratio[:, 0:(T - ell + 1), :]
    mu1_64 = mu0_64 * sum_ratio_window

    mu0 = mu0_64.to(out_dtype)
    mu1 = mu1_64.to(out_dtype)
    return mu0, mu1, (mu0 + mu1)


# ============================================================
# Diagonal product operators — matched-stat (shifted) window
# (backported from learnability project, 2026-03)
# ============================================================

@torch.no_grad()
def mu_diag_product_first_order_matched(
    cs_log: torch.Tensor,
    cs_ratio: torch.Tensor,
    leak: torch.Tensor,
    rdiag: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    First-order corrected diagonal product over all valid *shifted*
    windows of length ell (matched-statistic semantics).

    Uses a shifted window: product from step (t-ℓ+1) to step t, aligned
    so that mu[b, t, q] corresponds to the kernel connecting h_t back to
    h_{t-ℓ}.  This is the kernel used inside ψ_t(ℓ) = Σ_q μ δ v.

    Returns:
        (mu0, mu1, mu_tl) each of shape (B, T-ℓ, H)

    Backported from learnability project.  The anticollapse codebase
    previously had only the envelope (unshifted) kernel.
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell >= T:
        z = torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
        return z, z, z

    log_prod = cs_log[:, (ell + 1):(T + 1), :] - cs_log[:, 1:(T - ell + 1), :]
    mu0_64 = torch.exp(log_prod)

    sum_ratio = cs_ratio[:, (ell + 1):(T + 1), :] - cs_ratio[:, 1:(T - ell + 1), :]
    mu1_64 = mu0_64 * sum_ratio

    mu0 = mu0_64.to(out_dtype)
    mu1 = mu1_64.to(out_dtype)
    return mu0, mu1, (mu0 + mu1)


# ============================================================
# Pure diagonal product (no first-order correction)
# ============================================================

@torch.no_grad()
def prod_from_prefix(
    cs_log: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Pure diagonal product Π factor over window length ell (no first-order correction).
    Uses envelope (unshifted) window semantics.

    Args:
        cs_log: (B, T+1, H) — prefix sums of log(factor)
        ell: window length
        out_dtype: output dtype

    Returns:
        (B, T-ell+1, H)
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
    if ell == 1:
        return torch.exp(cs_log[:, 1:, :] - cs_log[:, :-1, :]).to(out_dtype)
    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]
    return torch.exp(log_prod).to(out_dtype)


@torch.no_grad()
def prod_from_prefix_matched(
    cs_log: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Pure diagonal product Π factor with matched-stat (shifted) window semantics.

    Returns:
        (B, T-ell, H)

    Backported from learnability project.
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell >= T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
    log_prod = cs_log[:, (ell + 1):(T + 1), :] - cs_log[:, 1:(T - ell + 1), :]
    return torch.exp(log_prod).to(out_dtype)


# ============================================================
# Model-specific mu_tl computation — envelope (unshifted)
# ============================================================

@torch.no_grad()
def compute_mu_tl_for_lag(
    model_name: str,
    intermediates: dict,
    ell: int,
    out_dtype: torch.dtype,
    _prefix_cache: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mu_tl for a specific lag using envelope (unshifted) window semantics.

    Args:
        model_name: one of {const, shared, diag, gru, lstm}
        intermediates: dict from model.forward_with_intermediates()
        ell: lag value
        out_dtype: output tensor dtype
        _prefix_cache: dict to cache prefix sums across lags (mutated in-place).
                       Pass {} on first call; subsequent calls reuse cached values.

    Returns:
        (mu0, mu1, mu_tl) each of shape (B, T-ell+1, H)

    Note: previously returned only mu_tl. Now returns (mu0, mu1, mu_tl)
    to expose the zero-order/first-order decomposition for diagnostics.
    Callers that only need mu_tl can use mu_tl = result[2] or
    result[-1] for backward compatibility.
    """
    if _prefix_cache is None:
        _prefix_cache = {}

    model_name = model_name.lower().strip()

    leak = intermediates["leak"]
    rdiag = intermediates["rdiag"]

    # Ensure base prefix sums are cached
    if "cs_log" not in _prefix_cache:
        cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)
        _prefix_cache["cs_log"] = cs_log
        _prefix_cache["cs_ratio"] = cs_ratio
    cs_log = _prefix_cache["cs_log"]
    cs_ratio = _prefix_cache["cs_ratio"]

    if model_name in ("const", "shared", "diag", "diaggate", "multigate"):
        # Standard first-order diagonal product
        return mu_diag_product_first_order(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )

    elif model_name == "gru":
        # GRU: mu_tl = gamma + rho0 + eta0
        gamma0, gamma1, gamma = mu_diag_product_first_order(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )

        # Reset gate product
        if "cs_log_r" not in _prefix_cache:
            r = torch.clamp(intermediates["r"], 1e-12, 1.0)
            cs_log_r, _ = precompute_prefix_sums(r, torch.zeros_like(r))
            _prefix_cache["cs_log_r"] = cs_log_r
        rho0 = prod_from_prefix(_prefix_cache["cs_log_r"], ell, out_dtype)

        # Combined (1-z)*r product
        if "cs_log_eta" not in _prefix_cache:
            r = torch.clamp(intermediates["r"], 1e-12, 1.0)
            eta = torch.clamp(leak * r, 1e-12, 1.0)
            cs_log_eta, _ = precompute_prefix_sums(eta, torch.zeros_like(eta))
            _prefix_cache["cs_log_eta"] = cs_log_eta
        eta0 = prod_from_prefix(_prefix_cache["cs_log_eta"], ell, out_dtype)

        # Decomposition: mu0 includes all zero-order terms, mu1 is the
        # first-order correction from the leak product only (gamma1)
        mu0 = gamma0 + rho0 + eta0
        mu1 = gamma1
        return mu0, mu1, (mu0 + mu1)

    elif model_name == "lstm":
        # LSTM: mu_tl = base * e_end
        base0, base1, base = mu_diag_product_first_order(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )
        e = intermediates["e"]
        e_end = e[:, (ell - 1):, :]  # aligns with T-ell+1 valid start times

        mu0 = base0 * e_end
        mu1 = base1 * e_end
        return mu0, mu1, (mu0 + mu1)

    else:
        raise ValueError(f"Unknown model for transport: {model_name!r}")


# ============================================================
# Model-specific mu_tl computation — matched-stat (shifted)
# (backported from learnability project, 2026-03)
# ============================================================

@torch.no_grad()
def compute_mu_tl_for_matched_stat(
    model_name: str,
    intermediates: dict,
    ell: int,
    out_dtype: torch.dtype,
    _prefix_cache: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mu_tl for a specific lag using matched-stat (shifted) window semantics.

    Uses a shifted window: product from step (t-ℓ+1) to step t, aligned
    so that mu[b, t, q] connects h_t back to h_{t-ℓ}.

    Args:
        model_name: one of {const, shared, diag, gru, lstm}
        intermediates: dict from model.forward_with_intermediates()
        ell: lag value
        out_dtype: output tensor dtype
        _prefix_cache: dict to cache prefix sums across lags

    Returns:
        (mu0, mu1, mu_tl) each of shape (B, T-ℓ, H)

    Backported from learnability project.  The anticollapse codebase
    previously had only compute_mu_tl_for_lag (envelope semantics).
    """
    if _prefix_cache is None:
        _prefix_cache = {}

    model_name = model_name.lower().strip()

    leak = intermediates["leak"]
    rdiag = intermediates["rdiag"]

    if "cs_log" not in _prefix_cache:
        cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)
        _prefix_cache["cs_log"] = cs_log
        _prefix_cache["cs_ratio"] = cs_ratio
    cs_log = _prefix_cache["cs_log"]
    cs_ratio = _prefix_cache["cs_ratio"]

    if model_name in ("const", "shared", "diag", "diaggate", "multigate"):
        return mu_diag_product_first_order_matched(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )

    elif model_name == "gru":
        gamma0, gamma1, gamma = mu_diag_product_first_order_matched(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )

        if "cs_log_r" not in _prefix_cache:
            r = torch.clamp(intermediates["r"], 1e-12, 1.0)
            cs_log_r, _ = precompute_prefix_sums(r, torch.zeros_like(r))
            _prefix_cache["cs_log_r"] = cs_log_r
        rho0 = prod_from_prefix_matched(_prefix_cache["cs_log_r"], ell, out_dtype)

        if "cs_log_eta" not in _prefix_cache:
            r = torch.clamp(intermediates["r"], 1e-12, 1.0)
            eta = torch.clamp(leak * r, 1e-12, 1.0)
            cs_log_eta, _ = precompute_prefix_sums(eta, torch.zeros_like(eta))
            _prefix_cache["cs_log_eta"] = cs_log_eta
        eta0 = prod_from_prefix_matched(_prefix_cache["cs_log_eta"], ell, out_dtype)

        mu0 = gamma0 + rho0 + eta0
        mu1 = gamma1
        return mu0, mu1, (mu0 + mu1)

    elif model_name == "lstm":
        B, T, H = intermediates["e"].shape
        if ell <= 0 or ell >= T:
            z = torch.zeros(B, 0, H, dtype=out_dtype, device=leak.device)
            return z, z, z

        base0, base1, base = mu_diag_product_first_order_matched(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )
        e = intermediates["e"]
        # Shifted window: e at position t (indices ell..T-1)
        e_end = e[:, ell:T, :].to(out_dtype)

        mu0 = base0 * e_end
        mu1 = base1 * e_end
        return mu0, mu1, (mu0 + mu1)

    else:
        raise ValueError(f"Unknown model for transport: {model_name!r}")
