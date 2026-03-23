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
  - compute_mu_tl_batch: model-specific mu_tl for a batch, given intermediates

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
# Diagonal product operators
# ============================================================

@torch.no_grad()
def mu_diag_product_first_order(
    cs_log: torch.Tensor,
    cs_ratio: torch.Tensor,
    leak: torch.Tensor,
    rdiag: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> torch.Tensor:
    """
    First-order corrected diagonal product over all valid windows of length ell.

    mu_tl[b, t, q] = mu0[b,t,q] + mu1[b,t,q]
    where:
      mu0 = Π_{p=t}^{t+ell-1} leak[b,p,q]
      mu1 = mu0 * Σ_{p=t}^{t+ell-1} (rdiag[b,p,q] / leak[b,p,q])

    Args:
        cs_log, cs_ratio: prefix sums from precompute_prefix_sums
        leak, rdiag: (B, T, H)
        ell: window length
        out_dtype: output dtype (usually float32)

    Returns:
        (B, T-ell+1, H)  first-order diagonal product
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)

    if ell == 1:
        return (leak + rdiag).to(out_dtype)

    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]

    # Compute mu0 and mu1 entirely in float64 before casting to prevent precision loss
    mu0_64 = torch.exp(log_prod)
    sum_ratio_window = cs_ratio[:, ell:(T + 1), :] - cs_ratio[:, 0:(T - ell + 1), :]
    mu1_64 = mu0_64 * sum_ratio_window

    return (mu0_64 + mu1_64).to(out_dtype)


@torch.no_grad()
def prod_from_prefix(
    cs_log: torch.Tensor,
    ell: int,
    out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Pure diagonal product Π factor over window length ell (no first-order correction).

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


# ============================================================
# Model-specific mu_tl computation
# ============================================================

@torch.no_grad()
def compute_mu_tl_for_lag(
    model_name: str,
    intermediates: dict,
    ell: int,
    out_dtype: torch.dtype,
    _prefix_cache: dict = None,
) -> torch.Tensor:
    """
    Compute mu_tl for a specific lag, dispatching on model type.

    Args:
        model_name: one of {const, shared, diag, gru, lstm}
        intermediates: dict from model.forward_with_intermediates()
        ell: lag value
        out_dtype: output tensor dtype
        _prefix_cache: dict to cache prefix sums across lags (mutated in-place).
                       Pass {} on first call; subsequent calls reuse cached values.

    Returns:
        mu_tl: (B, T-ell+1, H)
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
        gamma = mu_diag_product_first_order(
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

        return gamma + rho0 + eta0

    elif model_name == "lstm":
        # LSTM: mu_tl = base * e_end
        base = mu_diag_product_first_order(
            cs_log, cs_ratio, leak, rdiag, ell, out_dtype
        )
        e = intermediates["e"]
        e_end = e[:, (ell - 1):, :]  # aligns with T-ell+1 valid start times
        return base * e_end

    else:
        raise ValueError(f"Unknown model for transport: {model_name!r}")
