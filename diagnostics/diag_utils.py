#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for anticollapse diagnostic experiments.
=========================================================

Contains:
  - Dataset generation and training loop for diagnostic runs
  - Per-neuron envelope computation and tau extraction
  - Mixture construction and GELR-weighted envelope
  - Comparison metrics (correlation, RMSE, relative error)
  - I/O helpers

Model definitions and transport functions are imported from the
canonical modules (models.py, transport.py) in the project root.

Refactored 2026-03: removed inline copies of model classes and
transport functions.  All code now uses the single canonical
implementation in transport.py and models.py.
"""

import csv
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# ============================================================
# Import canonical modules from project root
# ============================================================

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_root_on_path():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_root_on_path()

# Import canonical transport functions (single source of truth)
from transport import (  # noqa: E402
    precompute_prefix_sums,
    mu_diag_product_first_order,
    mu_diag_product_first_order_matched,
    prod_from_prefix,
    prod_from_prefix_matched,
    compute_mu_tl_for_lag,
    compute_mu_tl_for_matched_stat,
)

# Import canonical model definitions (single source of truth)
from models import (  # noqa: E402
    BaseRNN,
    ConstGateRNN,
    SharedGateRNN,
    DiagGateRNN,
    GRUCustom,
    LSTMCustom,
)


# ============================================================
# Seeding
# ============================================================

def set_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_MAIN_DIAGNOSTICS = None


def load_main_diagnostics_module():
    """Load the root diagnostics.py so sidecar checks reuse main-pipeline logic."""
    global _MAIN_DIAGNOSTICS
    if _MAIN_DIAGNOSTICS is not None:
        return _MAIN_DIAGNOSTICS

    root = _project_root()
    module_path = os.path.join(root, "diagnostics.py")
    spec = importlib.util.spec_from_file_location(
        "_anticollapse_main_diagnostics", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load main diagnostics module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MAIN_DIAGNOSTICS = module
    return module


# ============================================================
# Dataset
# ============================================================

def make_dataset(Nseq: int, T: int, D: int,
                 task_lags: List[int] = [10, 50],
                 task_coeffs: List[float] = [1.0, 0.5],
                 noise_std: float = 0.1,
                 seed: Optional[int] = None):
    """
    Synthetic delayed-regression task:
        y_t = sum_k c_k u^T x_{t-l_k} + eps_t
    Returns (X, Y, u) as CPU tensors.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    u = rng.randn(D).astype(np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)

    X = rng.randn(Nseq, T, D).astype(np.float32)
    Y = np.zeros((Nseq, T, 1), dtype=np.float32)

    for k, lag in enumerate(task_lags):
        c = float(task_coeffs[k])
        if lag < T:
            proj = np.einsum("ntd,d->nt", X[:, :T - lag, :], u)
            Y[:, lag:, 0] += c * proj

    Y += noise_std * rng.randn(Nseq, T, 1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y), u


# ============================================================
# Model and optimizer construction
# ============================================================

def build_model(arch: str, D: int, H: int, const_s: float = 0.005) -> BaseRNN:
    """Build a model from the canonical models.py definitions."""
    arch = arch.lower()
    if arch == "const":
        return ConstGateRNN(D, H, s=const_s)
    if arch == "shared":
        return SharedGateRNN(D, H, init_s=const_s)
    if arch == "diag":
        return DiagGateRNN(D, H, init_s=const_s)
    if arch == "gru":
        return GRUCustom(D, H)
    if arch == "lstm":
        return LSTMCustom(D, H)
    raise ValueError(f"Unknown arch: {arch}")


def build_optimizer(model: nn.Module, opt_name: str, lr: float = 1e-3) -> torch.optim.Optimizer:
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {opt_name}")


# ============================================================
# Training loop
# ============================================================

def train_model(model: BaseRNN, optimizer: torch.optim.Optimizer,
                X: torch.Tensor, Y: torch.Tensor,
                device: torch.device, epochs: int,
                batch_size: int = 32, verbose: bool = True) -> List[float]:
    """Train model and return list of per-epoch losses."""
    model.to(device)
    model.train()
    Nseq = X.shape[0]
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(Nseq)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, Nseq, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            optimizer.zero_grad()
            yhat, _, _ = model(xb, return_intermediates=False)
            loss = F.mse_loss(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        avg_loss = ep_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={avg_loss:.6f}")
    model.eval()
    return losses


# ============================================================
# Envelope and per-neuron rate computation
#
# Uses the canonical transport functions from transport.py.
# mu_diag_product_first_order now returns (mu0, mu1, mu_tl);
# we use mu_tl (the combined term) for envelope magnitude.
# ============================================================

@torch.no_grad()
def compute_per_neuron_envelope(model: BaseRNN, X: torch.Tensor,
                                device: torch.device,
                                lags: np.ndarray,
                                batch_size: int = 32
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-neuron, per-lag envelope statistics.

    Returns:
        f_q:     (n_lags, H)  E_{seq,t}[|μ^(q)_{t,ℓ}|]
        log_f_q: (n_lags, H)  E_{seq,t}[log|μ^(q)_{t,ℓ}|]

    The log-space accumulation (log_f_q) matches the theoretical definition
    of μ̄_q = -lim (1/ℓ) E_t[log|μ^(q)|] and should be used for τ extraction.
    The linear-space version (f_q) is kept for mixture construction and
    other diagnostics that need E[|μ|] directly.
    """
    model.eval()
    lags = np.asarray(lags, dtype=int)
    Nseq = X.shape[0]
    H_model = None
    sum_f = None
    sum_log_f = None
    n_seq = 0

    for i in range(0, Nseq, batch_size):
        xb = X[i:i + batch_size].to(device)
        _, _, g = model(xb, return_intermediates=True)
        leak = g["leak"]
        rdiag = g["rdiag"]
        if H_model is None:
            H_model = leak.shape[-1]
            sum_f = np.zeros((len(lags), H_model), dtype=np.float64)
            sum_log_f = np.zeros((len(lags), H_model), dtype=np.float64)
        cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)
        for j, ell in enumerate(lags):
            # mu_diag_product_first_order returns (mu0, mu1, mu_tl)
            _mu0, _mu1, mu_tl = mu_diag_product_first_order(
                cs_log, cs_ratio, leak, rdiag, int(ell), leak.dtype)
            if mu_tl.numel() == 0:
                continue
            abs_mu = torch.abs(mu_tl).double()
            # E_t[|μ^(q)|] per sequence: (B, H)
            abs_mu_bh = abs_mu.mean(dim=1)
            sum_f[j] += abs_mu_bh.sum(dim=0).cpu().numpy()
            # E_t[log|μ^(q)|] per sequence: (B, H)
            log_abs_mu_bh = torch.log(abs_mu.clamp(min=1e-30)).mean(dim=1)
            sum_log_f[j] += log_abs_mu_bh.sum(dim=0).cpu().numpy()
        n_seq += xb.shape[0]

    f_q = sum_f / max(n_seq, 1)          # (n_lags, H)
    log_f_q = sum_log_f / max(n_seq, 1)  # (n_lags, H)
    return f_q, log_f_q


@torch.no_grad()
def compute_aggregate_envelope(f_q: np.ndarray) -> np.ndarray:
    """
    f_q: (n_lags, H)
    Returns f(ell) = (1/H) sum_q f_q[ell, q], shape (n_lags,).
    """
    return f_q.mean(axis=1)


# ============================================================
# Tau extraction from per-neuron slopes
# ============================================================

def extract_tau_spectrum(f_q: np.ndarray, lags: np.ndarray,
                         fit_lag_min: int = 20,
                         log_f_q: Optional[np.ndarray] = None,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each neuron q, fit  E[log|μ^(q)|](ℓ) ~ a_q - μ̄_q * ℓ
    over lags >= fit_lag_min, matching the theoretical definition of μ̄_q.

    Args:
      f_q:     (n_lags, H) — E_{seq,t}[|μ^(q)_{t,ℓ}|], used as fallback
      lags:    (n_lags,) lag values
      fit_lag_min: minimum lag for the fit window
      log_f_q: (n_lags, H) — E_{seq,t}[log|μ^(q)_{t,ℓ}|], the theory-matched
               observable.  If provided, the fit uses this directly.
               If None, falls back to log(f_q) (the old behavior).

    Returns:
      tau:   (H,)  — time scale 1/μ̄_q  (NaN for invalid fits)
      r2:    (H,)  — R² of the linear fit
      mu_bar:(H,)  — asymptotic decay rate (NaN for invalid fits)
    """
    lags = np.asarray(lags, dtype=float)
    n_lags, H = f_q.shape

    # Select fitting window
    mask = lags >= fit_lag_min
    if mask.sum() < 4:
        # Fallback: use all lags
        mask = np.ones(n_lags, dtype=bool)

    ells_fit = lags[mask]
    A = np.vstack([np.ones_like(ells_fit), ells_fit]).T

    tau = np.full(H, np.nan)
    r2 = np.full(H, np.nan)
    mu_bar = np.full(H, np.nan)

    for q in range(H):
        if log_f_q is not None:
            # Theory-matched: use pre-accumulated E[log|μ|]
            y = log_f_q[mask, q]
        else:
            # Fallback: log(E[|μ|])
            y = np.log(f_q[mask, q] + 1e-30)
        finite = np.isfinite(y)
        if finite.sum() < 4:
            continue
        coeff, _, _, _ = np.linalg.lstsq(A[finite], y[finite], rcond=None)
        b_q = float(coeff[1])

        yhat = A[finite] @ coeff
        ss_res = np.sum((y[finite] - yhat) ** 2)
        ss_tot = np.sum((y[finite] - y[finite].mean()) ** 2) + 1e-12
        r2[q] = 1.0 - ss_res / ss_tot

        if b_q >= 0:
            # Non-negative slope: envelope is flat or growing, not decaying.
            # Mark as invalid (NaN) rather than manufacturing a spurious
            # giant τ by clamping to 1e-12.
            continue

        mu_bar[q] = -b_q
        tau[q] = 1.0 / (-b_q)

    return tau, r2, mu_bar


# ============================================================
# Mixture construction
# ============================================================

def build_mixture_envelope(tau: np.ndarray, lags: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Construct mixture-of-exponentials envelope:
      f_mix(ell) = (1/H) sum_q w_q * exp(-ell / tau_q)

    Args:
      tau:     (H,)  — per-neuron time scales
      lags:    (n_lags,) — lag values
      weights: (H,) optional — per-neuron weights (e.g. Lambda^(q)).
               If None, uniform weights.

    Returns: (n_lags,)
    """
    tau = np.asarray(tau, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.float64)

    H_total = len(tau)
    valid = np.isfinite(tau) & (tau > 0)
    tau_v = tau[valid]
    if tau_v.size == 0:
        return np.zeros(len(lags))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[valid]
    else:
        w = np.ones(tau_v.size)

    # f_mix(ell) = (1/H) sum_q w_q exp(-ell/tau_q)
    # Sum over valid neurons, divide by total H (invalid neurons contribute 0)
    exponents = -lags[:, None] / tau_v[None, :]
    f_mix = (w[None, :] * np.exp(exponents)).sum(axis=1) / H_total
    return f_mix


@torch.no_grad()
def compute_gelr_mixture_envelope(model: BaseRNN,
                                  optimizer: torch.optim.Optimizer,
                                  X: torch.Tensor,
                                  device: torch.device,
                                  lags: np.ndarray,
                                  tau: np.ndarray,
                                  batch_size: int = 32) -> Dict[str, Any]:
    """
    Build the GELR-weighted mixture using the same lag-dependent Rayleigh path
    as the main experimental pipeline:

      f_GELR-mix(ell) = E_{seq,t} (1/H) sum_q Lambda^(q)_{t,ell} exp(-ell/tau_q)
    """
    main_diag = load_main_diagnostics_module()

    model.eval()
    lags = np.asarray(lags, dtype=int)
    tau = np.asarray(tau, dtype=np.float64)
    valid_tau = np.isfinite(tau) & (tau > 0)
    tau_safe = np.where(valid_tau, tau, 1.0)
    exp_terms = np.zeros((len(lags), len(tau)), dtype=np.float64)
    for j, ell in enumerate(lags):
        exp_terms[j, valid_tau] = np.exp(-float(ell) / tau_safe[valid_tau])

    lambda_matrix, lambda_rowmean, lambda_meta = main_diag.extract_adaptive_rate_matrix(
        model, optimizer
    )
    lambda_rowmean_t = torch.tensor(lambda_rowmean, dtype=torch.float32, device=device)
    use_lag_dependent = lambda_matrix is not None
    if use_lag_dependent:
        lambda_matrix = lambda_matrix.to(device=device, dtype=torch.float32)

    Btot, T_full, _ = X.shape
    bs = int(batch_size)

    sum_mix = np.zeros(len(lags), dtype=np.float64)
    sum_lambda_mean = np.zeros(len(lags), dtype=np.float64)
    sum_lambda_sq = np.zeros(len(lags), dtype=np.float64)
    count_lambda = np.zeros(len(lags), dtype=np.float64)
    count_seq = 0

    exp_cache = {
        int(ell): torch.tensor(exp_terms[j], dtype=torch.float32, device=device).view(1, 1, -1)
        for j, ell in enumerate(lags)
    }

    for i in range(0, Btot, bs):
        xb = X[i:i + bs].to(device)
        _, hseq, _ = model.forward_with_intermediates(xb)
        B = int(xb.shape[0])

        for j, ell in enumerate(lags):
            T_valid = T_full - int(ell) + 1
            if T_valid <= 0:
                continue

            if use_lag_dependent:
                lambda_ell = main_diag.compute_lag_dependent_rates(
                    lambda_matrix=lambda_matrix,
                    hseq=hseq,
                    T_valid=T_valid,
                    fallback_rate=lambda_rowmean_t,
                )
            else:
                lambda_ell = lambda_rowmean_t.view(1, 1, -1).expand(B, T_valid, -1)

            weighted_exp = lambda_ell.double() * exp_cache[int(ell)].double()
            mix_mass = weighted_exp.mean(dim=2).mean(dim=1)
            sum_mix[j] += float(mix_mass.sum().item())

            lam_flat = lambda_ell.float()
            n_lam = int(lam_flat.numel())
            sum_lambda_mean[j] += float(lam_flat.mean().item()) * n_lam
            sum_lambda_sq[j] += float((lam_flat ** 2).mean().item()) * n_lam
            count_lambda[j] += n_lam

        count_seq += B

    lambda_mean = sum_lambda_mean / np.maximum(count_lambda, 1.0)
    lambda_var = (sum_lambda_sq / np.maximum(count_lambda, 1.0)) - (lambda_mean ** 2)
    lambda_std = np.sqrt(np.maximum(lambda_var, 0.0))

    return {
        "f_gelr_mix": (sum_mix / max(count_seq, 1)).astype(np.float64),
        "lambda_mean": lambda_mean.astype(np.float64),
        "lambda_std": lambda_std.astype(np.float64),
        "lambda_rowmean": lambda_rowmean.astype(np.float64),
        "gelr_mode": str(lambda_meta.get("mode", "uniform_fallback")),
        "used_lag_dependent_rates": bool(use_lag_dependent),
        "recurrent_matrices": list(lambda_meta.get("recurrent_matrices", [])),
    }


# ============================================================
# Comparison metrics
# ============================================================

def envelope_correlation(f_actual: np.ndarray, f_compare: np.ndarray) -> Dict[str, float]:
    """
    Compute Spearman rank correlation and log-space Pearson correlation
    between two envelope curves.

    Returns dict with keys: spearman_rho, spearman_p, pearson_r, pearson_p.
    """
    mask = (f_actual > 1e-30) & (f_compare > 1e-30) & np.isfinite(f_actual) & np.isfinite(f_compare)
    f_a = f_actual[mask]
    f_c = f_compare[mask]

    if len(f_a) < 4:
        return {"spearman_rho": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan, "n_points": 0}

    sp_rho, sp_p = stats.spearmanr(f_a, f_c)
    log_a = np.log10(f_a)
    log_c = np.log10(f_c)
    pe_r, pe_p = stats.pearsonr(log_a, log_c)

    return {"spearman_rho": float(sp_rho), "spearman_p": float(sp_p),
            "pearson_r": float(pe_r), "pearson_p": float(pe_p),
            "n_points": int(len(f_a))}


def relative_error_profile(f_actual: np.ndarray, f_compare: np.ndarray) -> np.ndarray:
    """Element-wise |f_actual - f_compare| / f_actual."""
    denom = np.maximum(np.abs(f_actual), 1e-30)
    return np.abs(f_actual - f_compare) / denom


def log_rmse(f_actual: np.ndarray, f_compare: np.ndarray) -> float:
    """RMSE in log10 space on the supported overlap of two envelopes."""
    mask = (f_actual > 1e-30) & (f_compare > 1e-30) & np.isfinite(f_actual) & np.isfinite(f_compare)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    diff = np.log10(f_actual[mask]) - np.log10(f_compare[mask])
    return float(np.sqrt(np.mean(diff ** 2)))


def relative_l2_error(f_actual: np.ndarray, f_compare: np.ndarray) -> float:
    """Relative L2 error over finite supported lags."""
    mask = np.isfinite(f_actual) & np.isfinite(f_compare)
    if np.count_nonzero(mask) < 4:
        return float("nan")
    numer = float(np.linalg.norm(f_actual[mask] - f_compare[mask]))
    denom = float(np.linalg.norm(f_actual[mask]) + 1e-30)
    return numer / denom


def mean_relative_error_supported(f_actual: np.ndarray,
                                  f_compare: np.ndarray,
                                  support_ratio: float = 1e-6) -> float:
    """
    Mean relative error restricted to lags where the reference envelope is above
    a fixed fraction of its peak, avoiding floor-driven blow-ups.
    """
    if f_actual.size == 0:
        return float("nan")
    peak = float(np.nanmax(np.abs(f_actual)))
    if (not np.isfinite(peak)) or peak <= 0:
        return float("nan")
    mask = (
        np.isfinite(f_actual)
        & np.isfinite(f_compare)
        & (np.abs(f_actual) >= peak * float(support_ratio))
    )
    if np.count_nonzero(mask) < 4:
        return float("nan")
    denom = np.maximum(np.abs(f_actual[mask]), 1e-30)
    return float(np.mean(np.abs(f_actual[mask] - f_compare[mask]) / denom))


# ============================================================
# I/O helpers
# ============================================================

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_csv(header: List[str], rows: List[List], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def save_run_manifest(path: str, args, **extra):
    """Persist the exact run configuration alongside each diagnostic run."""
    payload = {"args": vars(args).copy()}
    payload.update(extra)
    save_json(payload, path)
