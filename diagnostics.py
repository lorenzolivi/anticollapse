#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic pipeline for Anti-Collapse experiments.
=====================================================

Contains all measurement and estimation routines:
  - Tau spectrum extraction (per-neuron effective time scales)
  - CCDF power-law tail fitting (spectral exponent β̂)
  - Envelope-β consistency diagnostic (Laplace representation)
  - Macro envelope computation
  - Gradient noise tail index estimation (α̂)
  - Unified checkpoint diagnostic routine

All functions are model-agnostic — model-specific mu_tl computation
is handled by transport.compute_mu_tl_for_lag().
"""

import math
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transport import compute_mu_tl_for_lag
from seed_utils import write_csv, append_csv_row


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Tau spectrum estimation
# ============================================================

def estimate_tau_spectrum(
    model_name: str,
    model: nn.Module,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    diag_batch_size: int,
    fit_lags: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract per-neuron effective time scales from the mu_tl envelope slopes.

    For each neuron q:
      f_q(ℓ) = E_{seq,t} |mu_tl_q(t, ℓ)|
      log f_q(ℓ) ≈ a_q - μ̄_q ℓ   (fit over fit_lags)
      τ_q = 1 / μ̄_q

    Args:
        model_name: architecture name for mu_tl dispatch
        model: trained model
        Xdg_cpu: diagnostic input sequences (B, T, D) on CPU
        device: computation device
        diag_batch_size: batch size for diagnostic passes
        fit_lags: array of lag values to use for the linear fit

    Returns:
        tau: (H,) per-neuron time scales
        info: dict with fit quality metrics
    """
    model.eval()
    fit_lags = np.asarray(fit_lags, dtype=int)
    fit_lags = np.unique(fit_lags[fit_lags > 0])
    if fit_lags.size < 4:
        raise ValueError("fit_lags too small; need >=4 distinct lags.")

    Btot, T, _ = Xdg_cpu.shape
    bs = int(diag_batch_size)
    nb = max(1, math.ceil(Btot / bs))

    H = None
    sum_f = None  # (L, H)
    n_seq = 0

    for bi in range(nb):
        lo = bi * bs
        hi = min(Btot, (bi + 1) * bs)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)

        with torch.no_grad():
            _, _, intermediates = model.forward_with_intermediates(xb)

            if H is None:
                H = int(intermediates["leak"].shape[-1])
                sum_f = np.zeros((fit_lags.size, H), dtype=np.float64)

            prefix_cache = {}
            for j, ell in enumerate(fit_lags):
                mu_tl = compute_mu_tl_for_lag(
                    model_name, intermediates, int(ell),
                    out_dtype=intermediates["leak"].dtype,
                    _prefix_cache=prefix_cache,
                )
                if mu_tl.numel() == 0:
                    continue
                abs_f = torch.abs(mu_tl).double()
                f_bh = abs_f.mean(dim=1)  # (B, H) mean over start times
                sum_f[j, :] += f_bh.sum(dim=0).detach().cpu().numpy()

        n_seq += int(xb.shape[0])
        del xb, intermediates

    assert sum_f is not None and H is not None
    f = sum_f / max(1, n_seq)  # (L, H)
    log_f = np.log(f + 1e-30)

    # Linear fit: log f_q(ℓ) = a_q + b_q * ℓ  → τ_q = -1/b_q
    ells = fit_lags.astype(np.float64)
    A = np.vstack([np.ones_like(ells), ells]).T  # (L, 2)

    mu_bar = np.zeros(H, dtype=np.float64)
    r2s = np.full(H, np.nan, dtype=np.float64)
    valid = np.zeros(H, dtype=np.int32)

    for q in range(H):
        y = log_f[:, q]
        mask = np.isfinite(y)
        if np.count_nonzero(mask) < 4:
            continue
        coeff, _, _, _ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        b_q = float(coeff[1])
        mu_q = max(1e-12, -b_q)  # enforce positive decay rate
        mu_bar[q] = mu_q

        yhat = A[mask] @ coeff
        ss_res = float(np.sum((y[mask] - yhat) ** 2))
        ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2) + 1e-12)
        r2s[q] = 1.0 - ss_res / ss_tot
        valid[q] = int(np.count_nonzero(mask))

    tau = 1.0 / np.clip(mu_bar, 1e-12, np.inf)

    info = {
        "fit_lags": fit_lags.tolist(),
        "n_seq": int(n_seq),
        "tau_mean": float(np.mean(tau)),
        "tau_q90": float(np.quantile(tau, 0.90)),
        "tau_q99": float(np.quantile(tau, 0.99)),
        "mu_bar_mean": float(np.mean(mu_bar)),
        "r2_mean": float(np.nanmean(r2s)),
        "r2_q10": float(np.nanquantile(r2s, 0.10)),
        "r2_q50": float(np.nanquantile(r2s, 0.50)),
        "r2_q90": float(np.nanquantile(r2s, 0.90)),
        "n_valid_neurons": int(np.count_nonzero(valid > 0)),
    }
    return tau.astype(np.float64), info


# ============================================================
# CCDF and power-law tail fit
# ============================================================

def compute_ccdf_curve(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute complementary CDF of positive tau values."""
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    if tau.size == 0:
        return np.zeros(0), np.zeros(0)
    tau_sorted = np.sort(tau)
    n = tau_sorted.size
    ccdf = (n - np.arange(1, n + 1)) / max(1, n)
    return tau_sorted, ccdf


def fit_tau_ccdf_powerlaw(
    tau: np.ndarray,
    qmin: float,
    qmax: float,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Fit power law to the upper tail of the tau CCDF.

    Fits: log P(τ > x) = c - β log x   on the quantile range [qmin, qmax].

    Args:
        tau: (H,) per-neuron time scales
        qmin, qmax: quantile bounds for the fit (e.g. 0.75, 0.995)

    Returns:
        dict with beta_hat, beta_r2, x_lo, x_hi, n_fit
    """
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    if tau.size < 16:
        return {"beta_hat": 0.0, "beta_r2": float("nan"),
                "x_lo": float("nan"), "x_hi": float("nan"), "n_fit": 0}

    qmin = float(np.clip(qmin, 0.0, 0.99))
    qmax = float(np.clip(qmax, qmin + 1e-3, 0.999999))

    x_lo = float(np.quantile(tau, qmin))
    x_hi = float(np.quantile(tau, qmax))
    if (not np.isfinite(x_lo)) or (not np.isfinite(x_hi)) or (x_hi <= x_lo):
        return {"beta_hat": 0.0, "beta_r2": float("nan"),
                "x_lo": x_lo, "x_hi": x_hi, "n_fit": 0}

    tau_sorted = np.sort(tau)
    n = tau_sorted.size
    mask = (tau_sorted >= x_lo) & (tau_sorted <= x_hi)
    xs = tau_sorted[mask]
    if xs.size < 8:
        return {"beta_hat": 0.0, "beta_r2": float("nan"),
                "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs.size)}

    idxs = np.searchsorted(tau_sorted, xs, side="right")
    surv = (n - idxs) / max(1, n)

    # Remove duplicates and zero-survival points
    xs_unique, unique_idx = np.unique(xs, return_index=True)
    surv_unique = surv[unique_idx]
    pos_mask = surv_unique > 0
    xs_unique = xs_unique[pos_mask]
    surv_unique = surv_unique[pos_mask]
    if xs_unique.size < 8:
        return {"beta_hat": 0.0, "beta_r2": float("nan"),
                "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs_unique.size)}

    X = np.log(xs_unique + eps)
    Y = np.log(surv_unique)
    A = np.vstack([np.ones_like(X), X]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    slope = float(coeff[1])
    beta_hat = float(max(0.0, -slope))

    Yhat = A @ coeff
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {"beta_hat": beta_hat, "beta_r2": float(r2),
            "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs.size)}


# ============================================================
# Envelope-β consistency diagnostic (Laplace representation)
# ============================================================

def envelope_beta_from_tau_spectrum(
    tau: np.ndarray,
    ell_min: int = 64,
    ell_max: int = 512,
    n_ells: int = 32,
) -> Dict[str, float]:
    """
    Compute envelope f(ℓ) = (1/H) Σ_q exp(-ℓ/τ_q) from the tau spectrum
    and fit log f ≈ -β_env log ℓ for consistency check.

    This is the Laplace representation: f(ℓ) = ∫ e^{-ℓ/τ} p(τ) dτ
    evaluated empirically as a finite sum over neurons.
    """
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    H = tau.size
    if H < 8:
        return {"beta_env": float("nan"), "beta_env_r2": float("nan")}

    ells = np.unique(np.linspace(ell_min, ell_max, n_ells, dtype=int))
    ells = ells[ells > 0].astype(np.float64)
    if ells.size < 4:
        return {"beta_env": float("nan"), "beta_env_r2": float("nan")}

    f = np.zeros(ells.size, dtype=np.float64)
    for j, ell in enumerate(ells):
        f[j] = float(np.mean(np.exp(-ell / tau)))

    mask = (f > 1e-30) & np.isfinite(f)
    if np.sum(mask) < 4:
        return {"beta_env": float("nan"), "beta_env_r2": float("nan")}

    log_ell = np.log(ells[mask])
    log_f = np.log(f[mask])
    A = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff, _, _, _ = np.linalg.lstsq(A, log_f, rcond=None)
    beta_env = float(max(0.0, -coeff[1]))

    yhat = A @ coeff
    ss_res = float(np.sum((log_f - yhat) ** 2))
    ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {"beta_env": beta_env, "beta_env_r2": float(r2)}


# ============================================================
# Macro envelope f(ℓ) computation
# ============================================================

def compute_macro_envelope(
    model_name: str,
    model: nn.Module,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    ells: np.ndarray,
    diag_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute macro envelope f(ℓ) = E_{seq,t} mean_q |mu_tl_q(ℓ)|.

    Returns:
        f_mean: (n_ells,) average envelope values
        log_f_mean: (n_ells,) log of average envelope values
    """
    model.eval()
    Btot, T, _ = Xdg_cpu.shape
    bs = int(diag_batch_size)
    nb = max(1, math.ceil(Btot / bs))

    sum_mass = np.zeros(len(ells), dtype=np.float64)
    sum_log_mass = np.zeros(len(ells), dtype=np.float64)
    count_seq = 0

    for bi in range(nb):
        lo = bi * bs
        hi = min(Btot, (bi + 1) * bs)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)

        with torch.no_grad():
            _, _, intermediates = model.forward_with_intermediates(xb)

            prefix_cache = {}
            for j, ell in enumerate(ells):
                mu_tl = compute_mu_tl_for_lag(
                    model_name, intermediates, int(ell),
                    out_dtype=intermediates["leak"].dtype,
                    _prefix_cache=prefix_cache,
                )
                if mu_tl.numel() == 0:
                    continue
                abs_f = torch.abs(mu_tl).double()
                mass_per_seq = abs_f.mean(dim=2).mean(dim=1)  # (B,)
                sum_mass[j] += float(mass_per_seq.sum().item())
                sum_log_mass[j] += float(
                    torch.log(mass_per_seq.clamp(min=1e-30)).sum().item()
                )

        count_seq += int(xb.shape[0])
        del xb, intermediates

    f_mean = sum_mass / max(1, count_seq)
    log_f_mean = sum_log_mass / max(1, count_seq)
    return f_mean.astype(np.float64), log_f_mean.astype(np.float64)


def extract_adaptive_rate_matrix(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[Optional[torch.Tensor], np.ndarray, Dict[str, object]]:
    """
    Extract the recurrent adaptive-rate matrix λ_{qj} from optimizer state.

    Returns:
        lambda_matrix: (H, H) tensor averaged across recurrent weight matrices,
            or None when no second-moment state is available.
        lambda_rowmean: (H,) numpy array of row-mean adaptive rates.
        meta: basic provenance about the extracted rates.
    """
    H = int(getattr(model, "H", 0))
    if H <= 0:
        raise ValueError("Model does not expose a valid hidden size H.")

    fallback_lr = float(optimizer.param_groups[0].get("lr", 1.0))

    recurrent_stems = {"wh", "us", "uz", "ur", "uh", "ui", "uf", "uo", "ug"}
    recurrent_params: List[Tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        stem = name.rsplit(".", 1)[0].split(".")[-1].lower()
        is_recurrent_name = (
            stem in recurrent_stems
            or "recurrent" in stem
            or stem.endswith("hh")
        )
        if (
            tuple(param.shape) == (H, H)
            and name.endswith(".weight")
            and "out" not in name
            and is_recurrent_name
        ):
            recurrent_params.append((name, param))

    if not recurrent_params:
        return None, np.full(H, fallback_lr, dtype=np.float64), {
            "mode": "uniform_fallback",
            "n_recurrent_matrices": 0,
            "n_matrices_with_state": 0,
            "recurrent_matrices": [],
        }

    group_cfg = {}
    for group in optimizer.param_groups:
        cfg = {
            "lr": float(group.get("lr", fallback_lr)),
            "eps": float(group.get("eps", 1e-8)),
            "beta2": float(group.get("betas", (0.9, 0.999))[1]),
        }
        for p in group["params"]:
            group_cfg[id(p)] = cfg

    lam_matrices = []
    used_names = []
    for name, param in recurrent_params:
        pstate = optimizer.state.get(param, {})
        cfg = group_cfg.get(id(param), {
            "lr": fallback_lr,
            "eps": 1e-8,
            "beta2": 0.999,
        })

        if "exp_avg_sq" in pstate:
            v = pstate["exp_avg_sq"]
            step = pstate.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = step.item()
            step = max(int(step), 1)
            v_hat = v / (1.0 - cfg["beta2"] ** step)
        elif "square_avg" in pstate:
            v_hat = pstate["square_avg"]
        else:
            continue

        if not torch.isfinite(v_hat).all():
            log(f"[adaptive_rate_matrix] {name}: non-finite optimizer state, skipping")
            continue

        lam = cfg["lr"] / (torch.sqrt(v_hat.float()) + cfg["eps"])
        lam_matrices.append(lam)
        used_names.append(name)

    if not lam_matrices:
        return None, np.full(H, fallback_lr, dtype=np.float64), {
            "mode": "uniform_fallback",
            "n_recurrent_matrices": len(recurrent_params),
            "n_matrices_with_state": 0,
            "recurrent_matrices": [name for name, _ in recurrent_params],
        }

    lambda_matrix = torch.stack(lam_matrices, dim=0).mean(dim=0).detach()
    lambda_rowmean = (
        lambda_matrix.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    )
    log(
        "[adaptive_rate_matrix] "
        f"mode=lag_dependent n_matrices={len(lam_matrices)} "
        f"row-mean range [{lambda_rowmean.min():.4e}, {lambda_rowmean.max():.4e}]"
    )
    return lambda_matrix, lambda_rowmean, {
        "mode": "lag_dependent",
        "n_recurrent_matrices": len(recurrent_params),
        "n_matrices_with_state": len(lam_matrices),
        "recurrent_matrices": used_names,
    }


def compute_lag_dependent_rates(
    lambda_matrix: torch.Tensor,
    hseq: torch.Tensor,
    T_valid: int,
    fallback_rate: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the lag-dependent Rayleigh-quotient base rates Λ^(q)_{r,ℓ}(b,t).
    """
    B, T_full, H = hseq.shape
    device = hseq.device
    dtype = hseq.dtype

    h_pre = torch.zeros(B, T_valid, H, device=device, dtype=dtype)
    n_copy = min(max(T_valid - 1, 0), T_full)
    if n_copy > 0:
        h_pre[:, 1:1 + n_copy, :] = hseq[:, :n_copy, :]

    h_sq = h_pre ** 2
    h_sq_sum = h_sq.sum(dim=2, keepdim=True)
    numer = torch.matmul(h_sq.float(), lambda_matrix.T.float())
    Lambda_ell = numer / (h_sq_sum.float() + 1e-30)

    zero_mask = (h_sq_sum.squeeze(-1) < 1e-30)
    if zero_mask.any():
        Lambda_ell[zero_mask] = fallback_rate.float().unsqueeze(0)
    return Lambda_ell


def compute_macro_envelope_comparison(
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    ells: np.ndarray,
    diag_batch_size: int,
) -> Dict[str, object]:
    """
    Compute both the transport-only and GELR-weighted macro envelopes.
    """
    model.eval()
    Btot, _, _ = Xdg_cpu.shape
    bs = int(diag_batch_size)
    nb = max(1, math.ceil(Btot / bs))

    lambda_matrix, lambda_rowmean, lambda_meta = extract_adaptive_rate_matrix(
        model, optimizer
    )
    lambda_rowmean_t = torch.tensor(
        lambda_rowmean, dtype=torch.float32, device=device
    )
    use_lag_dependent = lambda_matrix is not None
    if use_lag_dependent:
        lambda_matrix = lambda_matrix.to(device=device, dtype=torch.float32)

    fallback_mean = float(np.mean(lambda_rowmean))
    fallback_sq_mean = float(np.mean(lambda_rowmean ** 2))

    sum_transport = np.zeros(len(ells), dtype=np.float64)
    sum_log_transport = np.zeros(len(ells), dtype=np.float64)
    sum_gelr = np.zeros(len(ells), dtype=np.float64)
    sum_log_gelr = np.zeros(len(ells), dtype=np.float64)
    sum_lambda_mean = np.zeros(len(ells), dtype=np.float64)
    sum_lambda_sq = np.zeros(len(ells), dtype=np.float64)
    count_lambda = np.zeros(len(ells), dtype=np.float64)
    count_seq = 0

    for bi in range(nb):
        lo = bi * bs
        hi = min(Btot, (bi + 1) * bs)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)

        with torch.no_grad():
            _, hseq, intermediates = model.forward_with_intermediates(xb)

            prefix_cache = {}
            for j, ell in enumerate(ells):
                mu_tl = compute_mu_tl_for_lag(
                    model_name, intermediates, int(ell),
                    out_dtype=intermediates["leak"].dtype,
                    _prefix_cache=prefix_cache,
                )
                if mu_tl.numel() == 0:
                    continue

                abs_transport = torch.abs(mu_tl).double()
                mass_transport = abs_transport.mean(dim=2).mean(dim=1)
                sum_transport[j] += float(mass_transport.sum().item())
                sum_log_transport[j] += float(
                    torch.log(mass_transport.clamp(min=1e-30)).sum().item()
                )

                if use_lag_dependent:
                    lambda_ell = compute_lag_dependent_rates(
                        lambda_matrix, hseq, mu_tl.shape[1], lambda_rowmean_t
                    )
                    lam_flat = lambda_ell.detach().float()
                    lam_mean = float(lam_flat.mean().item())
                    lam_sq = float((lam_flat ** 2).mean().item())
                    n_lam = int(lam_flat.numel())
                else:
                    lambda_ell = lambda_rowmean_t.view(1, 1, -1)
                    n_lam = int(mu_tl.shape[0] * mu_tl.shape[1] * mu_tl.shape[2])
                    lam_mean = fallback_mean
                    lam_sq = fallback_sq_mean

                mu_gelr = mu_tl * lambda_ell.to(mu_tl.dtype)
                abs_gelr = torch.abs(mu_gelr).double()
                mass_gelr = abs_gelr.mean(dim=2).mean(dim=1)
                sum_gelr[j] += float(mass_gelr.sum().item())
                sum_log_gelr[j] += float(
                    torch.log(mass_gelr.clamp(min=1e-30)).sum().item()
                )

                sum_lambda_mean[j] += lam_mean * n_lam
                sum_lambda_sq[j] += lam_sq * n_lam
                count_lambda[j] += n_lam

        count_seq += int(xb.shape[0])
        del xb, hseq, intermediates

    f_transport = sum_transport / max(1, count_seq)
    log_f_transport = sum_log_transport / max(1, count_seq)
    f_gelr = sum_gelr / max(1, count_seq)
    log_f_gelr = sum_log_gelr / max(1, count_seq)

    lambda_mean = sum_lambda_mean / np.maximum(count_lambda, 1.0)
    lambda_var = (sum_lambda_sq / np.maximum(count_lambda, 1.0)) - (lambda_mean ** 2)
    lambda_std = np.sqrt(np.maximum(lambda_var, 0.0))

    return {
        "f_transport": f_transport.astype(np.float64),
        "log_f_transport": log_f_transport.astype(np.float64),
        "f_gelr": f_gelr.astype(np.float64),
        "log_f_gelr": log_f_gelr.astype(np.float64),
        "lambda_mean": lambda_mean.astype(np.float64),
        "lambda_std": lambda_std.astype(np.float64),
        "lambda_rowmean": lambda_rowmean.astype(np.float64),
        "gelr_mode": str(lambda_meta.get("mode", "uniform_fallback")),
        "used_lag_dependent_rates": bool(use_lag_dependent),
        "recurrent_matrices": list(lambda_meta.get("recurrent_matrices", [])),
    }


def compute_and_save_final_envelopes(
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    mdir: str,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    ells: np.ndarray,
    diag_batch_size: int,
) -> Dict[str, object]:
    """
    Save the transport-only envelope bundle plus GELR-weighted comparison files.
    """
    env = compute_macro_envelope_comparison(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        Xdg_cpu=Xdg_cpu,
        device=device,
        ells=ells,
        diag_batch_size=diag_batch_size,
    )

    fit_transport = fit_log_envelope_exp_and_power(ells, env["log_f_transport"])
    write_csv(
        os.path.join(mdir, f"{model_name}_envelope.csv"),
        ["ell", "f_mean", "log_f_mean"],
        [
            [int(e), float(fv), float(lv)]
            for e, fv, lv in zip(ells, env["f_transport"], env["log_f_transport"])
        ],
    )
    with open(os.path.join(mdir, f"{model_name}_envelope_fit.json"), "w") as jf:
        json.dump(fit_transport, jf, indent=2)

    log_f_exp, log_f_pow, log_f_temp = eval_envelope_fit_curves(
        ells, fit_transport, include_tempered=True
    )
    write_csv(
        os.path.join(mdir, f"{model_name}_envelope_fit_curves.csv"),
        ["ell", "log_f_data", "log_f_exp_fit", "log_f_power_fit", "log_f_tempered_fit"],
        [
            [int(e), float(ld), float(le), float(lp), float(lt)]
            for e, ld, le, lp, lt in zip(
                ells, env["log_f_transport"], log_f_exp, log_f_pow, log_f_temp
            )
        ],
    )

    write_csv(
        os.path.join(mdir, f"{model_name}_adaptive_base_rates.csv"),
        ["neuron_q", "Lambda_q"],
        [[int(q), float(v)] for q, v in enumerate(env["lambda_rowmean"])],
    )

    fit_gelr = fit_log_envelope_exp_and_power(ells, env["log_f_gelr"])
    write_csv(
        os.path.join(mdir, f"{model_name}_gelr_envelope_compare.csv"),
        [
            "ell",
            "f_transport",
            "log_f_transport",
            "f_gelr",
            "log_f_gelr",
            "lambda_mean",
            "lambda_std",
        ],
        [
            [int(e), float(ft), float(lft), float(fg), float(lfg), float(lm), float(ls)]
            for e, ft, lft, fg, lfg, lm, ls in zip(
                ells,
                env["f_transport"],
                env["log_f_transport"],
                env["f_gelr"],
                env["log_f_gelr"],
                env["lambda_mean"],
                env["lambda_std"],
            )
        ],
    )

    with open(os.path.join(mdir, f"{model_name}_gelr_fit.json"), "w") as jf:
        json.dump({
            "gelr_mode": env["gelr_mode"],
            "used_lag_dependent_rates": env["used_lag_dependent_rates"],
            "recurrent_matrices": env["recurrent_matrices"],
            "lambda_rowmean_min": float(np.min(env["lambda_rowmean"])),
            "lambda_rowmean_max": float(np.max(env["lambda_rowmean"])),
            "lambda_rowmean_mean": float(np.mean(env["lambda_rowmean"])),
            "transport_fit": fit_transport,
            "gelr_fit": fit_gelr,
        }, jf, indent=2)

    log_f_gelr_exp, log_f_gelr_pow, log_f_gelr_temp = eval_envelope_fit_curves(
        ells, fit_gelr, include_tempered=True
    )
    write_csv(
        os.path.join(mdir, f"{model_name}_gelr_fit_curves.csv"),
        [
            "ell",
            "log_f_gelr_data",
            "log_f_gelr_exp_fit",
            "log_f_gelr_power_fit",
            "log_f_gelr_tempered_fit",
        ],
        [
            [int(e), float(ld), float(le), float(lp), float(lt)]
            for e, ld, le, lp, lt in zip(
                ells,
                env["log_f_gelr"],
                log_f_gelr_exp,
                log_f_gelr_pow,
                log_f_gelr_temp,
            )
        ],
    )

    return {
        "transport_fit": fit_transport,
        "gelr_fit": fit_gelr,
        "gelr_mode": env["gelr_mode"],
        "used_lag_dependent_rates": env["used_lag_dependent_rates"],
    }


# ============================================================
# Envelope fit utilities (exp vs power law)
# ============================================================

def fit_log_envelope_exp_and_power(
    ells: np.ndarray, log_mu: np.ndarray
) -> Dict:
    """
    Fit exponential, pure power-law, and tempered power-law models.

    exp:   log f = a + b ℓ         → f ~ exp(-ℓ/τ), τ = -1/b
    power: log f = c + d log ℓ    → f ~ ℓ^d
    temp:  log f = a + d log ℓ + b ℓ
           → f ~ A ℓ^{-β} exp(-ℓ/τ_max), with β = -d, τ_max = -1/b

    Function name is retained for backward compatibility.
    """
    ells = np.asarray(ells, dtype=float)
    log_mu = np.asarray(log_mu, dtype=float)
    mask = np.isfinite(ells) & np.isfinite(log_mu) & (ells > 0)
    ells = ells[mask]
    log_mu = log_mu[mask]
    if ells.size < 6:
        return {}

    ss_tot = float(np.sum((log_mu - np.mean(log_mu)) ** 2) + 1e-12)

    # Exponential fit
    A_exp = np.vstack([np.ones_like(ells), ells]).T
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, log_mu, rcond=None)
    pred_exp = A_exp @ coeff_exp
    ss_res_exp = float(np.sum((log_mu - pred_exp) ** 2))
    r2_exp = 1.0 - ss_res_exp / ss_tot
    a, b = float(coeff_exp[0]), float(coeff_exp[1])
    tau_env = float(-1.0 / b) if b < 0 else float("inf")

    # Power-law fit
    log_ell = np.log(ells + 1e-12)
    A_pow = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, log_mu, rcond=None)
    pred_pow = A_pow @ coeff_pow
    ss_res_pow = float(np.sum((log_mu - pred_pow) ** 2))
    r2_pow = 1.0 - ss_res_pow / ss_tot
    c, d = float(coeff_pow[0]), float(coeff_pow[1])

    # Tempered power-law fit
    A_temp = np.vstack([np.ones_like(log_ell), log_ell, ells]).T
    coeff_temp, _, _, _ = np.linalg.lstsq(A_temp, log_mu, rcond=None)
    pred_temp = A_temp @ coeff_temp
    ss_res_temp = float(np.sum((log_mu - pred_temp) ** 2))
    r2_temp = 1.0 - ss_res_temp / ss_tot
    a_temp, d_log, b_ell = (
        float(coeff_temp[0]),
        float(coeff_temp[1]),
        float(coeff_temp[2]),
    )
    beta_temp = float(max(0.0, -d_log))
    tau_max = float(-1.0 / b_ell) if b_ell < 0 else float("inf")

    return {
        "exp": {"a": a, "b": b, "r2": float(r2_exp), "tau_env": tau_env},
        "power": {"c": c, "d": d, "r2": float(r2_pow)},
        "tempered": {
            "a": a_temp,
            "d_log": d_log,
            "b_ell": b_ell,
            "beta_env": beta_temp,
            "tau_max": tau_max,
            "r2": float(r2_temp),
        },
    }


def eval_envelope_fit_curves(
    ells_full: np.ndarray,
    fit: Dict,
    include_tempered: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fitted curves over a lag grid."""
    ells_full = np.asarray(ells_full, dtype=float)
    log_f_exp = np.full_like(ells_full, np.nan, dtype=np.float64)
    log_f_pow = np.full_like(ells_full, np.nan, dtype=np.float64)
    log_f_temp = np.full_like(ells_full, np.nan, dtype=np.float64)
    if not fit:
        return (log_f_exp, log_f_pow, log_f_temp) if include_tempered else (log_f_exp, log_f_pow)
    if "exp" in fit and all(k in fit["exp"] for k in ["a", "b"]):
        a, b = float(fit["exp"]["a"]), float(fit["exp"]["b"])
        log_f_exp = a + b * ells_full
    if "power" in fit and all(k in fit["power"] for k in ["c", "d"]):
        c, d = float(fit["power"]["c"]), float(fit["power"]["d"])
        log_f_pow = c + d * np.log(ells_full + 1e-12)
    if "tempered" in fit and all(k in fit["tempered"] for k in ["a", "d_log", "b_ell"]):
        a = float(fit["tempered"]["a"])
        d_log = float(fit["tempered"]["d_log"])
        b_ell = float(fit["tempered"]["b_ell"])
        log_f_temp = a + d_log * np.log(ells_full + 1e-12) + b_ell * ells_full
    return (log_f_exp, log_f_pow, log_f_temp) if include_tempered else (log_f_exp, log_f_pow)


# ============================================================
# Gradient noise α estimation
# ============================================================

def _random_unit_vector_for_params(
    model: nn.Module, device: torch.device, seed: int
) -> Dict[str, torch.Tensor]:
    """Generate a random unit vector in parameter space."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if len(params) == 0:
        return {}
    w = {
        k: torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype)
        for k, v in params.items()
    }
    norm2 = None
    for t in w.values():
        val = (t.detach() ** 2).sum()
        norm2 = val if norm2 is None else (norm2 + val)
    norm = torch.sqrt(norm2 + 1e-12)
    return {k: t / norm for k, t in w.items()}


def _grad_projection_sample(
    model: nn.Module, w_unit: Dict[str, torch.Tensor]
) -> float:
    """Project current gradient onto a fixed direction."""
    s = None
    for k, p in model.named_parameters():
        if (not p.requires_grad) or (p.grad is None) or (k not in w_unit):
            continue
        val = torch.sum(p.grad.detach() * w_unit[k])
        s = val if s is None else (s + val)
    if s is None:
        return 0.0
    return float(s.item())


def _collect_grad_projections(
    model: nn.Module,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    n_grad_batches: int,
    w_seed: int,
    grad_clip: float = 0.0,
    winsorize_pct: Optional[float] = None,
) -> np.ndarray:
    """
    Collect gradient projection samples onto a single random direction.

    Returns: 1-D array of shape (n_grad_batches,).
    """
    was_training = model.training
    model.train()

    Btot = int(X_cpu.shape[0])
    bs = min(int(batch_size), Btot)
    w_unit = _random_unit_vector_for_params(model, device=device, seed=w_seed)

    samples = np.zeros(int(n_grad_batches), dtype=np.float64)

    with torch.enable_grad():
        for i in range(int(n_grad_batches)):
            idx = torch.randint(low=0, high=Btot, size=(bs,), device="cpu")
            xb = X_cpu[idx].to(device, non_blocking=True)
            yb = Y_cpu[idx].to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
            loss = F.mse_loss(yhat, yb)
            loss.backward()

            if winsorize_pct is not None and winsorize_pct < 100:
                all_grads = torch.cat([
                    p.grad.detach().view(-1)
                    for p in model.parameters() if p.grad is not None
                ])
                threshold = torch.quantile(all_grads.abs(), winsorize_pct / 100.0)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.clamp_(-threshold, threshold)

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            samples[i] = _grad_projection_sample(model, w_unit)
            del xb, yb, yhat, loss

    model.train(was_training) if was_training else model.eval()
    return samples


# ============================================================
# Unified checkpoint diagnostics
# ============================================================

def run_checkpoint_diagnostics(
    args,
    model: nn.Module,
    model_name: str,
    mdir: str,
    epoch: int,
    Xtr_cpu: torch.Tensor,
    Ytr_cpu: torch.Tensor,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    fit_lags: np.ndarray,
    step: Optional[int] = None,
    alpha_batch_size_override: Optional[int] = None,
    alpha_grad_clip_override: Optional[float] = None,
    alpha_winsorize_pct: Optional[float] = None,
) -> Dict:
    """
    Run full diagnostic suite at a training checkpoint.

    Estimates:
      1. α̂ (gradient noise tail index) — K random directions, median aggregation
      2. τ spectrum (per-neuron effective time scales)
      3. β̂ (spectral exponent from CCDF power-law tail fit)
      4. β_env (envelope-β consistency check from Laplace representation)

    Saves per-checkpoint data to mdir subdirectories.

    Returns:
        row: dict with all diagnostic values for one phase_trajectory.csv row
    """
    ckpt_tag = f"ckpt_{int(epoch):04d}"

    # --- α estimation (dual-method: ECF + McCulloch) ---
    # Strategy:
    #   1. Collect gradient projections for K random directions
    #   2. Per direction: run McCulloch (works with ≥32 samples)
    #   3. Pool all samples (MAD-normalized per direction): run ECF + McCulloch
    #   4. Report both estimates; flag agreement as reliability indicator
    K = int(getattr(args, 'alpha_n_directions', 5))
    n_grad_batches = int(args.alpha_n_grad_batches_ckpt)
    if alpha_grad_clip_override is not None:
        gc = float(alpha_grad_clip_override)
    else:
        gc = float(args.grad_clip) if args.alpha_use_grad_clip else 0.0
    if alpha_batch_size_override is not None:
        bs_alpha = int(alpha_batch_size_override)
    else:
        bs_alpha = min(int(args.batch_size), int(args.alpha_grad_batch_size))

    # Step 1: collect raw samples per direction
    per_dir_samples = []   # list of K arrays, each (n_grad_batches,)
    sample_rows = []       # for CSV output
    sid = 0
    for k_dir in range(K):
        dir_seed = int(args.w_seed) + k_dir
        samp_k = _collect_grad_projections(
            model, Xtr_cpu, Ytr_cpu,
            device=device, batch_size=bs_alpha,
            n_grad_batches=n_grad_batches,
            w_seed=dir_seed, grad_clip=gc,
            winsorize_pct=alpha_winsorize_pct,
        )
        samp_k = samp_k - float(np.mean(samp_k))
        per_dir_samples.append(samp_k)
        for v in samp_k:
            sample_rows.append([sid, k_dir, float(v)])
            sid += 1

    # ── Step 2: per-direction bootstrap McCulloch ──
    # McCulloch works with ≥32 samples (just four quantiles + table lookup).
    # Bootstrap gives a 95% CI per direction for two purposes:
    #   a) weight the cross-direction aggregation by precision
    #   b) check whether the ECF point estimate falls inside the CI
    #      (cleaner agreement criterion than an arbitrary |Δ| tolerance)
    from alpha_utils import (
        estimate_alpha_sigma_mcculloch_symmetric_from_samples,
        estimate_alpha_sigma_ecf_symmetric,
    )
    N_BOOT = 200
    rng_boot = np.random.RandomState(42)

    mcc_per_dir = []          # point estimates per direction
    mcc_boot_ci_per_dir = []  # (lo, hi) 95% CI per direction
    mcc_boot_dist_per_dir = []  # full bootstrap distributions (for pooled CI)
    for samp_k in per_dir_samples:
        a_mcc, _ = estimate_alpha_sigma_mcculloch_symmetric_from_samples(samp_k)
        mcc_per_dir.append(a_mcc)
        n_k = samp_k.size
        if n_k >= 32:
            boot_alphas = np.empty(N_BOOT, dtype=np.float64)
            for b in range(N_BOOT):
                idx_b = rng_boot.randint(0, n_k, size=n_k)
                a_b, _ = estimate_alpha_sigma_mcculloch_symmetric_from_samples(samp_k[idx_b])
                boot_alphas[b] = a_b
            ci_lo = float(np.percentile(boot_alphas, 2.5))
            ci_hi = float(np.percentile(boot_alphas, 97.5))
            mcc_boot_ci_per_dir.append((ci_lo, ci_hi))
            mcc_boot_dist_per_dir.append(boot_alphas)
        else:
            mcc_boot_ci_per_dir.append((float("nan"), float("nan")))
            mcc_boot_dist_per_dir.append(None)

    # Weighted median across directions: weight by 1/(CI width)
    ci_widths = np.array([hi - lo if np.isfinite(hi - lo) else 1e6
                          for lo, hi in mcc_boot_ci_per_dir])
    weights = 1.0 / (ci_widths + 1e-8)
    weights /= weights.sum()
    mcc_arr = np.array(mcc_per_dir)
    sort_idx = np.argsort(mcc_arr)
    cum_w = np.cumsum(weights[sort_idx])
    median_idx = min(int(np.searchsorted(cum_w, 0.5)), len(sort_idx) - 1)
    alpha_mcc_wmedian = float(mcc_arr[sort_idx[median_idx]])

    # Pooled McCulloch bootstrap CI: merge all per-direction bootstrap
    # distributions (they all estimate the same α) to get a single CI
    valid_boot_dists = [d for d in mcc_boot_dist_per_dir if d is not None]
    if valid_boot_dists:
        pooled_boot = np.concatenate(valid_boot_dists)
        mcc_boot_ci_pooled = (
            float(np.percentile(pooled_boot, 2.5)),
            float(np.percentile(pooled_boot, 97.5)),
        )
    else:
        mcc_boot_ci_pooled = (float("nan"), float("nan"))

    alpha_mcc_median = float(np.median(mcc_per_dir))
    alpha_mcc_std = float(np.std(mcc_per_dir, ddof=1)) if K > 1 else 0.0
    mcc_mean_ci_width = float(np.mean(ci_widths[np.isfinite(ci_widths)])) \
        if np.any(np.isfinite(ci_widths)) else float("nan")

    # ── Step 3: pool samples for ECF ──
    # All 1-D marginals of a symmetric α-stable share the same α but may
    # differ in scale σ.  MAD normalization puts them on a common scale
    # so pooling is valid for α estimation.
    normalized_chunks = []
    for samp_k in per_dir_samples:
        med_k = float(np.median(samp_k))
        mad_k = float(np.median(np.abs(samp_k - med_k)))
        if mad_k > 1e-15:
            normalized_chunks.append((samp_k - med_k) / mad_k)
        else:
            normalized_chunks.append(samp_k - med_k)
    pooled = np.concatenate(normalized_chunks)
    n_pooled = pooled.size

    # ECF on pooled (primary estimator as recommended in the paper)
    alpha_ecf_pooled = float("nan")
    sigma_ecf_pooled = 0.0
    ecf_available = False
    if n_pooled >= 100:
        a_ecf, s_ecf = estimate_alpha_sigma_ecf_symmetric(pooled)
        alpha_ecf_pooled = a_ecf
        sigma_ecf_pooled = s_ecf
        ecf_available = True

    # McCulloch on pooled (secondary, for logging)
    alpha_mcc_pooled, sigma_mcc_pooled = \
        estimate_alpha_sigma_mcculloch_symmetric_from_samples(pooled)

    # ── Step 4: ECF primary, McCulloch+bootstrap as diagnostic filter ──
    #
    # Design: ECF is the primary estimate (uses full characteristic
    # function, recommended by the paper).  McCulloch+bootstrap provides
    # a diagnostic CI.  Agreement is checked by asking:
    #   "Does α̂_ECF fall inside the pooled McCulloch bootstrap 95% CI?"
    #
    # This is cleaner than an arbitrary |Δ| tolerance because the CI
    # width adapts to the actual sampling variability.
    #
    # alpha_hat:           the single value used downstream
    # alpha_ecf:           ECF pooled point estimate
    # alpha_mcculloch:     McCulloch bootstrap-weighted median
    # alpha_methods_agree: 1 if ECF falls within McCulloch bootstrap CI

    ci_lo, ci_hi = mcc_boot_ci_pooled
    if ecf_available and np.isfinite(alpha_ecf_pooled) and np.isfinite(ci_lo):
        ecf_in_ci = (ci_lo <= alpha_ecf_pooled <= ci_hi)
        alpha_methods_agree = int(ecf_in_ci)
        alpha_agreement = abs(alpha_ecf_pooled - alpha_mcc_wmedian)
        # ECF is always the primary estimate when available
        alpha_hat_best = alpha_ecf_pooled
        alpha_method_used = "ecf"
    elif ecf_available and np.isfinite(alpha_ecf_pooled):
        # ECF available but no valid McCulloch CI (very few samples)
        alpha_methods_agree = 0  # can't verify
        alpha_agreement = float("nan")
        alpha_hat_best = alpha_ecf_pooled
        alpha_method_used = "ecf"
    else:
        # ECF not available — fall back to McCulloch
        alpha_methods_agree = 0
        alpha_agreement = float("nan")
        alpha_hat_best = alpha_mcc_wmedian
        alpha_method_used = "mcculloch"

    alpha_hat_best = float(np.clip(alpha_hat_best, 1.0, 2.0))

    # Reliability — two tiers:
    #   Tier 1 (strongest): ECF available AND falls inside McCulloch bootstrap CI
    #   Tier 2 (acceptable): ECF not available, but McCulloch bootstrap CIs are
    #           tight across directions (mean CI width < 0.3) and cross-direction
    #           std is small — the estimate is stable even without ECF confirmation
    if ecf_available and alpha_methods_agree == 1:
        alpha_reliable = 1  # tier 1: cross-validated
    elif (not ecf_available
          and np.isfinite(mcc_mean_ci_width)
          and mcc_mean_ci_width < 0.3
          and alpha_mcc_std < 0.2):
        alpha_reliable = 1  # tier 2: McCulloch-only but tight CIs
    else:
        alpha_reliable = 0

    alpha_info = {
        "alpha_hat": alpha_hat_best,
        "alpha_ecf": alpha_ecf_pooled,
        "alpha_mcculloch": alpha_mcc_wmedian,
        "alpha_methods_agree": alpha_methods_agree,
        "alpha_agreement": alpha_agreement,
        "alpha_mcc_boot_ci_pooled": list(mcc_boot_ci_pooled),
        "alpha_mcc_pooled": alpha_mcc_pooled,
        "alpha_mcc_median": alpha_mcc_median,
        "sigma_alpha_hat": sigma_ecf_pooled if ecf_available else sigma_mcc_pooled,
        "alpha_hat_std": alpha_mcc_std,
        "alpha_hat_se": float(alpha_mcc_std / math.sqrt(K)) if K > 1 else 0.0,
        "alpha_hat_per_dir_mcc": mcc_per_dir,
        "alpha_boot_ci_per_dir": mcc_boot_ci_per_dir,
        "alpha_mcc_mean_ci_width": mcc_mean_ci_width,
        "alpha_reliable": alpha_reliable,
        "alpha_method": alpha_method_used,
        "grad_proj_mean": float(np.mean([r[2] for r in sample_rows])),
        "grad_proj_std": float(np.std([r[2] for r in sample_rows])),
        "n_samples": len(sample_rows),
        "n_pooled": n_pooled,
        "n_directions": K,
    }

    alpha_dir = os.path.join(mdir, "checkpoint_alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    write_csv(
        os.path.join(alpha_dir, f"{ckpt_tag}_alpha_grad_samples.csv"),
        ["sample_id", "direction_id", "grad_projection"],
        sample_rows,
    )
    # Sanitize NaN/Inf for JSON (json.dumps doesn't handle them)
    def _json_safe(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(os.path.join(alpha_dir, f"{ckpt_tag}_alpha_grad.json"), "w") as jf:
        json.dump(_json_safe(alpha_info), jf, indent=2)

    # --- τ spectrum ---
    tau, tau_fit_info = estimate_tau_spectrum(
        model_name, model, Xdg_cpu,
        device=device, diag_batch_size=int(args.diag_batch_size),
        fit_lags=fit_lags,
    )

    tau_dir = os.path.join(mdir, "checkpoint_taus")
    os.makedirs(tau_dir, exist_ok=True)
    np.save(os.path.join(tau_dir, f"{ckpt_tag}_taus.npy"), tau)
    write_csv(
        os.path.join(tau_dir, f"{ckpt_tag}_taus.csv"),
        ["unit_id", "tau"],
        [[int(i), float(t)] for i, t in enumerate(tau)],
    )
    with open(os.path.join(tau_dir, f"{ckpt_tag}_tau_slope_fit_info.json"), "w") as jf:
        json.dump(tau_fit_info, jf, indent=2)

    # --- CCDF + tail fit ---
    tau_sorted, ccdf = compute_ccdf_curve(tau)

    ccdf_dir = os.path.join(mdir, "checkpoint_tau_ccdf")
    if getattr(args, 'save_checkpoint_ccdf', False):
        os.makedirs(ccdf_dir, exist_ok=True)
        write_csv(
            os.path.join(ccdf_dir, f"{ckpt_tag}_tau_ccdf.csv"),
            ["tau", "ccdf"],
            [[float(x), float(y)] for x, y in zip(tau_sorted, ccdf)],
        )

    tail = fit_tau_ccdf_powerlaw(
        tau, qmin=float(args.tau_ccdf_qmin), qmax=float(args.tau_ccdf_qmax)
    )
    tail.update({
        "tau_mean": float(np.mean(tau)),
        "tau_q90": float(np.quantile(tau, 0.90)),
        "tau_q99": float(np.quantile(tau, 0.99)),
        "epoch": int(epoch),
        "model": str(model_name),
    })
    tail_dir = os.path.join(mdir, "checkpoint_tau_tail")
    os.makedirs(tail_dir, exist_ok=True)
    with open(os.path.join(tail_dir, f"{ckpt_tag}_tau_tail_fit.json"), "w") as jf:
        json.dump(tail, jf, indent=2)

    # --- Envelope-β consistency ---
    env_info = envelope_beta_from_tau_spectrum(tau)

    # --- Assemble row ---
    row = {
        "epoch": int(epoch),
        "step": int(step) if step is not None else float("nan"),
        "alpha_hat": float(alpha_info["alpha_hat"]),
        "alpha_ecf": float(alpha_info.get("alpha_ecf", float("nan"))),
        "alpha_mcculloch": float(alpha_info.get("alpha_mcculloch", float("nan"))),
        "sigma_alpha_hat": float(alpha_info["sigma_alpha_hat"]),
        "alpha_hat_std": float(alpha_info.get("alpha_hat_std", 0.0)),
        "alpha_hat_se": float(alpha_info.get("alpha_hat_se", 0.0)),
        "alpha_agreement": float(alpha_info.get("alpha_agreement", float("nan"))),
        "beta_hat": float(tail["beta_hat"]),
        "beta_r2": float(tail["beta_r2"]),
        "tau_mean": float(tail["tau_mean"]),
        "tau_q90": float(tail["tau_q90"]),
        "tau_q99": float(tail["tau_q99"]),
        "tau_fit_r2_mean": float(tau_fit_info.get("r2_mean", float("nan"))),
        "tau_fit_n_valid": int(tau_fit_info.get("n_valid_neurons", 0)),
        "fit_lags_min": int(np.min(fit_lags)),
        "fit_lags_max": int(np.max(fit_lags)),
        "alpha_reliable": int(alpha_info.get("alpha_reliable", 0)),
        "alpha_method": str(alpha_info.get("alpha_method", "mcculloch_median")),
        "n_samples": int(alpha_info.get("n_samples", 0)),
        "beta_env": float(env_info["beta_env"]),
        "beta_env_r2": float(env_info["beta_env_r2"]),
    }
    return row
