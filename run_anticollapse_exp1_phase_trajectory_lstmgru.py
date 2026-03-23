#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — Experiment 1 (GRU/LSTM, PHASE TRAJECTORY + learning curves)
===========================================================================

Models: GRU, LSTM

Core change vs static runner:
  - Track dynamical trajectory in (alpha_hat(t), beta_hat(t)) at checkpoints.
  - tau spectrum computed via learnability-style per-neuron slope fits:
        f_q(ell) = E_{seq,t} |mu_q(t, ell)|     <-- Note: envelope f is derived from mu_tl/mu
        log f_q(ell) ~ a_q - mu_bar_q * ell   (fit over large-lag window)
        tau_q = 1 / mu_bar_q
    where mu_q(t,ell) uses the first-order diagonal product expansions from your
    Learnability Window formulation:
      - GRU: mu_tl = (gamma + rho0 + eta0)
      - LSTM: mu_tl = (base * e_end)

Important implementation notes / fixes:
  - alpha estimation always re-enables autograd via torch.enable_grad().
  - checkpoint diagnostics are NOT decorated with @torch.no_grad (to avoid accidental grad disable).
  - model training/eval mode restored after alpha estimation.

Outputs (per model directory):
  - <model>_learning_curve.csv
  - phase_trajectory.csv
  - checkpoint_taus/ckpt_XXXX_taus.npy + .csv
  - checkpoint_taus/ckpt_XXXX_tau_slope_fit_info.json
  - checkpoint_tau_tail/ckpt_XXXX_tau_tail_fit.json
  - checkpoint_alpha/ckpt_XXXX_alpha_grad.json (+ samples csv)
  - checkpoint_tau_ccdf/ckpt_XXXX_tau_ccdf.csv (optional)

Optional final-only (set --save_final_envelope):
  - <model>_envelope.csv
  - <model>_envelope_fit.json
  - <model>_envelope_fit_curves.csv

Top-level:
  - cli_args.json, lag_grid.json

NO plotting here.
"""

import argparse, os, math, csv, json
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_utils import estimate_alpha_sigma
from seed_utils import write_csv, append_csv_row


# ============================================================
# Logger / utils
# ============================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def set_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def layernorm_if(enabled: bool, dim: int):
    return nn.LayerNorm(dim) if enabled else nn.Identity()


# ============================================================
# Dataset (CPU resident)
# ============================================================

def make_dataset_cpu(Nseq: int, T: int, D: int,
                     task_lags: List[int],
                     task_coeffs: List[float],
                     noise_std: float,
                     u_vec: Optional[np.ndarray] = None):
    """
    Synthetic regression task (CPU tensors):
        y_t = Σ_k c_k u^T x_{t-ℓ_k} + ε_t
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


# ============================================================
# First-order diagonal product helper (prefix sums)
# ============================================================

@torch.no_grad()
def precompute_prefix_sums(leak: torch.Tensor, rdiag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    leak: (B,T,H) positive factors (clipped)
    rdiag:(B,T,H) diagonal correction term
    Returns:
      cs_log  (B,T+1,H): cumsum log(leak)
      cs_ratio(B,T+1,H): cumsum (rdiag/leak)
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

@torch.no_grad()
def mu_diag_product_first_order(cs_log: torch.Tensor, cs_ratio: torch.Tensor,
                               leak: torch.Tensor, rdiag: torch.Tensor,
                               ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """
    First-order diagonal expansion of Π(D_t + R_t) with D_t=diag(leak_t):
      mu = prod(leak_window) + prod(leak_window) * sum_{p in window} rdiag_p / leak_p
    Returns (B, T-ell+1, H)
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)

    if ell == 1:
        return (leak + rdiag).to(out_dtype)

    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]

    # [NEW FIX B] Compute mu0 and mu1 entirely in float64 before casting down to prevent precision loss
    mu0_64 = torch.exp(log_prod)
    sum_ratio_window = cs_ratio[:, ell:(T + 1), :] - cs_ratio[:, 0:(T - ell + 1), :]
    mu1_64 = mu0_64 * sum_ratio_window

    return (mu0_64 + mu1_64).to(out_dtype)

@torch.no_grad()
def prod_from_prefix(cs_log: torch.Tensor, ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Pure diagonal product Π leak over window length ell.
    cs_log: (B,T+1,H)
    returns (B,T-ell+1,H)
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
# Models: GRU and LSTM with intermediates needed for µ
# ============================================================

class BaseRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def apply_orthogonal(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
                if getattr(m, '_skip_orth', False):
                    continue  # preserve deliberate zero-init on gate layers
                nn.init.orthogonal_(m.weight)

class GRUCustom(BaseRNN):
    def __init__(self, D: int, H: int, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H

        self.Wz = nn.Linear(D, H); self.Uz = nn.Linear(H, H, bias=False)
        self.Wr = nn.Linear(D, H); self.Ur = nn.Linear(H, H, bias=False)
        self.Wh = nn.Linear(D, H); self.Uh = nn.Linear(H, H, bias=False)

        self.ln_h = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)

        for m in [self.Wz, self.Wr, self.Wh, self.out]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_with_intermediates(self, x: torch.Tensor, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        ys = []
        if return_intermediates:
            diagUz = torch.diagonal(self.Uz.weight, 0)
            diagUr = torch.diagonal(self.Ur.weight, 0)
            diagUh = torch.diagonal(self.Uh.weight, 0)
            zs, rs, gs, hs = [], [], [], []
            leaks, rdiags = [], []

        for t in range(T):
            h_prev = h

            az = self.Wz(x[:, t]) + self.Uz(h_prev)
            z = torch.sigmoid(az)

            ar = self.Wr(x[:, t]) + self.Ur(h_prev)
            r = torch.sigmoid(ar)

            ag = self.Wh(x[:, t]) + self.Uh(r * h_prev)
            ag = self.ln_h(ag)
            g = torch.tanh(ag)

            h = (1.0 - z) * h_prev + z * g
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = torch.clamp(1.0 - z, 1e-12, 1.0)

                zprime = z * (1.0 - z)
                term1 = (g - h_prev) * zprime * diagUz.view(1, -1)

                gprime = 1.0 - g**2
                term2 = z * gprime * diagUh.view(1, -1) * r

                rprime = r * (1.0 - r)
                term3 = z * gprime * diagUh.view(1, -1) * h_prev * rprime * diagUr.view(1, -1)

                rdiag = term1 + term2 + term3

                hs.append(h)
                zs.append(z); rs.append(r); gs.append(g)
                leaks.append(leak); rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        zseq = torch.stack(zs, dim=1)
        rseq = torch.stack(rs, dim=1)
        gseq = torch.stack(gs, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"z": zseq, "r": rseq, "g": gseq, "leak": leak, "rdiag": rdiag}

class LSTMCustom(BaseRNN):
    def __init__(self, D: int, H: int, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H

        self.Wi = nn.Linear(D, H); self.Ui = nn.Linear(H, H, bias=False)
        self.Wf = nn.Linear(D, H); self.Uf = nn.Linear(H, H, bias=False)
        self.Wo = nn.Linear(D, H); self.Uo = nn.Linear(H, H, bias=False)
        self.Wg = nn.Linear(D, H); self.Ug = nn.Linear(H, H, bias=False)

        self.ln_g = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)

        for m in [self.Wi, self.Wf, self.Wo, self.Wg, self.out]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_with_intermediates(self, x: torch.Tensor, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        c = torch.zeros(B, self.H, device=x.device)

        ys = []
        if return_intermediates:
            diagUi = torch.diagonal(self.Ui.weight, 0)
            diagUf = torch.diagonal(self.Uf.weight, 0)
            diagUg = torch.diagonal(self.Ug.weight, 0)
            hs = []
            eseq = []
            leaks, rdiags = [], []
            emat, fmat = [], []

        for t in range(T):
            h_prev = h
            c_prev = c

            ai = self.Wi(x[:, t]) + self.Ui(h_prev)
            af = self.Wf(x[:, t]) + self.Uf(h_prev)
            ao = self.Wo(x[:, t]) + self.Uo(h_prev)
            ag = self.Wg(x[:, t]) + self.Ug(h_prev)
            ag = self.ln_g(ag)

            i = torch.sigmoid(ai)
            f = torch.sigmoid(af)
            o = torch.sigmoid(ao)
            g = torch.tanh(ag)

            c = f * c_prev + i * g
            tanh_c = torch.tanh(c)
            h = o * tanh_c
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                e = o * (1.0 - tanh_c**2)
                leak = torch.clamp(f, 1e-12, 1.0)

                fprime = f * (1.0 - f)
                iprime = i * (1.0 - i)
                gprime = 1.0 - g**2

                diagC = (c_prev * fprime) * diagUf.view(1, -1) \
                        + (i * gprime)    * diagUg.view(1, -1) \
                        + (g * iprime)    * diagUi.view(1, -1)

                e_prev = torch.zeros_like(e) if t == 0 else eseq[-1]
                rdiag_t = diagC * e_prev

                hs.append(h)
                eseq.append(e)
                leaks.append(leak); rdiags.append(rdiag_t)
                emat.append(e); fmat.append(f)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        e = torch.stack(emat, dim=1)
        f = torch.stack(fmat, dim=1)
        return y, hseq, {"e": e, "f": f, "leak": leak, "rdiag": rdiag}

def build_model(name: str, D: int, H: int, ln: bool) -> BaseRNN:
    name = name.lower()
    if name == "gru":
        return GRUCustom(D, H, ln=ln)
    if name == "lstm":
        return LSTMCustom(D, H, ln=ln)
    raise ValueError(f"Unknown model {name} (expected gru or lstm)")


# ============================================================
# Fit utilities
# ============================================================

def fit_log_envelope_exp_and_power(ells: np.ndarray, log_mu: np.ndarray) -> Dict:
    ells = np.asarray(ells, dtype=float)
    log_mu = np.asarray(log_mu, dtype=float)
    mask = np.isfinite(ells) & np.isfinite(log_mu) & (ells > 0)
    ells = ells[mask]
    log_mu = log_mu[mask]
    if ells.size < 6:
        return {}

    ss_tot = float(np.sum((log_mu - np.mean(log_mu))**2) + 1e-12)

    A_exp = np.vstack([np.ones_like(ells), ells]).T
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, log_mu, rcond=None)
    pred_exp = A_exp @ coeff_exp
    ss_res_exp = float(np.sum((log_mu - pred_exp)**2))
    r2_exp = 1.0 - ss_res_exp / ss_tot
    a, b = float(coeff_exp[0]), float(coeff_exp[1])
    tau_env = float(-1.0 / b) if b < 0 else float("inf")

    log_ell = np.log(ells + 1e-12)
    A_pow = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, log_mu, rcond=None)
    pred_pow = A_pow @ coeff_pow
    ss_res_pow = float(np.sum((log_mu - pred_pow)**2))
    r2_pow = 1.0 - ss_res_pow / ss_tot
    c, d = float(coeff_pow[0]), float(coeff_pow[1])

    return {
        "exp": {"a": a, "b": b, "r2": float(r2_exp), "tau_env": tau_env},
        "power": {"c": c, "d": d, "r2": float(r2_pow)}
    }

def eval_envelope_fit_curves(ells_full: np.ndarray, fit: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ells_full = np.asarray(ells_full, dtype=float)
    log_exp = np.full_like(ells_full, np.nan, dtype=np.float64)
    log_pow = np.full_like(ells_full, np.nan, dtype=np.float64)
    if not fit:
        return log_exp, log_pow
    if "exp" in fit and all(k in fit["exp"] for k in ["a", "b"]):
        a = float(fit["exp"]["a"]); b = float(fit["exp"]["b"])
        log_exp = a + b * ells_full
    if "power" in fit and all(k in fit["power"] for k in ["c", "d"]):
        c = float(fit["power"]["c"]); d = float(fit["power"]["d"])
        log_pow = c + d * np.log(ells_full + 1e-12)
    return log_exp, log_pow


# ============================================================
# tau CCDF tail fit
# ============================================================

def compute_ccdf_curve(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    if tau.size == 0:
        return np.zeros(0), np.zeros(0)
    tau_sorted = np.sort(tau)
    n = tau_sorted.size
    ccdf = (n - np.arange(1, n + 1)) / max(1, n)
    return tau_sorted, ccdf

def fit_tau_ccdf_powerlaw(tau: np.ndarray, qmin: float, qmax: float, eps: float = 1e-12) -> Dict[str, float]:
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    if tau.size < 16:
        return {"beta_hat": 0.0, "beta_r2": float("nan"), "x_lo": float("nan"), "x_hi": float("nan"), "n_fit": 0}

    qmin = float(np.clip(qmin, 0.0, 0.99))
    qmax = float(np.clip(qmax, qmin + 1e-3, 0.999999))

    x_lo = float(np.quantile(tau, qmin))
    x_hi = float(np.quantile(tau, qmax))
    if (not np.isfinite(x_lo)) or (not np.isfinite(x_hi)) or (x_hi <= x_lo):
        return {"beta_hat": 0.0, "beta_r2": float("nan"), "x_lo": x_lo, "x_hi": x_hi, "n_fit": 0}

    tau_sorted = np.sort(tau)
    n = tau_sorted.size
    mask = (tau_sorted >= x_lo) & (tau_sorted <= x_hi)
    xs = tau_sorted[mask]
    if xs.size < 8:
        return {"beta_hat": 0.0, "beta_r2": float("nan"), "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs.size)}

    idxs = np.searchsorted(tau_sorted, xs, side="right")
    surv = (n - idxs) / max(1, n)

    # Remove duplicate tau values (keep first occurrence → largest survival)
    # and drop points with surv=0 to avoid log(0)
    xs_unique, unique_idx = np.unique(xs, return_index=True)
    surv_unique = surv[unique_idx]
    pos_mask = surv_unique > 0
    xs_unique = xs_unique[pos_mask]
    surv_unique = surv_unique[pos_mask]
    if xs_unique.size < 8:
        return {"beta_hat": 0.0, "beta_r2": float("nan"), "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs_unique.size)}

    X = np.log(xs_unique + eps)
    Y = np.log(surv_unique)
    A = np.vstack([np.ones_like(X), X]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    slope = float(coeff[1])
    beta_hat = float(max(0.0, -slope))

    Yhat = A @ coeff
    ss_res = float(np.sum((Y - Yhat)**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {"beta_hat": beta_hat, "beta_r2": float(r2), "x_lo": x_lo, "x_hi": x_hi, "n_fit": int(xs.size)}


# ============================================================
# Envelope-β consistency diagnostic
# ============================================================

def envelope_beta_from_tau_spectrum(tau: np.ndarray,
                                   ell_min: int = 64,
                                   ell_max: int = 512,
                                   n_ells: int = 32) -> Dict[str, float]:
    """
    Compute the envelope f(ell) = (1/H) sum_q exp(-ell/tau_q)
    from the tau spectrum and fit a power law log f ~ -beta_env * log ell
    to check consistency with the CCDF-derived beta_hat.
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

    # Fit log f = a - beta_env * log ell
    mask = (f > 1e-30) & np.isfinite(f)
    if np.sum(mask) < 4:
        return {"beta_env": float("nan"), "beta_env_r2": float("nan")}

    log_ell = np.log(ells[mask])
    log_f = np.log(f[mask])
    A = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff, _, _, _ = np.linalg.lstsq(A, log_f, rcond=None)
    beta_env = float(max(0.0, -coeff[1]))

    yhat = A @ coeff
    ss_res = float(np.sum((log_f - yhat)**2))
    ss_tot = float(np.sum((log_f - np.mean(log_f))**2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {"beta_env": beta_env, "beta_env_r2": float(r2)}


# ============================================================
# Alpha (gradient projection) — uses alpha_utils
# ============================================================

def _random_unit_vector_for_params(model: nn.Module, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if len(params) == 0:
        return {}
    w = {k: torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype) for k, v in params.items()}
    norm2 = None
    for t in w.values():
        val = (t.detach() ** 2).sum()
        norm2 = val if norm2 is None else (norm2 + val)
    norm = torch.sqrt(norm2 + 1e-12)
    return {k: t / norm for k, t in w.items()}

def _grad_projection_sample(model: nn.Module, w_unit: Dict[str, torch.Tensor]) -> float:
    s = None
    for (k, p) in model.named_parameters():
        if (not p.requires_grad) or (p.grad is None) or (k not in w_unit):
            continue
        val = torch.sum(p.grad.detach() * w_unit[k])
        s = val if s is None else (s + val)
    if s is None:
        return 0.0
    return float(s.item())

def estimate_alpha_from_minibatch_gradients(model: nn.Module,
                                           X_cpu: torch.Tensor,
                                           Y_cpu: torch.Tensor,
                                           device: torch.device,
                                           batch_size: int,
                                           n_grad_batches: int,
                                           w_seed: int,
                                           grad_clip: float = 0.0,
                                           alpha_method: str = "mcculloch",
                                           min_samples_alpha: int = 500) -> Tuple[Dict[str, float], np.ndarray]:
    was_training = model.training
    model.train()

    Btot = int(X_cpu.shape[0])
    bs = int(batch_size)
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

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            samples[i] = _grad_projection_sample(model, w_unit)
            del xb, yb, yhat, loss

    # restore mode
    model.train(was_training) if was_training else model.eval()

    alpha_hat, sigma_hat, reliable = estimate_alpha_sigma(
        samples, method=alpha_method, min_samples=min_samples_alpha
    )
    info = {
        "alpha_hat": float(alpha_hat),
        "sigma_alpha_hat": float(sigma_hat),
        "alpha_reliable": int(reliable),
        "alpha_method": str(alpha_method),
        "grad_proj_mean": float(np.mean(samples)),
        "grad_proj_std": float(np.std(samples)),
        "n_samples": int(samples.size),
    }
    return info, samples


# ============================================================
# tau spectrum from per-neuron slope fits of f_q(ell) = E|mu_q(t,ell)|
# ============================================================

# [NEW FIX A] Removed @torch.no_grad() decorator here
def estimate_tau_spectrum_from_mu_slope(model_name: str,
                                       model: BaseRNN,
                                       Xdg_cpu: torch.Tensor,
                                       device: torch.device,
                                       diag_batch_size: int,
                                       fit_lags: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    For each neuron q:
      f_q(ell) = E_{seq,t} |mu_tl_q(t, ell)|   <-- Note: envelope f is derived from mu_tl
      log f_q(ell) ~ a_q - mu_bar_q * ell
      tau_q = 1/mu_bar_q
    """
    model.eval()
    fit_lags = np.asarray(fit_lags, dtype=int)
    fit_lags = np.unique(fit_lags[(fit_lags > 0)])
    if fit_lags.size < 4:
        raise ValueError("fit_lags too small; need >=4 distinct lags.")

    Btot, T, _ = Xdg_cpu.shape
    bs = int(diag_batch_size)
    nb = max(1, math.ceil(Btot / bs))

    H = None
    sum_f = None  # (L,H) tracking the envelope f
    n_seq = 0

    for bi in range(nb):
        lo = bi * bs
        hi = min(Btot, (bi + 1) * bs)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)

        # [NEW FIX A] Apply local scoping for torch.no_grad
        with torch.no_grad():
            _, _, g = model.forward_with_intermediates(xb)

            if model_name == "gru":
                leak = g["leak"]
                rdiag = g["rdiag"]
                cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)

                r = torch.clamp(g["r"], 1e-12, 1.0)
                cs_log_r, _ = precompute_prefix_sums(r, torch.zeros_like(r))
                eta = torch.clamp(leak * r, 1e-12, 1.0)
                cs_log_eta, _ = precompute_prefix_sums(eta, torch.zeros_like(eta))

                if H is None:
                    H = int(leak.shape[-1])
                    sum_f = np.zeros((fit_lags.size, H), dtype=np.float64)

                for j, ell in enumerate(fit_lags):
                    ell = int(ell)
                    gamma = mu_diag_product_first_order(cs_log, cs_ratio, leak, rdiag, ell, out_dtype=leak.dtype)
                    rho0  = prod_from_prefix(cs_log_r, ell, out_dtype=leak.dtype)
                    eta0  = prod_from_prefix(cs_log_eta, ell, out_dtype=leak.dtype)

                    # [NEW FIX C] Using mu_tl to represent the Jacobian path sum
                    mu_tl = (gamma + rho0 + eta0)   # (B, T-ell+1, H)

                    if mu_tl.numel() == 0:
                        continue

                    abs_f = torch.abs(mu_tl).double()   # envelope proxy f
                    f_bh = abs_f.mean(dim=1)            # (B,H)
                    sum_f[j, :] += f_bh.sum(dim=0).detach().cpu().numpy()

            elif model_name == "lstm":
                leak = g["leak"]     # f_t
                rdiag = g["rdiag"]   # diagC_t * e_{t-1}
                e = g["e"]           # e_t
                cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)

                if H is None:
                    H = int(leak.shape[-1])
                    sum_f = np.zeros((fit_lags.size, H), dtype=np.float64)

                for j, ell in enumerate(fit_lags):
                    ell = int(ell)
                    base = mu_diag_product_first_order(cs_log, cs_ratio, leak, rdiag, ell, out_dtype=leak.dtype)
                    e_end = e[:, (ell - 1):, :]   # aligns with T-ell+1

                    # [NEW FIX C] Using mu_tl to represent the Jacobian path sum
                    mu_tl = (base * e_end)

                    if mu_tl.numel() == 0:
                        continue

                    abs_f = torch.abs(mu_tl).double()   # envelope proxy f
                    f_bh = abs_f.mean(dim=1)
                    sum_f[j, :] += f_bh.sum(dim=0).detach().cpu().numpy()

            else:
                raise ValueError("estimate_tau_spectrum_from_mu_slope supports only gru/lstm")

        n_seq += int(xb.shape[0])
        del xb, g

    assert sum_f is not None and H is not None

    f = sum_f / max(1, n_seq)       # (L,H) envelope f
    log_f = np.log(f + 1e-30)

    ells = fit_lags.astype(np.float64)
    A = np.vstack([np.ones_like(ells), ells]).T

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
        mu_q = max(1e-12, -b_q)
        mu_bar[q] = mu_q

        yhat = (A[mask] @ coeff)
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
# Optional final-only macro envelope (for plotting)
# ============================================================

# [NEW FIX A] Removed @torch.no_grad() decorator here
def compute_macro_envelope_abs_mu(model_name: str,
                                  model: BaseRNN,
                                  Xdg_cpu: torch.Tensor,
                                  device: torch.device,
                                  ells: np.ndarray,
                                  diag_batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Macro envelope proxy:
      f(ell) = E_{seq,t} mean_q |mu_tl_q(t,ell)|   <-- Note: envelope f is derived from mu_tl

    Returns:
      f_mean[ell], log_f_mean[ell]
    """
    model.eval()
    Btot, T, _ = Xdg_cpu.shape
    bs = int(diag_batch_size)
    nb = max(1, math.ceil(Btot / bs))

    sum_f = np.zeros(len(ells), dtype=np.float64)
    sum_log_f = np.zeros(len(ells), dtype=np.float64)
    count_seq = 0

    for bi in range(nb):
        lo = bi * bs
        hi = min(Btot, (bi + 1) * bs)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)

        # [NEW FIX A] Apply local scoping for torch.no_grad
        with torch.no_grad():
            _, _, g = model.forward_with_intermediates(xb)

            if model_name == "gru":
                leak = g["leak"]
                rdiag = g["rdiag"]
                cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)

                r = torch.clamp(g["r"], 1e-12, 1.0)
                cs_log_r, _ = precompute_prefix_sums(r, torch.zeros_like(r))
                eta = torch.clamp(leak * r, 1e-12, 1.0)
                cs_log_eta, _ = precompute_prefix_sums(eta, torch.zeros_like(eta))

                for j, ell in enumerate(ells):
                    ell = int(ell)
                    gamma = mu_diag_product_first_order(cs_log, cs_ratio, leak, rdiag, ell, out_dtype=leak.dtype)
                    rho0  = prod_from_prefix(cs_log_r, ell, out_dtype=leak.dtype)
                    eta0  = prod_from_prefix(cs_log_eta, ell, out_dtype=leak.dtype)

                    # [NEW FIX C] Using mu_tl to represent the Jacobian path sum
                    mu_tl = (gamma + rho0 + eta0)

                    if mu_tl.numel() == 0:
                        continue

                    abs_f = torch.abs(mu_tl).double()   # envelope proxy f
                    f_per_seq = abs_f.mean(dim=2).mean(dim=1)  # (B,) mean over start times
                    sum_f[j] += float(f_per_seq.sum().item())
                    sum_log_f[j] += float(torch.log(f_per_seq.clamp(min=1e-30)).sum().item())

            elif model_name == "lstm":
                leak = g["leak"]
                rdiag = g["rdiag"]
                e = g["e"]
                cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)

                for j, ell in enumerate(ells):
                    ell = int(ell)
                    base = mu_diag_product_first_order(cs_log, cs_ratio, leak, rdiag, ell, out_dtype=leak.dtype)
                    e_end = e[:, (ell - 1):, :]

                    # [NEW FIX C] Using mu_tl to represent the Jacobian path sum
                    mu_tl = (base * e_end)

                    if mu_tl.numel() == 0:
                        continue

                    abs_f = torch.abs(mu_tl).double()   # envelope proxy f
                    f_per_seq = abs_f.mean(dim=2).mean(dim=1)
                    sum_f[j] += float(f_per_seq.sum().item())
                    sum_log_f[j] += float(torch.log(f_per_seq.clamp(min=1e-30)).sum().item())

            else:
                raise ValueError("compute_macro_envelope_abs_mu supports only gru/lstm")

        count_seq += int(xb.shape[0])
        del xb, g

    f_mean = sum_f / max(1, count_seq)
    log_f_mean = sum_log_f / max(1, count_seq)
    return f_mean.astype(np.float64), log_f_mean.astype(np.float64)


# ============================================================
# Training with checkpoint diagnostics
# ============================================================

def _init_learning_curve_csv(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])

def _append_learning_curve_csv(path: str, epoch: int, loss: float):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([int(epoch), float(loss)])


def run_checkpoint_diagnostics(args,
                              model: BaseRNN,
                              model_name: str,
                              mdir: str,
                              epoch: int,
                              Xtr_cpu: torch.Tensor,
                              Ytr_cpu: torch.Tensor,
                              Xdg_cpu: torch.Tensor,
                              device: torch.device,
                              fit_lags: np.ndarray) -> Dict:
    ckpt_tag = f"ckpt_{int(epoch):04d}"

    # --- alpha (M1: K fixed directions, median aggregation) ---
    K = int(getattr(args, 'alpha_n_directions', 5))
    all_alpha_hats = []
    all_sigma_hats = []
    all_reliables = []
    sample_rows = []
    sid = 0
    for k_dir in range(K):
        dir_seed = int(args.w_seed) + k_dir
        a_info_k, a_samples_k = estimate_alpha_from_minibatch_gradients(
            model, Xtr_cpu, Ytr_cpu, device=device,
            batch_size=min(int(args.batch_size), int(args.alpha_grad_batch_size)),
            n_grad_batches=int(args.alpha_n_grad_batches_ckpt),
            w_seed=dir_seed,
            grad_clip=float(args.grad_clip) if args.alpha_use_grad_clip else 0.0,
            alpha_method=str(getattr(args, 'alpha_method', 'ecf')),
            min_samples_alpha=int(getattr(args, 'min_samples_alpha', 500))
        )
        all_alpha_hats.append(a_info_k["alpha_hat"])
        all_sigma_hats.append(a_info_k["sigma_alpha_hat"])
        all_reliables.append(a_info_k.get("alpha_reliable", 1))
        for v in a_samples_k:
            sample_rows.append([sid, k_dir, float(v)])
            sid += 1

    alpha_hat_median = float(np.median(all_alpha_hats))
    sigma_hat_median = float(np.median(all_sigma_hats))
    alpha_hat_std = float(np.std(all_alpha_hats, ddof=1)) if K > 1 else 0.0
    alpha_hat_se = float(alpha_hat_std / math.sqrt(K)) if K > 1 else 0.0
    alpha_reliable = int(all(r == 1 for r in all_reliables))
    alpha_method_str = str(getattr(args, 'alpha_method', 'ecf'))
    alpha_info = {
        "alpha_hat": alpha_hat_median,
        "sigma_alpha_hat": sigma_hat_median,
        "alpha_hat_std": alpha_hat_std,
        "alpha_hat_se": alpha_hat_se,
        "alpha_hat_per_dir": all_alpha_hats,
        "alpha_reliable": alpha_reliable,
        "alpha_method": alpha_method_str,
        "grad_proj_mean": float(np.mean([r[2] for r in sample_rows])),
        "grad_proj_std": float(np.std([r[2] for r in sample_rows])),
        "n_samples": len(sample_rows),
        "n_directions": K,
    }
    alpha_dir = os.path.join(mdir, "checkpoint_alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    write_csv(
        os.path.join(alpha_dir, f"{ckpt_tag}_alpha_grad_samples.csv"),
        ["sample_id", "direction_id", "grad_projection"],
        sample_rows
    )
    with open(os.path.join(alpha_dir, f"{ckpt_tag}_alpha_grad.json"), "w") as jf:
        json.dump(alpha_info, jf, indent=2)

    # --- tau spectrum ---
    tau, tau_fit_info = estimate_tau_spectrum_from_mu_slope(
        model_name=model_name,
        model=model,
        Xdg_cpu=Xdg_cpu,
        device=device,
        diag_batch_size=int(args.diag_batch_size),
        fit_lags=fit_lags
    )

    tau_dir = os.path.join(mdir, "checkpoint_taus")
    os.makedirs(tau_dir, exist_ok=True)
    np.save(os.path.join(tau_dir, f"{ckpt_tag}_taus.npy"), tau)
    write_csv(
        os.path.join(tau_dir, f"{ckpt_tag}_taus.csv"),
        ["unit_id", "tau"],
        [[int(i), float(t)] for i, t in enumerate(tau)]
    )
    with open(os.path.join(tau_dir, f"{ckpt_tag}_tau_slope_fit_info.json"), "w") as jf:
        json.dump(tau_fit_info, jf, indent=2)

    # --- CCDF + tail fit ---
    tau_sorted, ccdf = compute_ccdf_curve(tau)

    if args.save_checkpoint_ccdf:
        ccdf_dir = os.path.join(mdir, "checkpoint_tau_ccdf")
        os.makedirs(ccdf_dir, exist_ok=True)
        write_csv(
            os.path.join(ccdf_dir, f"{ckpt_tag}_tau_ccdf.csv"),
            ["tau", "ccdf"],
            [[float(x), float(y)] for x, y in zip(tau_sorted, ccdf)]
        )

    tail = fit_tau_ccdf_powerlaw(tau, qmin=float(args.tau_ccdf_qmin), qmax=float(args.tau_ccdf_qmax))
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

    # --- Envelope-β consistency diagnostic ---
    env_info = envelope_beta_from_tau_spectrum(tau)

    row = {
        "epoch": int(epoch),
        "alpha_hat": float(alpha_info["alpha_hat"]),
        "sigma_alpha_hat": float(alpha_info["sigma_alpha_hat"]),
        "alpha_hat_std": float(alpha_info.get("alpha_hat_std", 0.0)),
        "alpha_hat_se": float(alpha_info.get("alpha_hat_se", 0.0)),
        "beta_hat": float(tail["beta_hat"]),
        "beta_r2": float(tail["beta_r2"]),
        "tau_mean": float(tail["tau_mean"]),
        "tau_q90": float(tail["tau_q90"]),
        "tau_q99": float(tail["tau_q99"]),
        "tau_fit_r2_mean": float(tau_fit_info.get("r2_mean", float("nan"))),
        "tau_fit_n_valid": int(tau_fit_info.get("n_valid_neurons", 0)),
        "fit_lags_min": int(np.min(fit_lags)),
        "fit_lags_max": int(np.max(fit_lags)),
        "alpha_reliable": int(alpha_info.get("alpha_reliable", 1)),
        "alpha_method": str(alpha_info.get("alpha_method", "ecf")),
        "n_samples": int(alpha_info.get("n_samples", 0)),
        "beta_env": float(env_info["beta_env"]),
        "beta_env_r2": float(env_info["beta_env_r2"]),
    }
    return row


def train_with_phase_tracking(args,
                              model: BaseRNN,
                              model_name: str,
                              mdir: str,
                              Xtr_cpu: torch.Tensor,
                              Ytr_cpu: torch.Tensor,
                              Xdg_cpu: torch.Tensor,
                              device: torch.device,
                              fit_lags: np.ndarray) -> None:
    # optimizer
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    if args.orth_init:
        model.apply_orthogonal()

    # learning curve
    lc_csv = os.path.join(mdir, f"{model_name}_learning_curve.csv")
    _init_learning_curve_csv(lc_csv)

    # phase trajectory
    traj_csv = os.path.join(mdir, "phase_trajectory.csv")
    _traj_cols = ["epoch", "alpha_hat", "sigma_alpha_hat", "alpha_hat_std", "alpha_hat_se",
                  "beta_hat", "beta_r2",
                  "tau_mean", "tau_q90", "tau_q99",
                  "tau_fit_r2_mean", "tau_fit_n_valid",
                  "fit_lags_min", "fit_lags_max",
                  "alpha_reliable", "alpha_method", "n_samples",
                  "beta_env", "beta_env_r2"]
    write_csv(traj_csv, _traj_cols, [])

    Btot = int(Xtr_cpu.shape[0])
    bs = int(args.batch_size)
    nb = max(1, math.ceil(Btot / bs))
    log_every = max(1, int(args.epochs) // 5)

    ckpt_every = int(max(1, args.checkpoint_every))
    ckpt_epochs = set([1, int(args.epochs)] + list(range(ckpt_every, int(args.epochs) + 1, ckpt_every)))

    log(f"[train:{model_name}] start epochs={args.epochs} bs={bs} opt={args.optimizer} lr={args.lr}")

    nan_halt = False
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(Btot)  # CPU
        loss_sum = 0.0
        n_seen = 0

        for bi in range(nb):
            lo = bi * bs
            hi = min(Btot, (bi + 1) * bs)
            idx = perm[lo:hi]

            xb = Xtr_cpu[idx].to(device, non_blocking=True)
            yb = Ytr_cpu[idx].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
            loss = F.mse_loss(yhat, yb)

            if not torch.isfinite(loss):
                log(f"[train:{model_name}] NaN/Inf loss detected at epoch={ep}, batch={bi}. Halting training.")
                nan_halt = True
                del xb, yb, yhat, loss
                break

            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()

            loss_sum += float(loss.item()) * int(hi - lo)
            n_seen += int(hi - lo)
            del xb, yb, yhat, loss

        if nan_halt:
            break

        train_loss_epoch = loss_sum / max(1, n_seen)
        _append_learning_curve_csv(lc_csv, ep, train_loss_epoch)

        if (ep == 1) or (ep == int(args.epochs)) or (ep % log_every == 0):
            log(f"[train:{model_name}] ep {ep}/{args.epochs} avg_loss={train_loss_epoch:.4g}")

        if ep in ckpt_epochs:
            log(f"[ckpt:{model_name}] diagnostics at epoch={ep} ...")
            row = run_checkpoint_diagnostics(
                args, model, model_name, mdir, ep,
                Xtr_cpu, Ytr_cpu, Xdg_cpu, device=device, fit_lags=fit_lags
            )
            append_csv_row(traj_csv, [row[k] for k in _traj_cols])

    log(f"[train:{model_name}] done")


def run_for_model(args, model_name: str,
                  outdir: str,
                  Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                  Xdg_cpu: torch.Tensor,
                  device: torch.device,
                  ells: np.ndarray,
                  fit_lags: np.ndarray) -> Dict:
    mdir = os.path.join(outdir, model_name)
    os.makedirs(mdir, exist_ok=True)

    model = build_model(model_name, args.D, args.H, ln=args.layernorm).to(device)

    train_with_phase_tracking(
        args, model, model_name, mdir,
        Xtr_cpu, Ytr_cpu, Xdg_cpu,
        device=device, fit_lags=fit_lags
    )

    if args.save_final_envelope:
        log(f"[final:{model_name}] computing macro envelope (abs, first-order corrected) ...")
        f_mean, log_f_mean = compute_macro_envelope_abs_mu(
            model_name=model_name,
            model=model,
            Xdg_cpu=Xdg_cpu,
            device=device,
            ells=ells,
            diag_batch_size=int(args.diag_batch_size)
        )
        write_csv(
            os.path.join(mdir, f"{model_name}_envelope.csv"),
            ["ell", "f_mean", "log_f_mean"],
            [[int(e), float(fv), float(lv)] for e, fv, lv in zip(ells, f_mean, log_f_mean)]
        )
        fit = fit_log_envelope_exp_and_power(ells, log_f_mean)
        with open(os.path.join(mdir, f"{model_name}_envelope_fit.json"), "w") as jf:
            json.dump(fit, jf, indent=2)

        log_exp, log_pow = eval_envelope_fit_curves(ells, fit)
        write_csv(
            os.path.join(mdir, f"{model_name}_envelope_fit_curves.csv"),
            ["ell", "log_f_data", "log_f_exp_fit", "log_f_power_fit"],
            [[int(e), float(ld), float(le), float(lp)]
             for e, ld, le, lp in zip(ells, log_f_mean, log_exp, log_pow)]
        )

    return {"ok": True}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--models", type=str, default="gru,lstm")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--w_seed", type=int, default=41)

    p.add_argument("--Nseq_train", type=int, default=8000)
    p.add_argument("--Nseq_diag", type=int, default=8000)

    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--H", type=int, default=512)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd_momentum"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--orth_init", action="store_true")
    p.add_argument("--layernorm", action="store_true")

    p.add_argument("--lag_min", type=int, default=4)
    p.add_argument("--lag_max", type=int, default=256)
    p.add_argument("--num_lags", type=int, default=128)

    p.add_argument("--task_lags", type=str, default="32,64,128,192,256")
    p.add_argument("--task_coeffs", type=str, default="0.6,0.5,0.4,0.32,0.26")
    p.add_argument("--noise_std", type=float, default=0.3)

    p.add_argument("--diag_batch_size", type=int, default=256)
    p.add_argument("--checkpoint_every", type=int, default=50)

    p.add_argument("--alpha_n_grad_batches_ckpt", type=int, default=256)
    p.add_argument("--alpha_grad_batch_size", type=int, default=256)
    p.add_argument("--alpha_use_grad_clip", action="store_true")
    p.add_argument("--alpha_n_directions", type=int, default=5,
                    help="K fixed random projection directions for alpha estimation (M1).")
    p.add_argument("--alpha_method", type=str, default="ecf",
                    choices=["mcculloch", "ecf"],
                    help="Alpha estimation method: ecf (empirical char. function, default) or mcculloch (quantile-ratio).")
    p.add_argument("--min_samples_alpha", type=int, default=500,
                    help="Minimum samples for reliable alpha estimation.")

    p.add_argument("--tau_fit_lag_min", type=int, default=64)
    p.add_argument("--tau_fit_lag_max", type=int, default=256)
    p.add_argument("--tau_fit_num_lags", type=int, default=24)

    p.add_argument("--tau_ccdf_qmin", type=float, default=0.75)
    p.add_argument("--tau_ccdf_qmax", type=float, default=0.995)

    p.add_argument("--save_checkpoint_ccdf", action="store_true")
    p.add_argument("--save_final_envelope", action="store_true")

    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "mps", "cuda"])

    args = p.parse_args()
    args.task_lags = [int(s) for s in args.task_lags.split(",") if s.strip()]
    args.task_coeffs = [float(s) for s in args.task_coeffs.split(",") if s.strip()]
    assert len(args.task_lags) == len(args.task_coeffs)
    return args

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = resolve_device(args.device)
    log(f"Running on device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")

    ells = np.linspace(args.lag_min, args.lag_max, args.num_lags, dtype=int)
    ells = np.unique(np.clip(ells, 1, args.T - 1)).astype(int)

    fit_lags = np.linspace(args.tau_fit_lag_min, args.tau_fit_lag_max, args.tau_fit_num_lags, dtype=int)
    fit_lags = np.unique(np.clip(fit_lags, 1, args.T - 1)).astype(int)

    Xtr_cpu, Ytr_cpu, u_vec = make_dataset_cpu(
        args.Nseq_train, args.T, args.D,
        args.task_lags, args.task_coeffs, args.noise_std, u_vec=None
    )
    Xdg_cpu, _, _ = make_dataset_cpu(
        args.Nseq_diag, args.T, args.D,
        args.task_lags, args.task_coeffs, args.noise_std, u_vec=u_vec
    )

    if device.type == "cuda":
        Xtr_cpu = Xtr_cpu.pin_memory()
        Ytr_cpu = Ytr_cpu.pin_memory()
        Xdg_cpu = Xdg_cpu.pin_memory()

    with open(os.path.join(args.outdir, "cli_args.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)
    with open(os.path.join(args.outdir, "lag_grid.json"), "w") as jf:
        json.dump({"ells": ells.tolist(), "tau_fit_lags": fit_lags.tolist()}, jf, indent=2)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for mname in models:
        if mname not in ["gru", "lstm"]:
            raise ValueError(f"This runner supports only gru,lstm (got {mname})")
        log(f"[run] model={mname}")
        run_for_model(args, mname, args.outdir,
                      Xtr_cpu, Ytr_cpu, Xdg_cpu,
                      device=device, ells=ells, fit_lags=fit_lags)

    log("Done.")

if __name__ == "__main__":
    main()