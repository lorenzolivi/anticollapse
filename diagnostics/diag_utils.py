#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for anticollapse diagnostic experiments.
=========================================================

Contains:
  - Model definitions (ConstGate, DiagGate, GRU, LSTM)
  - Synthetic dataset generation
  - First-order diagonal expansion (prefix-sum mu_tl computation)
  - Per-neuron tau extraction from asymptotic slopes
  - Envelope construction and comparison metrics
  - Bridges to the main diagnostics pipeline for GELR/Rayleigh validation
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


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_main_diagnostics_module():
    """Load the root diagnostics.py so sidecar checks reuse main-pipeline logic."""
    global _MAIN_DIAGNOSTICS
    if _MAIN_DIAGNOSTICS is not None:
        return _MAIN_DIAGNOSTICS

    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

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
# Models
# ============================================================

class BaseRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, return_intermediates=True):
        return self.forward_with_intermediates(x, return_intermediates=return_intermediates)

    def apply_orthogonal(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
                if getattr(m, '_skip_orth', False):
                    continue
                nn.init.orthogonal_(m.weight)

    def neuron_param_indices(self, q: int) -> Dict[str, List[int]]:
        """
        Return dict mapping param_name -> list of flat indices
        that 'belong' to neuron q (row q for weight matrices, element q for biases).
        Subclasses should override for architecture-specific mapping.
        """
        raise NotImplementedError


class ConstGateRNN(BaseRNN):
    def __init__(self, D: int, H: int, s: float = 0.7):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.out = nn.Linear(H, 1)
        s = float(np.clip(s, 1e-6, 1.0 - 1e-6))
        self.register_buffer("s_const", torch.tensor(s, dtype=torch.float32))
        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        s = self.s_const
        ys = []
        hs = []
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            leaks, rdiags = [], []
        for t in range(T):
            h_prev = h
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            ys.append(self.out(h))
            hs.append(h)
            if return_intermediates:
                sH = s.expand(B, self.H)
                leak = torch.clamp(1 - sH, 1e-12, 1.0)
                tanh_prime = 1.0 - h_tilde ** 2
                rdiag = (sH * tanh_prime) * wh_diag.view(1, -1)
                leaks.append(leak)
                rdiags.append(rdiag)
        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, dim=1), {"leak": torch.stack(leaks, dim=1),
                                           "rdiag": torch.stack(rdiags, dim=1)}

    def neuron_param_indices(self, q):
        idx = {}
        # Wx.weight: (H, D) -> row q
        idx["Wx.weight"] = list(range(q * self.D, (q + 1) * self.D))
        # Wx.bias: (H,) -> element q
        idx["Wx.bias"] = [q]
        # Wh.weight: (H, H) -> row q
        idx["Wh.weight"] = list(range(q * self.H, (q + 1) * self.H))
        return idx


class DiagGateRNN(BaseRNN):
    def __init__(self, D: int, H: int, init_s: float = 0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.Ws = nn.Linear(D, H, bias=True)
        self.Us = nn.Linear(H, H, bias=False)
        self.Ws._skip_orth = True
        self.Us._skip_orth = True
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.Ws.weight)
        nn.init.zeros_(self.Us.weight)
        init_s = float(np.clip(init_s, 1e-6, 1.0 - 1e-6))
        gate_bias = float(np.log(init_s / (1.0 - init_s)))
        nn.init.constant_(self.Ws.bias, gate_bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        ys = []
        hs = []
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_diag = torch.diagonal(self.Us.weight, 0)
            leaks, rdiags = [], []
        for t in range(T):
            h_prev = h
            a_s = self.Ws(x[:, t]) + self.Us(h_prev)
            s = torch.sigmoid(a_s)
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            ys.append(self.out(h))
            hs.append(h)
            if return_intermediates:
                leak = torch.clamp(1 - s, 1e-12, 1.0)
                tanh_prime = 1.0 - h_tilde ** 2
                s_prime = s * (1 - s)
                rdiag_gate = (h_tilde - h_prev) * (s_prime * us_diag.view(1, -1))
                rdiag_rec = (s * tanh_prime) * wh_diag.view(1, -1)
                leaks.append(leak)
                rdiags.append(rdiag_gate + rdiag_rec)
        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, dim=1), {"leak": torch.stack(leaks, dim=1),
                                           "rdiag": torch.stack(rdiags, dim=1)}

    def neuron_param_indices(self, q):
        idx = {}
        idx["Wx.weight"] = list(range(q * self.D, (q + 1) * self.D))
        idx["Wx.bias"] = [q]
        idx["Wh.weight"] = list(range(q * self.H, (q + 1) * self.H))
        idx["Ws.weight"] = list(range(q * self.D, (q + 1) * self.D))
        idx["Ws.bias"] = [q]
        idx["Us.weight"] = list(range(q * self.H, (q + 1) * self.H))
        return idx


class GRUCustom(BaseRNN):
    def __init__(self, D: int, H: int):
        super().__init__()
        self.D, self.H = D, H
        self.Wz = nn.Linear(D, H); self.Uz = nn.Linear(H, H, bias=False)
        self.Wr = nn.Linear(D, H); self.Ur = nn.Linear(H, H, bias=False)
        self.Wh = nn.Linear(D, H); self.Uh = nn.Linear(H, H, bias=False)
        self.out = nn.Linear(H, 1)
        for m in [self.Wz, self.Wr, self.Wh, self.out]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        ys = []
        hs = []
        if return_intermediates:
            diagUz = torch.diagonal(self.Uz.weight, 0)
            diagUr = torch.diagonal(self.Ur.weight, 0)
            diagUh = torch.diagonal(self.Uh.weight, 0)
            leaks, rdiags, rseq = [], [], []
        for t in range(T):
            h_prev = h
            z = torch.sigmoid(self.Wz(x[:, t]) + self.Uz(h_prev))
            r = torch.sigmoid(self.Wr(x[:, t]) + self.Ur(h_prev))
            g = torch.tanh(self.Wh(x[:, t]) + self.Uh(r * h_prev))
            h = (1.0 - z) * h_prev + z * g
            ys.append(self.out(h))
            hs.append(h)
            if return_intermediates:
                leak = torch.clamp(1.0 - z, 1e-12, 1.0)
                zprime = z * (1.0 - z)
                gprime = 1.0 - g ** 2
                term1 = (g - h_prev) * zprime * diagUz.view(1, -1)
                term2 = z * gprime * diagUh.view(1, -1) * r
                term3 = z * gprime * diagUh.view(1, -1) * h_prev * (r * (1 - r)) * diagUr.view(1, -1)
                leaks.append(leak)
                rdiags.append(term1 + term2 + term3)
                rseq.append(r)
        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, dim=1), {"leak": torch.stack(leaks, dim=1),
                                           "rdiag": torch.stack(rdiags, dim=1),
                                           "r": torch.stack(rseq, dim=1)}

    def neuron_param_indices(self, q):
        idx = {}
        for name in ["Wz", "Wr", "Wh"]:
            idx[f"{name}.weight"] = list(range(q * self.D, (q + 1) * self.D))
            idx[f"{name}.bias"] = [q]
        for name in ["Uz", "Ur", "Uh"]:
            idx[f"{name}.weight"] = list(range(q * self.H, (q + 1) * self.H))
        return idx


class LSTMCustom(BaseRNN):
    def __init__(self, D: int, H: int):
        super().__init__()
        self.D, self.H = D, H
        self.Wi = nn.Linear(D, H); self.Ui = nn.Linear(H, H, bias=False)
        self.Wf = nn.Linear(D, H); self.Uf = nn.Linear(H, H, bias=False)
        self.Wo = nn.Linear(D, H); self.Uo = nn.Linear(H, H, bias=False)
        self.Wg = nn.Linear(D, H); self.Ug = nn.Linear(H, H, bias=False)
        self.out = nn.Linear(H, 1)
        for m in [self.Wi, self.Wf, self.Wo, self.Wg, self.out]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        c = torch.zeros(B, self.H, device=x.device)
        ys = []
        hs = []
        if return_intermediates:
            diagUf = torch.diagonal(self.Uf.weight, 0)
            diagUg = torch.diagonal(self.Ug.weight, 0)
            diagUi = torch.diagonal(self.Ui.weight, 0)
            leaks, rdiags, eseq = [], [], []
        for t in range(T):
            h_prev, c_prev = h, c
            i = torch.sigmoid(self.Wi(x[:, t]) + self.Ui(h_prev))
            f = torch.sigmoid(self.Wf(x[:, t]) + self.Uf(h_prev))
            o = torch.sigmoid(self.Wo(x[:, t]) + self.Uo(h_prev))
            g = torch.tanh(self.Wg(x[:, t]) + self.Ug(h_prev))
            c = f * c_prev + i * g
            tanh_c = torch.tanh(c)
            h = o * tanh_c
            ys.append(self.out(h))
            hs.append(h)
            if return_intermediates:
                e = o * (1.0 - tanh_c ** 2)
                leak = torch.clamp(f, 1e-12, 1.0)
                diagC = (c_prev * (f * (1 - f))) * diagUf.view(1, -1) \
                        + (i * (1 - g ** 2)) * diagUg.view(1, -1) \
                        + (g * (i * (1 - i))) * diagUi.view(1, -1)
                e_prev = torch.zeros_like(e) if t == 0 else eseq[-1]
                rdiag_t = diagC * e_prev
                eseq.append(e)
                leaks.append(leak)
                rdiags.append(rdiag_t)
        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, dim=1), {"leak": torch.stack(leaks, dim=1),
                                           "rdiag": torch.stack(rdiags, dim=1),
                                           "e": torch.stack(eseq, dim=1)}

    def neuron_param_indices(self, q):
        idx = {}
        for name in ["Wi", "Wf", "Wo", "Wg"]:
            idx[f"{name}.weight"] = list(range(q * self.D, (q + 1) * self.D))
            idx[f"{name}.bias"] = [q]
        for name in ["Ui", "Uf", "Uo", "Ug"]:
            idx[f"{name}.weight"] = list(range(q * self.H, (q + 1) * self.H))
        return idx


def build_model(arch: str, D: int, H: int, const_s: float = 0.005) -> BaseRNN:
    arch = arch.lower()
    if arch == "const":
        return ConstGateRNN(D, H, s=const_s)
    if arch == "diag":
        return DiagGateRNN(D, H, init_s=const_s)
    if arch == "gru":
        return GRUCustom(D, H)
    if arch == "lstm":
        return LSTMCustom(D, H)
    raise ValueError(f"Unknown arch: {arch}")


# ============================================================
# Optimizer construction
# ============================================================

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
# First-order diagonal expansion (prefix sums)
# ============================================================

@torch.no_grad()
def precompute_prefix_sums(leak: torch.Tensor, rdiag: torch.Tensor):
    """
    leak:  (B, T, H) — positive gate-product factors
    rdiag: (B, T, H) — first-order diagonal correction
    Returns cs_log (B, T+1, H), cs_ratio (B, T+1, H).
    """
    B, T, H = leak.shape
    device = leak.device
    leak64 = torch.clamp(leak.double(), 1e-12, 1.0)
    log_leak = torch.log(leak64)
    cs_log = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
    cs_log[:, 1:, :] = torch.cumsum(log_leak, dim=1)
    ratio = (rdiag.double() / leak64)
    cs_ratio = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
    cs_ratio[:, 1:, :] = torch.cumsum(ratio, dim=1)
    return cs_log, cs_ratio


@torch.no_grad()
def mu_diag_product_first_order(cs_log, cs_ratio, leak, rdiag, ell, out_dtype):
    """
    First-order diagonal expansion:
      mu_tl = mu0 + mu1,  mu0 = prod(leak), mu1 = mu0 * sum(rdiag/leak)
    Returns (B, T-ell+1, H).
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
    if ell == 1:
        return (leak + rdiag).to(out_dtype)
    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]
    mu0_64 = torch.exp(log_prod)
    sum_ratio = cs_ratio[:, ell:(T + 1), :] - cs_ratio[:, 0:(T - ell + 1), :]
    mu1_64 = mu0_64 * sum_ratio
    return (mu0_64 + mu1_64).to(out_dtype)


# ============================================================
# Envelope and per-neuron rate computation
# ============================================================

@torch.no_grad()
def compute_per_neuron_envelope(model: BaseRNN, X: torch.Tensor,
                                device: torch.device,
                                lags: np.ndarray,
                                batch_size: int = 32) -> np.ndarray:
    """
    Compute per-neuron, per-lag mean |mu^(q)_{t,ell}|.

    Returns f_q: (n_lags, H) array where f_q[j, q] = E_{seq,t} |mu^(q)_{t, lags[j]}|.
    This uses the first-order diagonal expansion and multiplies by global lr=1
    (transport factor only — caller handles the Λ weighting).
    """
    model.eval()
    lags = np.asarray(lags, dtype=int)
    Nseq = X.shape[0]
    H_model = None
    sum_f = None
    n_seq = 0

    for i in range(0, Nseq, batch_size):
        xb = X[i:i + batch_size].to(device)
        _, _, g = model(xb, return_intermediates=True)
        leak = g["leak"]
        rdiag = g["rdiag"]
        if H_model is None:
            H_model = leak.shape[-1]
            sum_f = np.zeros((len(lags), H_model), dtype=np.float64)
        cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)
        for j, ell in enumerate(lags):
            mu_tl = mu_diag_product_first_order(
                cs_log, cs_ratio, leak, rdiag, int(ell), leak.dtype)
            if mu_tl.numel() == 0:
                continue
            # |mu^(q)| averaged over start times → (B, H)
            abs_mu_bh = torch.abs(mu_tl).double().mean(dim=1)
            sum_f[j] += abs_mu_bh.sum(dim=0).cpu().numpy()
        n_seq += xb.shape[0]

    return sum_f / max(n_seq, 1)  # (n_lags, H)


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
                         fit_lag_min: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each neuron q, fit log|f_q(ell)| ~ a_q - mu_bar_q * ell
    over lags >= fit_lag_min.

    Returns:
      tau:   (H,)  — time scale 1/mu_bar_q
      r2:    (H,)  — R^2 of the linear fit
      mu_bar:(H,)  — asymptotic decay rate
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
        y = np.log(f_q[mask, q] + 1e-30)
        finite = np.isfinite(y)
        if finite.sum() < 4:
            continue
        coeff, _, _, _ = np.linalg.lstsq(A[finite], y[finite], rcond=None)
        b_q = float(coeff[1])
        mu_q = max(1e-12, -b_q)
        mu_bar[q] = mu_q
        tau[q] = 1.0 / mu_q

        yhat = A[finite] @ coeff
        ss_res = np.sum((y[finite] - yhat) ** 2)
        ss_tot = np.sum((y[finite] - y[finite].mean()) ** 2) + 1e-12
        r2[q] = 1.0 - ss_res / ss_tot

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
    nb = max(1, math.ceil(Btot / bs))

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
# Approximate per-neuron Lambda from optimizer state (GELR)
# ============================================================

def compute_approx_lambda(model: BaseRNN, optimizer: torch.optim.Optimizer,
                          H: int) -> Optional[np.ndarray]:
    """
    Compute approximate per-neuron Rayleigh projection Lambda^(q)
    from the optimizer's second-moment state.

    For Adam/RMSprop: Lambda^(q) ≈ mean of lr / (sqrt(v_entry) + eps)
    over all parameter entries belonging to neuron q.
    For SGD: returns None (Lambda = lr for all neurons).

    Returns: (H,) array of approximate Lambda^(q), or None for SGD.
    """
    opt_state = optimizer.state
    opt_defaults = optimizer.defaults

    # Check optimizer type
    is_adam = isinstance(optimizer, torch.optim.Adam)
    is_rmsprop = isinstance(optimizer, torch.optim.RMSprop)
    if not (is_adam or is_rmsprop):
        return None

    lr = opt_defaults["lr"]
    eps = opt_defaults.get("eps", 1e-8)

    # For Adam, get beta2 and step for bias correction
    beta2 = opt_defaults.get("betas", (0.9, 0.999))[1] if is_adam else None

    lambda_q = np.zeros(H, dtype=np.float64)
    count_q = np.zeros(H, dtype=np.float64)

    # Build param name -> param mapping
    param_name_map = {id(p): name for name, p in model.named_parameters()}

    for group in optimizer.param_groups:
        group_lr = group.get("lr", lr)
        group_eps = group.get("eps", eps)
        group_beta2 = group.get("betas", (0.9, 0.999))[1] if is_adam else None

        for p in group["params"]:
            if p not in opt_state or not p.requires_grad:
                continue

            state = opt_state[p]
            pname = param_name_map.get(id(p), None)
            if pname is None:
                continue

            # Get second moment estimate
            if is_adam and "exp_avg_sq" in state:
                v = state["exp_avg_sq"].detach().cpu().numpy().flatten()
                step = state.get("step", 1)
                if isinstance(step, torch.Tensor):
                    step = step.item()
                # Bias correction
                bc = 1.0 - group_beta2 ** step
                v_corrected = v / max(bc, 1e-12)
                eff_rates = group_lr / (np.sqrt(v_corrected) + group_eps)
            elif is_rmsprop and "square_avg" in state:
                v = state["square_avg"].detach().cpu().numpy().flatten()
                eff_rates = group_lr / (np.sqrt(v) + group_eps)
            else:
                continue

            # Map parameter entries to neurons
            # Weight matrices: shape (out_features, in_features)
            # Row q -> neuron q
            p_shape = tuple(p.shape)
            if pname.endswith(".weight") and len(p_shape) == 2:
                out_f, in_f = p_shape
                if out_f == H:
                    # Row q belongs to neuron q
                    eff_rates_2d = eff_rates.reshape(out_f, in_f)
                    for q in range(H):
                        lambda_q[q] += eff_rates_2d[q].sum()
                        count_q[q] += in_f
                elif out_f == 1 and in_f == H:
                    # Output layer: column q -> neuron q
                    for q in range(H):
                        lambda_q[q] += eff_rates[q]
                        count_q[q] += 1
            elif pname.endswith(".bias") and len(p_shape) == 1:
                if p_shape[0] == H:
                    for q in range(H):
                        lambda_q[q] += eff_rates[q]
                        count_q[q] += 1

    # Average
    count_q = np.maximum(count_q, 1)
    lambda_q = lambda_q / count_q
    return lambda_q


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
