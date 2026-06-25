#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared model definitions for all Anti-Collapse experiments.
============================================================

Architectures (increasing gating expressivity):
  1. ConstGate  — fixed scalar gate s (no learnable gate parameters)
  2. SharedGate — single scalar gate (shared across all neurons), input/hidden-dependent
  3. DiagGate   — per-neuron diagonal gate, input/hidden-dependent
  4. GRU        — standard GRU with update (z), reset (r), candidate (g)
  5. LSTM       — standard LSTM with input (i), forget (f), output (o), candidate (g)

All models return:
  y      : (B, T, 1)  predictions
  hseq   : (B, T, H)  hidden state sequence (or None)
  intermediates : dict with at minimum {"leak": (B,T,H), "rdiag": (B,T,H)}
                  plus model-specific tensors for transport factor computation.

Notation follows the companion papers:
  leak[t]  = zeroth-order diagonal factor at time t (the "carry" factor)
  rdiag[t] = first-order correction (derivative of nonlinearity × weight diagonal)
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ============================================================
# Utility
# ============================================================

def layernorm_if(enabled: bool, dim: int):
    return nn.LayerNorm(dim) if enabled else nn.Identity()


def _stable_sample_symmetric(
    alpha: float,
    scale: float,
    shape,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    clip: float = 0.0,
) -> torch.Tensor:
    """Draw symmetric alpha-stable noise with CMS sampling.

    When ``clip > 0`` the standardized draw is truncated to ``[-clip, clip]``
    (in dispersion units) before scaling. A symmetric alpha-stable law with
    alpha < 2 has infinite variance and produces rare, arbitrarily large
    excursions; injected into the recurrent state these overflow it to
    NaN/Inf. Truncating bounds those catastrophic draws while preserving the
    heavy tail over the operative range, which is also consistent with the
    tempered/truncated-jump interpretation used by the generator model.
    """
    if scale <= 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    eps = 1e-37
    U = (torch.rand(shape, generator=generator, device=device, dtype=dtype) - 0.5) * math.pi
    W = -torch.log(torch.rand(shape, generator=generator, device=device, dtype=dtype).clamp_min(eps))
    if abs(float(alpha) - 1.0) < 1e-6:
        X = torch.tan(U)
    else:
        sin_aU = torch.sin(float(alpha) * U)
        cos_U = torch.cos(U).clamp_min(eps)
        cos_au = torch.cos((float(alpha) - 1.0) * U).clamp_min(eps)
        X = (sin_aU / cos_U.pow(1.0 / float(alpha))) * (cos_au / W).pow((1.0 - float(alpha)) / float(alpha))
    if float(clip) > 0.0:
        X = X.clamp(-float(clip), float(clip))
    return X.mul_(float(scale))


def _apply_state_increment_injection(
    h: torch.Tensor,
    h_prev: torch.Tensor,
    state_injection: dict | None,
) -> torch.Tensor:
    """Perturb selected hidden-state increment coordinates in-place logically.

    The injection is applied after the deterministic recurrent update.  Its
    dispersion is calibrated to the RMS increment on the selected coordinates,
    so the same operator can be used across ConstGate, SharedGate, and DiagGate.
    """
    if not state_injection:
        return h
    scale_multiplier = float(state_injection.get("scale_multiplier", 0.0))
    if scale_multiplier <= 0.0:
        return h
    generator = state_injection.get("generator")
    if generator is None:
        return h
    mask = state_injection.get("mask")
    if mask is None:
        return h
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=h.device)
    mask = mask.to(device=h.device, dtype=torch.bool)
    if mask.numel() != h.shape[-1] or not bool(mask.any()):
        return h

    delta = (h - h_prev)[:, mask]
    rms = torch.linalg.vector_norm(delta.detach()).item() / max(math.sqrt(delta.numel()), 1e-37)
    scale = scale_multiplier * float(rms)
    if scale <= 0.0:
        return h
    noise = _stable_sample_symmetric(
        alpha=float(state_injection.get("alpha", 1.6)),
        scale=scale,
        shape=delta.shape,
        device=h.device,
        dtype=h.dtype,
        generator=generator,
        clip=float(state_injection.get("noise_clip", 0.0)),
    )
    h = h.clone()
    h[:, mask] = h[:, mask] + noise
    return h


# ============================================================
# Base class
# ============================================================

class BaseRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gate_rescale=None, return_intermediates=True,
                state_injection=None):
        return self.forward_with_intermediates(
            x, gate_rescale=gate_rescale, return_intermediates=return_intermediates,
            state_injection=state_injection,
        )

    def apply_orthogonal(self):
        """Apply orthogonal init to all 2-D weight matrices (except those marked _skip_orth)."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
                if getattr(m, '_skip_orth', False):
                    continue
                nn.init.orthogonal_(m.weight)


# ============================================================
# ConstGate RNN
# ============================================================

class ConstGateRNN(BaseRNN):
    """
    Fixed scalar gate: h_t = (1-s) h_{t-1} + s tanh(Wx x_t + Wh h_{t-1}).
    No learnable gate parameters.
    """
    def __init__(self, D: int, H: int, s: float = 0.7, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)

        s = float(np.clip(s, 1e-6, 1.0 - 1e-6))
        self.register_buffer("s_const", torch.tensor(s, dtype=torch.float32))

        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None,
                                    return_intermediates=True, state_injection=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        s = self.s_const
        if gate_rescale is not None:
            s = torch.clamp(s * gate_rescale, 0.0, 1.0)

        ys = []
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            leaks, rdiags, hs = [], [], []

        for t in range(T):
            h_prev = h
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln(pre)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            h = _apply_state_increment_injection(h, h_prev, state_injection)
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                sH = s.expand(B, self.H)
                leak = torch.clamp(1 - sH, 1e-12, 1.0)
                tanh_prime = 1.0 - h_tilde ** 2
                rdiag = (sH * tanh_prime) * wh_diag.view(1, -1)
                hs.append(h)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"leak": leak, "rdiag": rdiag}


# ============================================================
# SharedGate RNN
# ============================================================

class SharedGateRNN(BaseRNN):
    """
    Scalar gate shared across neurons:
      s_t = σ(Ws x_t + Us h_{t-1} + b_s)  ∈ ℝ (broadcast to H)
      h_t = (1-s_t) h_{t-1} + s_t tanh(Wx x_t + Wh h_{t-1})
    """
    def __init__(self, D: int, H: int, ln: bool = False, init_s: float = 0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)

        self.Ws = nn.Linear(D, 1, bias=True)
        self.Us = nn.Linear(H, 1, bias=False)
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

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None,
                                    return_intermediates=True, state_injection=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        ys = []
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_vec = self.Us.weight.view(-1)
            leaks, rdiags, hs = [], [], []

        for t in range(T):
            h_prev = h
            a_s = self.Ws(x[:, t]) + self.Us(h_prev)
            s = torch.sigmoid(a_s)
            if gate_rescale is not None:
                s = torch.clamp(s * gate_rescale, 0.0, 1.0)

            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)

            sH = s.expand(B, self.H)
            h = (1 - sH) * h_prev + sH * h_tilde
            h = _apply_state_increment_injection(h, h_prev, state_injection)
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = torch.clamp(1 - sH, 1e-12, 1.0)
                tanh_prime = 1.0 - h_tilde ** 2
                s_prime = (s * (1 - s)).expand(B, self.H)
                rdiag_gate = (h_tilde - h_prev) * (s_prime * us_vec.view(1, -1))
                rdiag_rec = (sH * tanh_prime) * wh_diag.view(1, -1)
                rdiag = rdiag_gate + rdiag_rec
                hs.append(h)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"leak": leak, "rdiag": rdiag}


# ============================================================
# DiagGate RNN
# ============================================================

class DiagGateRNN(BaseRNN):
    """
    Per-neuron diagonal gate:
      s_t^(q) = σ(Ws x_t + Us h_{t-1} + b_s)_q  ∈ ℝ^H
      h_t = (1 - s_t) ⊙ h_{t-1} + s_t ⊙ tanh(Wx x_t + Wh h_{t-1})
    """
    def __init__(self, D: int, H: int, ln: bool = False, init_s: float = 0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)

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

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None,
                                    return_intermediates=True, state_injection=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        ys = []
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_diag = torch.diagonal(self.Us.weight, 0)
            leaks, rdiags, hs = [], [], []

        for t in range(T):
            h_prev = h
            a_s = self.Ws(x[:, t]) + self.Us(h_prev)
            s = torch.sigmoid(a_s)
            if gate_rescale is not None:
                s = torch.clamp(s * gate_rescale, 0.0, 1.0)

            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)

            h = (1 - s) * h_prev + s * h_tilde
            h = _apply_state_increment_injection(h, h_prev, state_injection)
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = torch.clamp(1 - s, 1e-12, 1.0)
                tanh_prime = 1.0 - h_tilde ** 2
                s_prime = s * (1 - s)
                rdiag_gate = (h_tilde - h_prev) * (s_prime * us_diag.view(1, -1))
                rdiag_rec = (s * tanh_prime) * wh_diag.view(1, -1)
                rdiag = rdiag_gate + rdiag_rec
                hs.append(h)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"leak": leak, "rdiag": rdiag}


# ============================================================
# GRU
# ============================================================

class GRUCustom(BaseRNN):
    """
    Standard GRU with exposed intermediates for transport factor computation.

    Transport: mu_tl = gamma + rho0 + eta0
      gamma = first-order corrected diagonal product of leak=(1-z)
      rho0  = product of reset gate r
      eta0  = product of (1-z)*r
    """
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

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None,
                                    return_intermediates=True, state_injection=None):
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
            h = _apply_state_increment_injection(h, h_prev, state_injection)
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = torch.clamp(1.0 - z, 1e-12, 1.0)

                zprime = z * (1.0 - z)
                term1 = (g - h_prev) * zprime * diagUz.view(1, -1)

                gprime = 1.0 - g ** 2
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
        return y, hseq, {
            "z": zseq, "r": rseq, "g": gseq,
            "leak": leak, "rdiag": rdiag,
        }


# ============================================================
# LSTM
# ============================================================

class LSTMCustom(BaseRNN):
    """
    Standard LSTM with exposed intermediates for transport factor computation.

    Transport: mu_tl = base * e_end
      base  = first-order corrected diagonal product of leak=f (forget gate)
      e_end = o * (1 - tanh(c)^2)  (output gate × cell derivative)
    """
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

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None,
                                    return_intermediates=True, state_injection=None):
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
            h = _apply_state_increment_injection(h, h_prev, state_injection)
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                e = o * (1.0 - tanh_c ** 2)
                leak = torch.clamp(f, 1e-12, 1.0)

                fprime = f * (1.0 - f)
                iprime = i * (1.0 - i)
                gprime = 1.0 - g ** 2

                diagC = (c_prev * fprime) * diagUf.view(1, -1) \
                      + (i * gprime)      * diagUg.view(1, -1) \
                      + (g * iprime)      * diagUi.view(1, -1)

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
        return y, hseq, {
            "e": e, "f": f,
            "leak": leak, "rdiag": rdiag,
        }


# ============================================================
# Model builder
# ============================================================

def build_model(name: str, D: int, H: int, const_s: float = 0.005,
                ln: bool = False) -> BaseRNN:
    """
    Build an RNN model by name.

    Args:
        name: one of {const, shared, diag, gru, lstm}
        D: input dimension
        H: hidden dimension
        const_s: initial gate value for ConstGate / init_s for learned gates
        ln: whether to use LayerNorm
    """
    name = name.lower().strip()
    if name == "const":
        return ConstGateRNN(D, H, s=const_s, ln=ln)
    if name == "shared":
        return SharedGateRNN(D, H, ln=ln, init_s=const_s)
    if name in ("diag", "diaggate", "multigate"):
        return DiagGateRNN(D, H, ln=ln, init_s=const_s)
    if name == "gru":
        return GRUCustom(D, H, ln=ln)
    if name == "lstm":
        return LSTMCustom(D, H, ln=ln)
    raise ValueError(f"Unknown model: {name!r}. "
                     f"Choose from: const, shared, diag, gru, lstm")
