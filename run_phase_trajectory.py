#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — phase-trajectory runner (single seed)
=====================================================

Trains one or more gated recurrent architectures on a synthetic long-memory
regression task and records the diagnostic trajectory during training.

This is the unified version that handles all architectures in a single script,
using shared modules (models, transport, diagnostics, data).

Outputs (per model directory):
  - <model>_learning_curve.csv
  - phase_trajectory.csv
  - checkpoint_taus/ckpt_XXXX_taus.npy + .csv
  - checkpoint_taus/ckpt_XXXX_tau_slope_fit_info.json
  - checkpoint_tau_ccdf/ckpt_XXXX_tau_ccdf.csv (optional)
  - checkpoint_tau_tail/ckpt_XXXX_tau_tail_fit.json
  - checkpoint_alpha/ckpt_XXXX_alpha_grad.json (+ samples csv)

Optional analysis-only (run later with --analysis_only):
  - <model>_envelope.csv, <model>_envelope_fit.json, <model>_envelope_fit_curves.csv

Top-level:
  - cli_args.json, lag_grid.json

NO plotting here.

Usage:
  python run_phase_trajectory.py --outdir results/exp2_phase/seed_0042 --seed 42 --models shared,diag
"""

import argparse
import os
import math
import csv
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model, BaseRNN
from data import make_dataset_cpu, sample_heavy_tailed_lags, build_task_coeffs
from diagnostics import (
    log,
    run_checkpoint_diagnostics,
    compute_and_save_final_envelopes,
    detect_threshold_crossing,
)
from seed_utils import write_csv, append_csv_row


# ============================================================
# Heavy-tailed forcing injection
# ============================================================
#
# The paper-facing Path A intervention uses --inject_mode update: after the
# optimizer step, soft-tapered symmetric alpha-stable noise is added to the
# slow-unit rows of trainable parameters. This perturbs the training-time
# update driver without directly corrupting sequence-time hidden states. The
# older state/grad modes remain available for debug/comparison runs.

def _stable_sample_symmetric(
    alpha: float,
    scale: float,
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    clip: float = 0.0,
    taper: str = "hard",
) -> torch.Tensor:
    """Symmetric α-stable sample via Chambers-Mallows-Stuck.

    Returns iid samples with characteristic function exp(-(scale*|t|)^alpha)
    (β=0, location 0). For alpha=2 this is a Gaussian with standard
    deviation sqrt(2)*scale; for alpha=1 it is a Cauchy with scale `scale`.

    See Chambers, Mallows, Stuck (1976) and Nolan, "Univariate Stable
    Distributions" (2020) for the construction.
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    pi = math.pi
    eps = 1e-37
    U = (torch.rand(shape, generator=generator, device=device, dtype=dtype) - 0.5) * pi
    W = -torch.log(
        torch.rand(shape, generator=generator, device=device, dtype=dtype).clamp_min(eps)
    )
    if abs(alpha - 1.0) < 1e-6:
        X = torch.tan(U)  # symmetric Cauchy
    else:
        sin_aU = torch.sin(alpha * U)
        cos_U = torch.cos(U).clamp_min(eps)
        cos_au = torch.cos((alpha - 1.0) * U).clamp_min(eps)
        X = (sin_aU / cos_U.pow(1.0 / alpha)) * (cos_au / W).pow((1.0 - alpha) / alpha)
    if float(clip) > 0.0:
        clip = float(clip)
        if str(taper).lower() == "soft":
            X = clip * torch.tanh(X / clip)
        else:
            X = X.clamp(-clip, clip)
    return X.mul_(float(scale))


def apply_alpha_stable_grad_injection(
    model: torch.nn.Module,
    alpha: float,
    scale_multiplier: float,
    generator: torch.Generator,
    noise_clip: float = 0.0,
) -> Dict[str, float]:
    """Add symmetric α-stable noise to all trainable parameter gradients in place.

    The per-parameter dispersion is
        c_p = scale_multiplier * (||grad_p|| / sqrt(numel_p))
    so the injection is self-calibrated to the natural per-element gradient
    magnitude. When scale_multiplier=1 the injected noise has dispersion
    comparable to the average per-element gradient, but with stability
    index `alpha` instead of 2.

    Returns a summary dict for logging.
    """
    if scale_multiplier <= 0.0:
        return {"injected": 0.0}
    n_params_injected = 0
    total_disp = 0.0
    n_elements_injected = 0
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad
        n = g.numel()
        if n == 0:
            continue
        grad_norm = torch.linalg.vector_norm(g).item()
        rms = grad_norm / max(math.sqrt(n), 1e-37)
        c = float(scale_multiplier) * float(rms)
        if c <= 0.0:
            continue
        noise = _stable_sample_symmetric(
            alpha=alpha, scale=c, shape=g.shape,
            device=g.device, dtype=g.dtype, generator=generator,
            clip=float(noise_clip),
        )
        g.add_(noise)
        n_params_injected += 1
        total_disp += c * n
        n_elements_injected += n
    avg_disp = total_disp / max(1, n_elements_injected)
    return {
        "injected": 1.0,
        "alpha": float(alpha),
        "scale_multiplier": float(scale_multiplier),
        "noise_clip": float(noise_clip),
        "n_params_injected": float(n_params_injected),
        "avg_dispersion_per_element": float(avg_disp),
    }


def _build_injection_generator(args, device: torch.device):
    """Construct per-run RNG for stable-noise injection, or return None."""
    if str(getattr(args, "inject_mode", "none")) == "none":
        return None
    if float(getattr(args, "inject_alpha_noise", 0.0)) <= 0.0:
        return None
    seed_offset = int(getattr(args, "inject_grad_seed_offset", 1729))
    inj_seed = int(args.seed) + seed_offset
    gen = torch.Generator(device=str(device))
    gen.manual_seed(inj_seed)
    return gen


def _save_injection_metadata(mdir: str, args, info: Dict[str, float]) -> None:
    """Persist injection settings to <mdir>/injection_metadata.json."""
    if str(getattr(args, "inject_mode", "none")) == "none":
        return
    if float(getattr(args, "inject_alpha_noise", 0.0)) <= 0.0:
        return
    out = {
        "inject_mode": str(getattr(args, "inject_mode", "none")),
        "inject_alpha_noise": float(args.inject_alpha_noise),
        "inject_alpha": float(args.inject_alpha),
        "inject_noise_clip": float(getattr(args, "inject_noise_clip", 0.0)),
        "inject_grad_seed_offset": int(args.inject_grad_seed_offset),
        "inject_seed_used": int(args.seed) + int(args.inject_grad_seed_offset),
        "first_step_summary": info,
    }
    path = os.path.join(mdir, "injection_metadata.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _initial_state_injection_mask(H: int, q_low: float, device: torch.device) -> torch.Tensor:
    """Deterministic mask used before the first tau-based far-left slice exists."""
    n = max(1, int(math.ceil(float(q_low) * int(H))))
    mask = torch.zeros(int(H), dtype=torch.bool, device=device)
    mask[:n] = True
    return mask


def _far_left_mask_from_tau(tau: np.ndarray, H: int, q_low: float, device: torch.device) -> torch.Tensor:
    """Select units in the far-left zeta slice, equivalently the largest taus."""
    tau = np.asarray(tau, dtype=np.float64)
    valid = np.isfinite(tau) & (tau > 0)
    mask_np = np.zeros(int(H), dtype=bool)
    if np.any(valid):
        zeta = np.full(int(H), np.nan, dtype=np.float64)
        zeta[valid] = -np.log(tau[valid])
        thr = np.nanquantile(zeta[valid], float(q_low))
        mask_np = valid & (zeta <= thr)
    if not np.any(mask_np):
        mask_np[:max(1, int(math.ceil(float(q_low) * int(H))))] = True
    return torch.as_tensor(mask_np, dtype=torch.bool, device=device)


def _selected_slow_rows(param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the leading-dimension slow rows selected by ``mask``."""
    return param[mask] if param.ndim >= 1 else param


def _is_slow_row_parameter(name: str, param: torch.Tensor, H: int) -> bool:
    """Parameter selector for slow-relevant update rows.

    A unit q's effective-rate dynamics are attached to tensors whose leading
    dimension indexes hidden units. This includes input/recurrent rows and,
    for DiagGate, per-unit gate rows. Shared gates and readout weights are
    naturally excluded because their leading dimension is not H.
    """
    if (not param.requires_grad) or param.ndim < 1:
        return False
    if str(name).startswith("out."):
        return False
    return int(param.shape[0]) == int(H)


def _snapshot_slow_row_params(
    model: torch.nn.Module,
    slow_mask: torch.Tensor,
    H: int,
) -> Dict[str, torch.Tensor]:
    """Snapshot selected slow rows before the optimizer step."""
    snap: Dict[str, torch.Tensor] = {}
    mask = slow_mask.to(dtype=torch.bool)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if _is_slow_row_parameter(name, p, H):
                snap[name] = _selected_slow_rows(p.detach(), mask).clone()
    return snap


def _append_update_forcing_samples(
    buffer: Dict[str, object],
    total_update: torch.Tensor,
    injected_update: torch.Tensor,
) -> None:
    """Append flattened update samples, keeping a bounded recent window."""
    max_samples = int(buffer.get("max_samples", 20000))
    if max_samples <= 0:
        return

    def _to_np(x: torch.Tensor) -> np.ndarray:
        arr = x.detach().reshape(-1).to("cpu").numpy().astype(np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > max_samples:
            idx = np.linspace(0, arr.size - 1, num=max_samples, dtype=int)
            arr = arr[idx]
        return arr

    for key, tensor in (("total", total_update), ("injected", injected_update)):
        arr = _to_np(tensor)
        if arr.size == 0:
            continue
        chunks = buffer.setdefault(key, [])
        chunks.append(arr)
        merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if merged.size > max_samples:
            merged = merged[-max_samples:]
        buffer[key] = [merged]


def _update_forcing_payload(
    buffer: Dict[str, object],
    slow_mask: torch.Tensor,
    q_low: float,
) -> Dict[str, object]:
    """Materialize and reset the update-forcing buffer for diagnostics."""
    total_chunks = buffer.get("total", [])
    injected_chunks = buffer.get("injected", [])
    total = np.concatenate(total_chunks) if total_chunks else np.empty(0, dtype=np.float64)
    injected = np.concatenate(injected_chunks) if injected_chunks else np.empty(0, dtype=np.float64)
    payload = {
        "total": total,
        "injected": injected,
        "n_slow_units": int(slow_mask.detach().to("cpu").sum().item()),
        "slow_fraction": float(slow_mask.detach().to("cpu").float().mean().item()) if slow_mask.numel() else float("nan"),
        "slow_q_low": float(q_low),
    }
    buffer["total"] = []
    buffer["injected"] = []
    return payload


def apply_update_space_injection(
    model: torch.nn.Module,
    pre_step_snapshot: Dict[str, torch.Tensor],
    slow_mask: torch.Tensor,
    args,
    generator: torch.Generator | None,
    update_buffer: Dict[str, object],
) -> Dict[str, float]:
    """Inject soft-tapered stable noise into slow-row post-Adam updates."""
    H = int(args.H)
    mask = slow_mask.to(dtype=torch.bool)
    inject_active = (
        str(getattr(args, "inject_mode", "none")).lower() == "update"
        and generator is not None
        and float(getattr(args, "inject_alpha_noise", 0.0)) > 0.0
    )
    n_params = 0
    n_elements = 0
    total_disp = 0.0

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in pre_step_snapshot or not _is_slow_row_parameter(name, p, H):
                continue
            before = pre_step_snapshot[name].to(device=p.device, dtype=p.dtype)
            after = _selected_slow_rows(p.data, mask)
            adam_delta = after - before
            if adam_delta.numel() == 0:
                continue

            noise = torch.zeros_like(adam_delta)
            if inject_active:
                rms = torch.linalg.vector_norm(adam_delta.detach()).item() / max(math.sqrt(adam_delta.numel()), 1e-37)
                scale = float(getattr(args, "inject_alpha_noise", 0.0)) * float(rms)
                if scale > 0.0:
                    noise = _stable_sample_symmetric(
                        alpha=float(getattr(args, "inject_alpha", 1.5)),
                        scale=scale,
                        shape=adam_delta.shape,
                        device=p.device,
                        dtype=p.dtype,
                        generator=generator,
                        clip=float(getattr(args, "inject_noise_clip", 0.0)),
                        taper="soft",
                    )
                    if p.ndim == 1:
                        p.data[mask] = p.data[mask] + noise
                    else:
                        p.data[mask, ...] = p.data[mask, ...] + noise
                    total_disp += scale * int(noise.numel())

            _append_update_forcing_samples(update_buffer, adam_delta + noise, noise)
            n_params += 1
            n_elements += int(adam_delta.numel())

    return {
        "injected": 1.0 if inject_active else 0.0,
        "mode": "update",
        "alpha": float(getattr(args, "inject_alpha", 1.5)),
        "scale_multiplier": float(getattr(args, "inject_alpha_noise", 0.0)),
        "noise_clip": float(getattr(args, "inject_noise_clip", 0.0)),
        "noise_taper": "soft_tanh",
        "slow_mode_q_low": float(getattr(args, "slow_mode_q_low", 0.10)),
        "n_slow_units_targeted": float(mask.detach().to("cpu").sum().item()),
        "n_params_targeted": float(n_params),
        "n_elements_targeted": float(n_elements),
        "avg_dispersion_per_element": float(total_disp / max(1, n_elements)),
    }


def _state_injection_summary(mask: torch.Tensor, args) -> Dict[str, float]:
    return {
        "injected": 1.0,
        "mode": "state",
        "alpha": float(args.inject_alpha),
        "scale_multiplier": float(args.inject_alpha_noise),
        "noise_clip": float(getattr(args, "inject_noise_clip", 0.0)),
        "slow_mode_q_low": float(getattr(args, "slow_mode_q_low", 0.10)),
        "n_state_units_targeted": float(mask.detach().to("cpu").sum().item()),
    }


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Note: MPS is intentionally NOT auto-selected. The diagnostic
    # pipeline (transport.py, diagnostics.py, diagnostics/diag_utils.py)
    # uses float64 cumulative log-sums on the device for numerical
    # stability of the τ-spectrum estimation, and MPS does not support
    # float64. Until the diagnostic float64 work is moved to CPU
    # explicitly, --device auto on macOS must fall through to CPU.
    return torch.device("cpu")


# ============================================================
# Learning curve CSV helpers
# ============================================================

def _init_learning_curve_csv(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "train_loss"])


def _append_learning_curve_csv(path: str, epoch: int, step: int, loss: float):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([int(epoch), int(step), float(loss)])


# ============================================================
# Phase trajectory columns (canonical order)
# ============================================================

TRAJ_COLS = [
    "epoch",
    "step",
    "alpha_hat", "alpha_ecf", "alpha_mcculloch",
    "sigma_alpha_hat", "alpha_hat_std", "alpha_hat_se", "alpha_agreement",
    "beta_hat", "beta_r2",
    "beta_median", "beta_lo", "beta_hi", "p_beta_lt1", "beta_bootstrap_B_eff",
    "tau_mean", "tau_q90", "tau_q99",
    "zeta_q10", "zeta_q90", "delta_zeta",
    "tau_fit_r2_mean", "tau_fit_n_valid",
    "fit_lags_min", "fit_lags_max",
    "alpha_reliable", "alpha_method", "n_samples",
    "forcing_alpha_hat", "forcing_alpha_hill", "forcing_alpha_pickands",
    "forcing_alpha_moment",
    "forcing_alpha_eff", "forcing_alpha_eff_lo", "forcing_alpha_eff_hi",
    "forcing_xi_hat", "forcing_alpha_moment_raw", "forcing_alpha_hill_raw",
    "forcing_alpha_reliable",
    "forcing_alpha_detectably_heavy",
    "forcing_alpha_substantively_heavy",
    "forcing_alpha_resolvably_heavy",
    "forcing_alpha_substantive_threshold",
    "forcing_gaussian_test_alpha",
    "forcing_gaussian_p_value", "forcing_gaussian_p_value_lo", "forcing_gaussian_p_value_hi",
    "forcing_gaussian_p_value_mc_se",
    "forcing_gaussian_p_value_floor", "forcing_gaussian_p_value_at_floor",
    "forcing_gaussian_reject",
    "forcing_alpha2_band_lo", "forcing_alpha2_band_hi",
    "forcing_heavy_fraction", "forcing_n_samples", "forcing_k_selected",
    "beta_env", "beta_env_r2",
    "phase_label",
]


# ============================================================
# Training loop with phase tracking
# ============================================================

def build_optimizer_for_model(args, model: BaseRNN):
    """Construct the optimizer used by the experiment."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.0,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer {args.optimizer}")


def save_analysis_checkpoint(
    args,
    model: BaseRNN,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    mdir: str,
    epoch: int,
    step: int,
) -> None:
    """Save final state needed by the later plotting/analysis pass."""
    ckpt_dir = os.path.join(mdir, "analysis_checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model_name": model_name,
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    log(f"[ckpt:{model_name}] saved final analysis checkpoint -> {ckpt_path}")

def train_with_phase_tracking(
    args,
    model: BaseRNN,
    model_name: str,
    mdir: str,
    Xtr_cpu: torch.Tensor,
    Ytr_cpu: torch.Tensor,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    fit_lags: np.ndarray,
    ells: np.ndarray,
) -> None:
    """Train model and run diagnostics at checkpoint epochs."""

    opt = build_optimizer_for_model(args, model)

    if args.orth_init:
        model.apply_orthogonal()

    # Initialize CSVs
    lc_csv = os.path.join(mdir, f"{model_name}_learning_curve.csv")
    _init_learning_curve_csv(lc_csv)

    traj_csv = os.path.join(mdir, "phase_trajectory.csv")
    write_csv(traj_csv, TRAJ_COLS, [])

    Btot = int(Xtr_cpu.shape[0])
    bs = int(args.batch_size)
    nb = max(1, math.ceil(Btot / bs))
    log_every = max(1, int(args.epochs) // 5)

    ckpt_every = int(max(1, args.checkpoint_every))
    ckpt_epochs = set(
        [1, int(args.epochs)]
        + list(range(ckpt_every, int(args.epochs) + 1, ckpt_every))
    )

    log(f"[train:{model_name}] start epochs={args.epochs} bs={bs} "
        f"opt={args.optimizer} lr={args.lr}")

    # Heavy-tailed forcing injection (Path A). Inactive when
    # --inject_mode none or --inject_alpha_noise == 0.
    inject_mode = str(getattr(args, "inject_mode", "none")).lower()
    injection_gen = _build_injection_generator(args, device)
    state_injection = None
    if injection_gen is not None:
        log(f"[train:{model_name}] heavy-tailed forcing injection ENABLED: "
            f"mode={inject_mode}, "
            f"alpha={float(args.inject_alpha):.3f}, "
            f"scale_multiplier={float(args.inject_alpha_noise):.3g}, "
            f"seed={int(args.seed) + int(args.inject_grad_seed_offset)}")
        if inject_mode in {"state", "update"}:
            slow_mask = _initial_state_injection_mask(
                H=int(args.H),
                q_low=float(getattr(args, "slow_mode_q_low", 0.10)),
                device=device,
            )
        if inject_mode == "state":
            state_injection = {
                "alpha": float(args.inject_alpha),
                "scale_multiplier": float(args.inject_alpha_noise),
                "noise_clip": float(getattr(args, "inject_noise_clip", 0.0)),
                "mask": slow_mask,
                "generator": injection_gen,
            }
    if "slow_mask" not in locals():
        slow_mask = _initial_state_injection_mask(
            H=int(args.H),
            q_low=float(getattr(args, "slow_mode_q_low", 0.10)),
            device=device,
        )
    update_forcing_buffer: Dict[str, object] = {
        "total": [],
        "injected": [],
        "max_samples": int(getattr(args, "forcing_tail_max_samples", 20000)),
    }
    _inject_metadata_saved = False

    nan_halt = False
    global_step = 0
    trajectory_rows = []  # accumulate checkpoint rows for threshold-crossing detection
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(Btot)
        loss_sum = 0.0
        n_seen = 0

        for bi in range(nb):
            lo = bi * bs
            hi = min(Btot, (bi + 1) * bs)
            idx = perm[lo:hi]

            xb = Xtr_cpu[idx].to(device, non_blocking=True)
            yb = Ytr_cpu[idx].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            yhat, _, _ = model.forward_with_intermediates(
                xb, return_intermediates=False,
                state_injection=state_injection,
            )
            loss = F.mse_loss(yhat, yb)

            if not torch.isfinite(loss):
                log(f"[train:{model_name}] NaN/Inf loss at epoch={ep}, "
                    f"batch={bi}. Halting.")
                nan_halt = True
                del xb, yb, yhat, loss
                break

            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.grad_clip)
                )

            # Legacy gradient-noise injection. The manuscript Path A uses
            # update injection below; this mode is retained only for comparison.
            if injection_gen is not None and inject_mode == "grad":
                _inj_info = apply_alpha_stable_grad_injection(
                    model=model,
                    alpha=float(args.inject_alpha),
                    scale_multiplier=float(args.inject_alpha_noise),
                    generator=injection_gen,
                    noise_clip=float(getattr(args, "inject_noise_clip", 0.0)),
                )
                if not _inject_metadata_saved:
                    _save_injection_metadata(mdir, args, _inj_info)
                    _inject_metadata_saved = True
            elif injection_gen is not None and inject_mode == "state" and not _inject_metadata_saved:
                _save_injection_metadata(
                    mdir,
                    args,
                    _state_injection_summary(state_injection["mask"], args),
                )
                _inject_metadata_saved = True

            pre_step_snapshot = _snapshot_slow_row_params(model, slow_mask, int(args.H))
            opt.step()
            _update_info = apply_update_space_injection(
                model=model,
                pre_step_snapshot=pre_step_snapshot,
                slow_mask=slow_mask,
                args=args,
                generator=injection_gen,
                update_buffer=update_forcing_buffer,
            )
            if injection_gen is not None and inject_mode == "update" and not _inject_metadata_saved:
                _save_injection_metadata(mdir, args, _update_info)
                _inject_metadata_saved = True
            global_step += 1

            loss_sum += float(loss.item()) * int(hi - lo)
            n_seen += int(hi - lo)
            del xb, yb, yhat, loss

        if nan_halt:
            break

        train_loss_epoch = loss_sum / max(1, n_seen)
        _append_learning_curve_csv(lc_csv, ep, global_step, train_loss_epoch)

        if (ep == 1) or (ep == int(args.epochs)) or (ep % log_every == 0):
            log(f"[train:{model_name}] ep {ep}/{args.epochs} "
                f"avg_loss={train_loss_epoch:.4g}")

        if ep in ckpt_epochs:
            log(f"[ckpt:{model_name}] diagnostics at epoch={ep} ...")
            row = run_checkpoint_diagnostics(
                args, model, model_name, mdir, ep,
                Xtr_cpu, Ytr_cpu, Xdg_cpu,
                device=device, fit_lags=fit_lags, ells=ells,
                step=global_step,
                update_forcing_samples=_update_forcing_payload(
                    update_forcing_buffer,
                    slow_mask=slow_mask,
                    q_low=float(getattr(args, "slow_mode_q_low", 0.10)),
                ),
            )
            append_csv_row(traj_csv, [row[k] for k in TRAJ_COLS])
            trajectory_rows.append(row)

            if state_injection is not None or inject_mode == "update":
                tau_path = os.path.join(
                    mdir, "checkpoint_taus", f"ckpt_{int(ep):04d}_taus.npy"
                )
                if os.path.isfile(tau_path):
                    tau_ckpt = np.load(tau_path)
                    slow_mask = _far_left_mask_from_tau(
                        tau_ckpt,
                        H=int(args.H),
                        q_low=float(getattr(args, "slow_mode_q_low", 0.10)),
                        device=device,
                    )
                    if state_injection is not None:
                        state_injection["mask"] = slow_mask

            # Optionally save model checkpoint
            if getattr(args, 'save_model_checkpoints', False):
                ckpt_path = os.path.join(mdir, "model_checkpoints",
                                         f"ckpt_{ep:04d}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

    if args.save_analysis_checkpoint and not nan_halt:
        save_analysis_checkpoint(
            args, model, opt, model_name, mdir,
            epoch=int(args.epochs), step=global_step,
        )

    # Threshold-crossing detection
    tcross = detect_threshold_crossing(trajectory_rows, persistence=2)
    tcross_path = os.path.join(mdir, f"{model_name}_threshold_crossing.json")
    with open(tcross_path, "w") as f:
        json.dump(tcross, f, indent=2, default=str)
    if tcross["crossed"]:
        log(f"[tcross:{model_name}] threshold crossed at step "
            f"{tcross['t_cross_step']} (epoch {tcross['t_cross_epoch']}), "
            f"alpha={tcross['alpha_at_cross']}, "
            f"beta_env={tcross['beta_env_at_cross']}")
    elif tcross.get("left_censored"):
        log(f"[tcross:{model_name}] already anti-collapsed at the first "
            f"observed checkpoint (left-censored at step "
            f"{tcross['first_observed_step']})")
    else:
        log(f"[tcross:{model_name}] never crossed threshold "
            f"(right-censored at step {tcross['horizon_step']})")

    log(f"[train:{model_name}] done")


# ============================================================
# Per-model entry point
# ============================================================

def run_for_model(
    args,
    model_name: str,
    outdir: str,
    Xtr_cpu: torch.Tensor,
    Ytr_cpu: torch.Tensor,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    ells: np.ndarray,
    fit_lags: np.ndarray,
) -> Dict:
    """Train one model and optionally compute final envelope."""
    mdir = os.path.join(outdir, model_name)
    os.makedirs(mdir, exist_ok=True)

    model = build_model(
        model_name, args.D, args.H,
        const_s=args.const_s, ln=args.layernorm,
    ).to(device)

    train_with_phase_tracking(
        args, model, model_name, mdir,
        Xtr_cpu, Ytr_cpu, Xdg_cpu,
        device=device, fit_lags=fit_lags, ells=ells,
    )

    return {"ok": True}


def analyze_final_envelope_for_model(
    args,
    model_name: str,
    outdir: str,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    ells: np.ndarray,
    fit_lags: np.ndarray,
) -> None:
    """Load a final analysis checkpoint and compute final diagnostics."""
    mdir = os.path.join(outdir, model_name)
    ckpt_path = os.path.join(mdir, "analysis_checkpoint", "final.pt")
    if not os.path.isfile(ckpt_path):
        log(f"[analysis:{model_name}] missing analysis checkpoint: {ckpt_path}")
        return

    model = build_model(
        model_name, args.D, args.H,
        const_s=args.const_s, ln=args.layernorm,
    ).to(device)
    opt = build_optimizer_for_model(args, model)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])

    log(f"[analysis:{model_name}] computing final envelopes from {ckpt_path}")
    compute_and_save_final_envelopes(
        model_name=model_name,
        model=model,
        optimizer=opt,
        mdir=mdir,
        Xdg_cpu=Xdg_cpu,
        device=device,
        ells=ells,
        diag_batch_size=int(args.diag_batch_size),
        fit_lags=fit_lags,
        tau_ccdf_qmin=float(args.tau_ccdf_qmin),
        tau_ccdf_qmax=float(args.tau_ccdf_qmax),
        beta_bootstrap_B=int(args.beta_bootstrap_B),
        beta_bootstrap_ci=float(args.beta_bootstrap_ci),
        phase_r2_threshold=float(args.phase_r2_threshold),
        power_window_beta_min=float(args.power_window_beta_min),
        power_window_min_points=int(args.power_window_min_points),
        power_window_min_fraction=float(args.power_window_min_fraction),
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Anti-Collapse phase-trajectory runner"
    )

    # Output / seeds
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--models", type=str, default="const,shared,diag",
                   help="Comma-separated model names")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--w_seed", type=int, default=41,
                   help="Base seed for gradient projection directions")

    # Data
    p.add_argument("--Nseq_train", type=int, default=8000)
    p.add_argument("--Nseq_diag", type=int, default=8000)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--H", type=int, default=512)

    # Optimization
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd", "sgd_momentum"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Heavy-tailed forcing injection. The manuscript Path A uses
    # --inject_mode update; state/grad are retained only as legacy comparison
    # modes.
    p.add_argument("--inject_mode", type=str, default="none",
                   choices=["none", "update", "state", "grad"],
                   help="Heavy-tailed forcing injection target: none, update, legacy state, or legacy grad.")
    p.add_argument("--inject_alpha_noise", type=float, default=0.0,
                   help="Scale multiplier for heavy-tailed forcing injection. 0 disables.")
    p.add_argument("--inject_alpha", type=float, default=1.5,
                   help="Stability index α∈(0,2] of the injected α-stable noise.")
    p.add_argument("--inject_noise_clip", type=float, default=100.0,
                   help="Bound the standardized α-stable draw before scaling. Update-mode "
                        "uses a soft tanh taper; legacy state/grad modes use hard clipping. "
                        "0 disables the bound.")
    p.add_argument("--inject_grad_seed_offset", type=int, default=1729,
                   help="Offset added to --seed for the injection RNG (reproducibility).")

    # Architecture
    p.add_argument("--const_s", type=float, default=0.005)
    p.add_argument("--orth_init", action="store_true")
    p.add_argument("--layernorm", action="store_true")

    # Diagnostic lag grids
    p.add_argument("--lag_min", type=int, default=4)
    p.add_argument("--lag_max", type=int, default=256)
    p.add_argument("--num_lags", type=int, default=128)

    # Task
    p.add_argument("--task_lags", type=str, default="32,64,128,192,256")
    p.add_argument("--task_coeffs", type=str, default="0.6,0.5,0.4,0.32,0.26")
    p.add_argument("--noise_std", type=float, default=0.3)

    # Task variant: fixed-lag (default) or heavy-tailed-lag (truncated Pareto on lags).
    # The heavy-tailed-lag variant overrides --task_lags/--task_coeffs at runtime.
    p.add_argument("--task_variant", type=str, default="fixed",
                   choices=["fixed", "heavy_tail"])
    p.add_argument("--task_alpha", type=float, default=1.0,
                   help="Tail index α_task for truncated Pareto lag distribution (heavy_tail variant)")
    p.add_argument("--task_lag_min", type=int, default=8)
    p.add_argument("--task_lag_max", type=int, default=384)
    p.add_argument("--task_K", type=int, default=8,
                   help="Number of target lags sampled in heavy_tail variant")
    p.add_argument("--task_coeff_base", type=float, default=0.6)
    p.add_argument("--task_coeff_decay", type=float, default=0.85)
    p.add_argument("--task_lag_seed", type=int, default=20260410,
                   help="Seed for sampling the per-realization lag set (heavy_tail variant)")

    # Checkpointing
    p.add_argument("--diag_batch_size", type=int, default=256)
    p.add_argument("--checkpoint_every", type=int, default=50)

    # Alpha estimation
    p.add_argument("--alpha_n_grad_batches_ckpt", type=int, default=256)
    p.add_argument("--alpha_grad_batch_size", type=int, default=256)
    p.add_argument("--alpha_use_grad_clip", action="store_true")
    p.add_argument("--alpha_n_directions", type=int, default=5,
                   help="K fixed random projection directions for alpha estimation")
    p.add_argument("--alpha_method", type=str, default="ecf",
                   choices=["mcculloch", "ecf"])
    p.add_argument("--min_samples_alpha", type=int, default=500)

    # Slow-mode forcing tail diagnostic.
    p.add_argument("--slow_mode_q_low", type=float, default=0.10,
                   help="Far-left zeta quantile used for slow-mode forcing measurement.")
    p.add_argument("--forcing_tail_max_samples", type=int, default=20000,
                   help="Maximum absolute increment samples used by the slow-mode tail estimator.")
    p.add_argument("--forcing_tail_bootstrap_B", type=int, default=200,
                   help="Synthetic-stable draws per alpha for the calibration curve.")
    p.add_argument("--forcing_tail_ci_B", type=int, default=100,
                   help="Data-bootstrap resamples for the effective-alpha CI.")
    p.add_argument("--forcing_tail_k_min", type=int, default=50,
                   help="Minimum upper-order statistic count for the calibrated EVI estimator.")
    p.add_argument("--forcing_tail_k_frac", type=float, default=0.08,
                   help="Genuine-tail order-statistic fraction (k = k_frac * n) for the EVI estimator.")
    p.add_argument("--forcing_tail_k_max_frac", type=float, default=0.20,
                   help="Deprecated legacy stability-scan upper bound; unused by the calibrated estimator.")
    p.add_argument("--forcing_tail_substantive_alpha", type=float, default=1.8,
                   help="Substantive heaviness cutoff: alpha_eff_hi must be <= this value.")
    p.add_argument("--forcing_tail_gaussian_test_alpha", type=float, default=0.05,
                   help="One-sided calibrated Gaussian-boundary test level for the slow-mode tail statistic.")

    # Tau fit
    p.add_argument("--tau_fit_lag_min", type=int, default=64)
    p.add_argument("--tau_fit_lag_max", type=int, default=256)
    p.add_argument("--tau_fit_num_lags", type=int, default=24)

    # CCDF tail fit
    p.add_argument("--tau_ccdf_qmin", type=float, default=0.75)
    p.add_argument("--tau_ccdf_qmax", type=float, default=0.995)

    # Bootstrap β̂ over neuron population
    p.add_argument("--beta_bootstrap_B", type=int, default=2000,
                   help="Number of bootstrap resamples for beta stability interval")
    p.add_argument("--beta_bootstrap_ci", type=float, default=0.90,
                   help="Confidence level for bootstrap percentile interval")

    # Phase classification
    p.add_argument("--phase_r2_threshold", type=float, default=0.90,
                   help="R² threshold for CCDF tail fit to be considered reliable")
    p.add_argument("--power_window_beta_min", type=float, default=0.10,
                   help="Minimum envelope exponent for a visible power-law window")
    p.add_argument("--power_window_min_points", type=int, default=8,
                   help="Minimum lag-grid points in the power-law window")
    p.add_argument("--power_window_min_fraction", type=float, default=0.05,
                   help="Minimum fraction of lag-grid points in the power-law window")

    # Saving / later analysis
    p.add_argument("--save_checkpoint_ccdf", action="store_true")
    p.add_argument("--save_final_envelope", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--save_analysis_checkpoint", action="store_true",
                   help="Save final model+optimizer state for later plot/analysis pass")
    p.add_argument("--save_model_checkpoints", action="store_true")
    p.add_argument("--analysis_only", action="store_true",
                   help="Skip training and compute final envelope analysis from saved checkpoint")

    # Device
    p.add_argument("--device", type=str, default="cuda",
                   choices=["auto", "cpu", "mps", "cuda"])

    args = p.parse_args()
    if args.task_variant == "fixed":
        args.task_lags = [int(s) for s in args.task_lags.split(",") if s.strip()]
        args.task_coeffs = [float(s) for s in args.task_coeffs.split(",") if s.strip()]
    elif args.task_variant == "heavy_tail":
        rng = np.random.default_rng(args.task_lag_seed)
        args.task_lags = sample_heavy_tailed_lags(
            K=args.task_K,
            lag_min=args.task_lag_min,
            lag_max=args.task_lag_max,
            alpha_task=args.task_alpha,
            rng=rng,
        )
        args.task_coeffs = build_task_coeffs(
            K=len(args.task_lags),
            coeff_base=args.task_coeff_base,
            coeff_decay=args.task_coeff_decay,
        )
    else:
        raise ValueError(f"Unknown task_variant={args.task_variant}")
    assert len(args.task_lags) == len(args.task_coeffs)
    return args


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = resolve_device(args.device)
    log(f"Running on device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")

    # Lag grids
    ells = np.linspace(args.lag_min, args.lag_max, args.num_lags, dtype=int)
    ells = np.unique(np.clip(ells, 1, args.T - 1)).astype(int)

    fit_lags = np.linspace(
        args.tau_fit_lag_min, args.tau_fit_lag_max, args.tau_fit_num_lags, dtype=int
    )
    fit_lags = np.unique(np.clip(fit_lags, 1, args.T - 1)).astype(int)

    # Generate datasets
    Xtr_cpu, Ytr_cpu, u_vec = make_dataset_cpu(
        args.Nseq_train, args.T, args.D,
        args.task_lags, args.task_coeffs, args.noise_std, u_vec=None,
    )
    Xdg_cpu, _, _ = make_dataset_cpu(
        args.Nseq_diag, args.T, args.D,
        args.task_lags, args.task_coeffs, args.noise_std, u_vec=u_vec,
    )

    if device.type == "cuda":
        Xtr_cpu = Xtr_cpu.pin_memory()
        Ytr_cpu = Ytr_cpu.pin_memory()
        Xdg_cpu = Xdg_cpu.pin_memory()

    # Save metadata for simulation runs.  Analysis-only passes should not
    # rewrite the original data-collection manifest.
    if not args.analysis_only:
        with open(os.path.join(args.outdir, "cli_args.json"), "w") as jf:
            json.dump(vars(args), jf, indent=2)
        with open(os.path.join(args.outdir, "lag_grid.json"), "w") as jf:
            json.dump({"ells": ells.tolist(), "tau_fit_lags": fit_lags.tolist()}, jf, indent=2)

    # Run each model, or perform the deferred final-envelope analysis.
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for mname in models:
        if args.analysis_only:
            log(f"[analysis] model={mname}")
            analyze_final_envelope_for_model(
                args, mname, args.outdir,
                Xdg_cpu,
                device=device, ells=ells, fit_lags=fit_lags,
            )
        else:
            log(f"[run] model={mname}")
            run_for_model(
                args, mname, args.outdir,
                Xtr_cpu, Ytr_cpu, Xdg_cpu,
                device=device, ells=ells, fit_lags=fit_lags,
            )

    log("Done.")


if __name__ == "__main__":
    main()
