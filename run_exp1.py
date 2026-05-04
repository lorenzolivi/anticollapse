#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — phase-trajectory runner (single seed)
=====================================================

Trains all 5 gated RNN architectures (ConstGate, SharedGate, DiagGate, GRU, LSTM)
on a synthetic long-memory regression task and tracks the dynamical trajectory
in the (α̂, β̂) phase plane during training.

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

Optional final-only (set --save_final_envelope):
  - <model>_envelope.csv, <model>_envelope_fit.json, <model>_envelope_fit_curves.csv

Top-level:
  - cli_args.json, lag_grid.json

NO plotting here.

Usage:
  python run_exp1.py --outdir results/exp2_phase/seed_0042 --seed 42 --models shared,diag,gru,lstm
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
    "tau_fit_r2_mean", "tau_fit_n_valid",
    "fit_lags_min", "fit_lags_max",
    "alpha_reliable", "alpha_method", "n_samples",
    "beta_env", "beta_env_r2",
    "phase_label",
]


# ============================================================
# Training loop with phase tracking
# ============================================================

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

    # Optimizer setup
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.0,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd_momentum":
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

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
                xb, return_intermediates=False
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
            opt.step()
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
                device=device, fit_lags=fit_lags,
                step=global_step,
            )
            append_csv_row(traj_csv, [row[k] for k in TRAJ_COLS])
            trajectory_rows.append(row)

            # Optionally save model checkpoint
            if getattr(args, 'save_model_checkpoints', False):
                ckpt_path = os.path.join(mdir, "model_checkpoints",
                                         f"ckpt_{ep:04d}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

    if args.save_final_envelope and not nan_halt:
        # Final GELR envelope diagnostics need the live optimizer state.
        # Also runs definitive phase classification (Table 1 with AIC).
        # NOTE: the canonical final_phase.json is only emitted when
        # --save_final_envelope is set.  Paper runs should always use
        # this flag; otherwise only checkpoint-level (provisional)
        # phase labels are available.
        log(f"[final:{model_name}] computing transport and GELR envelopes ...")
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


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Anti-Collapse phase-trajectory runner"
    )

    # Output / seeds
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--models", type=str, default="const,shared,diag,gru,lstm",
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

    # Saving
    p.add_argument("--save_checkpoint_ccdf", action="store_true")
    p.add_argument("--save_final_envelope", action="store_true")
    p.add_argument("--save_model_checkpoints", action="store_true")

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

    # Save metadata
    with open(os.path.join(args.outdir, "cli_args.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)
    with open(os.path.join(args.outdir, "lag_grid.json"), "w") as jf:
        json.dump({"ells": ells.tolist(), "tau_fit_lags": fit_lags.tolist()}, jf, indent=2)

    # Run each model
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for mname in models:
        log(f"[run] model={mname}")
        run_for_model(
            args, mname, args.outdir,
            Xtr_cpu, Ytr_cpu, Xdg_cpu,
            device=device, ells=ells, fit_lags=fit_lags,
        )

    log("Done.")


if __name__ == "__main__":
    main()
