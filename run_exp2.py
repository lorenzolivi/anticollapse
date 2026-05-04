#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — Experiment 3: Stochastic Forcing Ablation (Unified)
===================================================================

Retrains anti-collapse architectures (DiagGate, GRU, LSTM) under interventions
that suppress stochastic forcing, to causally demonstrate that heavy-tailed
gradient noise is the active mechanism sustaining broad time-scale spectra.

Two modes:
  1. FROM-SCRATCH ablation: Train from random init under ablated conditions.
     Tests whether anti-collapse emerges when forcing is suppressed.

  2. WARM-START ablation: Train normally for --warmup_epochs, then apply
     the intervention. Tests whether anti-collapse REVERSES when forcing
     is suppressed after it has already emerged.

Ablation conditions:
  - baseline:          Standard non-ablated training hyperparameters (control)
  - batch_ablation:    Massive batch sizes (suppress η_J → 0)
  - clip_ablation:     Aggressive gradient clipping (norm-based)
  - winsorize_ablation: Percentile-based gradient clipping (push α → 2)

Uses shared modules: models, transport, diagnostics, data.

Usage:
  # From-scratch baseline
  python run_exp2.py --outdir results/exp3_forcing/seed_0042 \\
    --condition baseline --seed 42

  # Batch ablation from scratch
  python run_exp2.py --outdir results/exp3_forcing/seed_0042 \\
    --condition batch_ablation --ablation_values 2048,4096,8192

  # Warm-start winsorization (train normally for 250 epochs, then intervene)
  python run_exp2.py --outdir results/exp3_forcing/seed_0042 \\
    --condition winsorize_ablation --ablation_values 95,90,80 \\
    --warmup_epochs 250
"""

import argparse
import os
import math
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional

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
    return torch.device("cpu")


# ============================================================
# CSV helpers
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
# Training loop with phase tracking + ablation
# ============================================================

def train_with_phase_tracking_ablation(
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
    # Ablation parameters (applied after warmup_epochs if warm-starting)
    ablation_batch_size: Optional[int] = None,
    ablation_grad_clip: Optional[float] = None,
    winsorize_pct: Optional[float] = None,
    warmup_epochs: int = 0,
) -> None:
    """
    Train model with optional warm-start ablation.

    If warmup_epochs > 0:
      - Epochs [1, warmup_epochs]: normal training (baseline hyperparameters)
      - Epochs [warmup_epochs+1, total]: ablated training
    If warmup_epochs == 0:
      - All epochs use ablated hyperparameters (from-scratch ablation)
    """

    # Optimizer
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
    log_every = max(1, int(args.epochs) // 5)

    ckpt_every = int(max(1, args.checkpoint_every))
    ckpt_epochs = set(
        [1, int(args.epochs)]
        + list(range(ckpt_every, int(args.epochs) + 1, ckpt_every))
    )
    # Always checkpoint at warmup boundary if warm-starting
    if warmup_epochs > 0:
        ckpt_epochs.add(warmup_epochs)
        ckpt_epochs.add(warmup_epochs + 1)

    # Determine baseline vs ablation batch sizes
    baseline_bs = int(args.batch_size)
    ablation_bs = ablation_batch_size if ablation_batch_size is not None else baseline_bs

    # Determine baseline vs ablation grad clip
    baseline_clip = float(args.grad_clip)
    ablation_clip = ablation_grad_clip if ablation_grad_clip is not None else baseline_clip

    desc_parts = [f"epochs={args.epochs}", f"opt={args.optimizer}", f"lr={args.lr}"]
    if warmup_epochs > 0:
        desc_parts.append(f"warmup={warmup_epochs}")
    if ablation_batch_size is not None:
        desc_parts.append(f"abl_bs={ablation_batch_size}")
    if ablation_grad_clip is not None:
        desc_parts.append(f"abl_clip={ablation_grad_clip}")
    if winsorize_pct is not None:
        desc_parts.append(f"winsorize={winsorize_pct}")

    log(f"[train:{model_name}] start {', '.join(desc_parts)}")

    nan_halt = False
    global_step = 0
    trajectory_rows = []  # accumulate checkpoint rows for threshold-crossing detection
    for ep in range(1, int(args.epochs) + 1):
        model.train()

        # Determine current-epoch parameters
        in_ablation = (warmup_epochs == 0) or (ep > warmup_epochs)
        current_bs = ablation_bs if in_ablation else baseline_bs
        current_clip = ablation_clip if in_ablation else baseline_clip
        current_winsorize = winsorize_pct if in_ablation else None

        nb = max(1, math.ceil(Btot / current_bs))
        perm = torch.randperm(Btot)
        loss_sum = 0.0
        n_seen = 0

        for bi in range(nb):
            lo = bi * current_bs
            hi = min(Btot, (bi + 1) * current_bs)
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

            # Winsorization: percentile-based gradient clipping
            if current_winsorize is not None and current_winsorize < 100:
                all_grads = torch.cat([
                    p.grad.detach().view(-1)
                    for p in model.parameters() if p.grad is not None
                ])
                threshold = torch.quantile(
                    all_grads.abs(), current_winsorize / 100.0
                )
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.clamp_(-threshold, threshold)

            # Standard gradient clipping
            if current_clip and current_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(current_clip)
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
            phase_label = "ablation" if in_ablation else "warmup"
            log(f"[train:{model_name}] ep {ep}/{args.epochs} "
                f"avg_loss={train_loss_epoch:.4g} [{phase_label}]")

        if ep in ckpt_epochs:
            log(f"[ckpt:{model_name}] diagnostics at epoch={ep} ...")
            row = run_checkpoint_diagnostics(
                args, model, model_name, mdir, ep,
                Xtr_cpu, Ytr_cpu, Xdg_cpu,
                device=device, fit_lags=fit_lags,
                step=global_step,
                alpha_batch_size_override=current_bs,
                alpha_grad_clip_override=current_clip,
                alpha_winsorize_pct=current_winsorize,
            )
            append_csv_row(traj_csv, [row[k] for k in TRAJ_COLS])
            trajectory_rows.append(row)

            # Save model checkpoint at warmup boundary
            if ep == warmup_epochs and warmup_epochs > 0:
                ckpt_path = os.path.join(mdir, "model_checkpoints",
                                         f"ckpt_warmup_{ep:04d}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                log(f"[ckpt:{model_name}] saved warmup checkpoint at epoch={ep}")

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
# Run all models for one condition
# ============================================================

def _condition_dir_name(condition: str, condition_value) -> str:
    """Build condition directory name matching orchestrator's naming convention."""
    if condition == "baseline":
        return "condition_baseline"
    elif condition == "batch_ablation":
        return f"condition_batch_ablation_{int(condition_value)}"
    elif condition == "clip_ablation":
        formatted = f"{float(condition_value):.4f}".rstrip('0').rstrip('.')
        return f"condition_clip_ablation_{formatted}"
    elif condition == "winsorize_ablation":
        return f"condition_winsorize_ablation_{int(condition_value)}"
    else:
        return f"condition_{condition}_{condition_value}"


def run_for_condition(
    args,
    condition: str,
    condition_value,
    outdir: str,
    Xtr_cpu: torch.Tensor,
    Ytr_cpu: torch.Tensor,
    Xdg_cpu: torch.Tensor,
    device: torch.device,
    fit_lags: np.ndarray,
    ells: np.ndarray,
) -> None:
    """Run all models for a single (condition, condition_value) pair.

    Directory layout: outdir/<model>/<condition_dir>/...
    This matches the convention expected by main_exp2.py aggregation.
    """

    cond_tag = _condition_dir_name(condition, condition_value)
    warmup = int(getattr(args, 'warmup_epochs', 0))

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    for mname in models:
        # model-first layout: outdir/<model>/condition_<...>/
        mdir = os.path.join(outdir, mname, cond_tag)
        os.makedirs(mdir, exist_ok=True)

        # Save condition metadata per (model, condition) dir
        with open(os.path.join(mdir, "condition_info.json"), "w") as jf:
            json.dump({
                "condition": condition,
                "condition_value": float(condition_value) if isinstance(condition_value, (int, float)) else str(condition_value),
                "warmup_epochs": warmup,
                "mode": "warm_start" if warmup > 0 else "from_scratch",
            }, jf, indent=2)

        model = build_model(
            mname, args.D, args.H,
            const_s=args.const_s, ln=args.layernorm,
        ).to(device)

        # Determine ablation parameters
        abl_batch_size = None
        abl_grad_clip = None
        winsorize_pct = None

        if condition == "batch_ablation":
            abl_batch_size = int(condition_value)
        elif condition == "clip_ablation":
            abl_grad_clip = float(condition_value)
        elif condition == "winsorize_ablation":
            winsorize_pct = float(condition_value)
            # Disable standard clipping during winsorization
            abl_grad_clip = 0.0

        log(f"[run] model={mname} condition={condition} "
            f"value={condition_value} warmup={warmup}")

        train_with_phase_tracking_ablation(
            args, model, mname, mdir,
            Xtr_cpu, Ytr_cpu, Xdg_cpu,
            device=device, fit_lags=fit_lags, ells=ells,
            ablation_batch_size=abl_batch_size,
            ablation_grad_clip=abl_grad_clip,
            winsorize_pct=winsorize_pct,
            warmup_epochs=warmup,
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Anti-Collapse Experiment 3: Stochastic Forcing Ablation"
    )

    # Output / seeds
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--models", type=str, default="diag,gru,lstm")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--w_seed", type=int, default=41)

    # Data
    p.add_argument("--Nseq_train", type=int, default=8000)
    p.add_argument("--Nseq_diag", type=int, default=8000)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--H", type=int, default=512)

    # Optimization (baseline)
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
    p.add_argument("--task_variant", type=str, default="fixed",
                   choices=["fixed", "heavy_tail"],
                   help="fixed uses --task_lags/--task_coeffs; heavy_tail samples truncated-Pareto lags")
    p.add_argument("--task_alpha", type=float, default=1.0,
                   help="Tail index alpha_task for the heavy_tail task variant")
    p.add_argument("--task_lag_min", type=int, default=8)
    p.add_argument("--task_lag_max", type=int, default=384)
    p.add_argument("--task_K", type=int, default=8,
                   help="Number of target lags sampled in the heavy_tail task variant")
    p.add_argument("--task_coeff_base", type=float, default=0.6)
    p.add_argument("--task_coeff_decay", type=float, default=0.85)
    p.add_argument("--task_lag_seed", type=int, default=20260410,
                   help="Seed for sampling the lag set in the heavy_tail task variant")

    # Checkpointing
    p.add_argument("--diag_batch_size", type=int, default=256)
    p.add_argument("--checkpoint_every", type=int, default=50)

    # Alpha estimation
    p.add_argument("--alpha_n_grad_batches_ckpt", type=int, default=256)
    p.add_argument("--alpha_grad_batch_size", type=int, default=256)
    p.add_argument("--alpha_use_grad_clip", action="store_true")
    p.add_argument("--alpha_n_directions", type=int, default=5)
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

    # Ablation condition
    p.add_argument("--condition", type=str, default="baseline",
                   choices=["baseline", "batch_ablation", "clip_ablation",
                            "winsorize_ablation"])
    p.add_argument("--ablation_values", type=str, default="",
                   help="Comma-separated values for the ablation")

    # Warm-start
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Number of normal training epochs before applying ablation. "
                        "0 = from-scratch ablation (default).")

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

    # Determine conditions to run
    if args.condition == "baseline":
        conditions_to_run = [("baseline", None)]
    elif args.condition == "batch_ablation":
        if args.ablation_values:
            batch_sizes = [int(s) for s in args.ablation_values.split(",") if s.strip()]
        else:
            batch_sizes = [2048, 4096, 8192]
        conditions_to_run = [("batch_ablation", bs) for bs in batch_sizes]
    elif args.condition == "clip_ablation":
        if args.ablation_values:
            clip_vals = [float(s) for s in args.ablation_values.split(",") if s.strip()]
        else:
            clip_vals = [0.1, 0.01, 0.001]
        conditions_to_run = [("clip_ablation", cv) for cv in clip_vals]
    elif args.condition == "winsorize_ablation":
        if args.ablation_values:
            winsorize_pcts = [float(s) for s in args.ablation_values.split(",") if s.strip()]
        else:
            winsorize_pcts = [95.0, 90.0, 80.0]
        conditions_to_run = [("winsorize_ablation", wp) for wp in winsorize_pcts]
    else:
        raise ValueError(f"Unknown condition {args.condition}")

    for cond, cond_val in conditions_to_run:
        log(f"[run] condition={cond} value={cond_val}")
        run_for_condition(
            args, cond, cond_val, args.outdir,
            Xtr_cpu, Ytr_cpu, Xdg_cpu,
            device=device, fit_lags=fit_lags, ells=ells,
        )

    log("Done.")


if __name__ == "__main__":
    main()
