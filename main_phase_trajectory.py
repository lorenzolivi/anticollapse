#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — phase-trajectory orchestration engine
=====================================================

Runs the per-model phase-trajectory pipeline for multiple seeds:
  1) For each seed: run the unified run_phase_trajectory.py for all models
  2) After all seeds: aggregate phase trajectories across seeds (mean ± stderr)
  3) Run the plotting pipeline on the aggregated results

In the current manuscript this engine is used by Experiment 1
(ConstGate structural negative control; models=const) and Experiment 2
(capacity ladder / dynamical phase trajectory; models selected by launcher).

Usage:
  python main_phase_trajectory.py --outdir results/exp2_phase --seeds 42,123,321

This script calls run_phase_trajectory.py as a subprocess.

Directory layout produced:
  <outdir>/<optimizer>/
      seed_0042/{const,shared,diag,gru,lstm}/...
      seed_0123/{const,shared,diag,gru,lstm}/...
      seed_0321/{const,shared,diag,gru,lstm}/...
      aggregated/{const,shared,diag,gru,lstm}/phase_trajectory.csv   (mean ± se)
      plots/...                                                       (final plots)
"""

import argparse, os, sys, subprocess, json, csv, math, shutil
from datetime import datetime
from typing import List, Dict

import numpy as np

from seed_utils import (
    write_csv, PHASE_TRAJ_COLS, NUMERIC_COLS, STRING_COLS,
    load_phase_trajectory_csv, load_learning_curve_csv,
    aggregate_trajectories as _su_aggregate_trajectories,
    aggregate_learning_curves as _su_aggregate_learning_curves,
    aggregate_numeric_csvs,
    aggregate_envelope_fit_jsons,
    aggregate_gelr_fit_jsons,
    aggregate_final_phase_jsons,
    find_file_in_seed_dir,
    write_json,
)

# ============================================================
# Utilities
# ============================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ============================================================
# Step 1: Run simulations
# ============================================================

def run_simulation(script_name: str, outdir: str, common_args: Dict, extra_args: Dict,
                   seed: int, w_seed: int, log_file: str):
    """
    Run a simulation script as a subprocess.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Simulation script not found: {script_path}")

    cmd = [sys.executable, script_path, "--outdir", outdir, "--seed", str(seed), "--w_seed", str(w_seed)]

    for k, v in common_args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    for k, v in extra_args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    log(f"[CMD] {' '.join(cmd)}")
    log(f"[LOG] -> {log_file}")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        log(f"[WARN] {script_name} exited with code {result.returncode}. Check {log_file}")
        return False
    return True


# ============================================================
# Step 2: Aggregate phase trajectories across seeds
# ============================================================

# PHASE_TRAJ_COLS, NUMERIC_COLS, STRING_COLS imported from seed_utils


def aggregate_trajectories(seed_dirs: List[str], model_name: str) -> Dict:
    """Delegate to seed_utils.aggregate_trajectories."""
    return _su_aggregate_trajectories(seed_dirs, model_name)


def save_aggregated_trajectory(agg: Dict, outpath: str):
    """Save aggregated trajectory to CSV."""
    if not agg or "epoch" not in agg:
        return
    epochs = agg["epoch"]
    header = ["epoch"]
    if "step" in agg:
        header.append("step")
    for col in NUMERIC_COLS:
        header.append(f"{col}_mean")
        header.append(f"{col}_se")

    rows = []
    for i in range(len(epochs)):
        row = [int(epochs[i])]
        if "step" in agg:
            step_val = agg["step"][i]
            row.append(int(round(step_val)) if np.isfinite(step_val) else "")
        for col in NUMERIC_COLS:
            row.append(float(agg[f"{col}_mean"][i]))
            row.append(float(agg[f"{col}_se"][i]))
        rows.append(row)

    write_csv(outpath, header, rows)


def create_aggregated_phase_trajectory(agg: Dict, outpath: str):
    """
    Write a standard phase_trajectory.csv from aggregated means so the
    existing plotting scripts can consume it directly.

    Each data row is built by iterating PHASE_TRAJ_COLS in order and
    dispatching by column type, so the data row matches the header
    position-for-position. Earlier versions wrote epoch, step, all
    NUMERIC_COLS, then all STRING_COLS, which permuted the
    {alpha_method, n_samples, beta_env, beta_env_r2} block relative
    to the interleaved header order in PHASE_TRAJ_COLS and produced
    column-shifted aggregated CSVs (e.g.\\ string ``ecf'' landing in
    the beta_env_r2 column).
    """
    if not agg or "epoch" not in agg:
        return
    epochs = agg["epoch"]
    n = len(epochs)
    header = list(PHASE_TRAJ_COLS)
    rows = []
    for i in range(n):
        row = []
        for col in PHASE_TRAJ_COLS:
            if col == "epoch":
                row.append(int(epochs[i]))
            elif col == "step":
                step_val = agg["step"][i] if "step" in agg else np.nan
                row.append(int(round(step_val)) if np.isfinite(step_val) else "")
            elif col in STRING_COLS:
                vals = agg.get(col, [""] * n)
                row.append(str(vals[i]) if i < len(vals) else "")
            else:
                vals = agg.get(f"{col}_mean", agg.get(col, np.full(n, np.nan)))
                row.append(float(vals[i]))
        rows.append(row)

    write_csv(outpath, header, rows)


# ============================================================
# Step 2b: Aggregate learning curves across seeds
# ============================================================

def aggregate_learning_curves(seed_dirs: List[str], model_name: str) -> Dict:
    """Delegate to seed_utils.aggregate_learning_curves."""
    return _su_aggregate_learning_curves(seed_dirs, model_name)


def aggregate_final_artifacts(seed_dirs: List[str], model_name: str, outdir: str):
    """
    Aggregate final-only artifacts across seeds.

    Raw checkpoint directories remain single-seed exemplar copies for auditability,
    but final envelope / phase artifacts should reflect all available seeds.
    """
    os.makedirs(outdir, exist_ok=True)

    csv_specs = [
        ("_envelope.csv", ["ell"]),
        ("_envelope_audit.csv", ["ell"]),
        ("_envelope_fit_curves.csv", ["ell"]),
        ("_adaptive_base_rates.csv", ["neuron_q"]),
        ("_gelr_envelope_compare.csv", ["ell"]),
        ("_gelr_fit_curves.csv", ["ell"]),
    ]
    json_specs = [
        ("_envelope_fit.json", aggregate_envelope_fit_jsons),
        ("_gelr_fit.json", aggregate_gelr_fit_jsons),
        ("_final_phase.json", aggregate_final_phase_jsons),
    ]

    for suffix, key_cols in csv_specs:
        dst = os.path.join(outdir, f"{model_name}{suffix}")
        paths = []
        for seed_dir in seed_dirs:
            src = os.path.join(seed_dir, model_name, f"{model_name}{suffix}")
            if os.path.exists(src):
                paths.append(src)
        agg = aggregate_numeric_csvs(paths, key_cols)
        if agg is not None:
            header, rows = agg
            write_csv(dst, header, rows)
        elif os.path.exists(dst):
            os.remove(dst)

    for suffix, agg_fn in json_specs:
        dst = os.path.join(outdir, f"{model_name}{suffix}")
        paths = []
        for seed_dir in seed_dirs:
            src = os.path.join(seed_dir, model_name, f"{model_name}{suffix}")
            if os.path.exists(src):
                paths.append(src)
        agg = agg_fn(paths)
        if agg is not None:
            write_json(dst, agg)
        elif os.path.exists(dst):
            os.remove(dst)


def save_aggregated_learning_curve(agg: Dict, outpath: str):
    """Save aggregated learning curve to CSV."""
    if not agg or "epoch" not in agg:
        return
    epochs = agg["epoch"]
    header = ["epoch"]
    if "step" in agg:
        header.append("step")
    header.extend(["train_loss_mean", "train_loss_se"])
    rows = []
    for i in range(len(epochs)):
        row = [int(epochs[i])]
        if "step" in agg:
            step_val = agg["step"][i]
            row.append(int(round(step_val)) if np.isfinite(step_val) else "")
        row.extend([
            float(agg["train_loss_mean"][i]),
            float(agg["train_loss_se"][i]),
        ])
        rows.append(row)
    write_csv(outpath, header, rows)


# ============================================================
# Step 3: Run plotting
# ============================================================

def run_plotting_multiseed(agg_dir: str, outdir: str, dpi: int, extra_plot_args: Dict):
    """Run plot_exp1_all_multiseed.py on the aggregated results (preferred for multi-seed)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_script = os.path.join(script_dir, "plot_exp1_all_multiseed.py")

    if not os.path.exists(plot_script):
        log(f"[WARN] Multiseed plot driver not found: {plot_script}, falling back to single-seed driver")
        return run_plotting_single(agg_dir, outdir, dpi, extra_plot_args)

    cmd = [sys.executable, plot_script, "--agg_dir", agg_dir, "--outdir", outdir, "--dpi", str(dpi)]

    for k, v in extra_plot_args.items():
        cmd.extend([f"--{k}", str(v)])

    log(f"[PLOT-MS] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            log(f"  [plot-ms] {line}")
    if result.returncode != 0:
        log(f"[WARN] Multiseed plotting exited with code {result.returncode}")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n"):
                log(f"  [plot-ms-err] {line}")
        return False
    return True


def run_plotting_single(indir: str, outdir: str, dpi: int, extra_plot_args: Dict):
    """Run plot_exp1_all.py on the aggregated results (single-seed fallback)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_script = os.path.join(script_dir, "plot_exp1_all.py")

    if not os.path.exists(plot_script):
        log(f"[WARN] Plot driver not found: {plot_script}")
        return False

    cmd = [sys.executable, plot_script, "--indir", indir, "--outdir", outdir, "--dpi", str(dpi)]

    for k, v in extra_plot_args.items():
        cmd.extend([f"--{k}", str(v)])

    log(f"[PLOT] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            log(f"  [plot] {line}")
    if result.returncode != 0:
        log(f"[WARN] Plotting exited with code {result.returncode}")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n"):
                log(f"  [plot-err] {line}")
        return False
    return True


# ============================================================
# Parse args & main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anti-Collapse: multi-seed phase-trajectory orchestration")

    # output
    p.add_argument("--outdir", type=str, required=True,
                   help="Root output directory (e.g., results/exp1)")

    # seeds
    p.add_argument("--seeds", type=str, default="42,123,321",
                   help="Comma-separated list of seeds (e.g., 42,123,321)")
    p.add_argument("--w_seed_base", type=int, default=1000,
                   help="Base for projection direction seeds (w_seed = w_seed_base + seed)")

    # model selection (unified — no more baselines/lstmgru split)
    p.add_argument("--models", type=str, default="const,shared,diag,gru,lstm",
                   help="Comma-separated model names (all handled by unified run_phase_trajectory.py)")

    # shared simulation params
    p.add_argument("--Nseq_train", type=int, default=8000)
    p.add_argument("--Nseq_diag", type=int, default=8000)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--H", type=int, default=512)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd_momentum"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Heavy-tailed gradient-noise injection (Path A: enforced C1).
    p.add_argument("--inject_alpha_noise", type=float, default=0.0,
                   help="Multiplier on per-parameter (||grad||/sqrt(numel)) for "
                        "heavy-tailed gradient-noise injection. 0 disables (default).")
    p.add_argument("--inject_alpha", type=float, default=1.6,
                   help="Stability index α∈(0,2] of the injected α-stable noise.")
    p.add_argument("--inject_grad_seed_offset", type=int, default=1729,
                   help="Offset added to --seed for the injection RNG (reproducibility).")

    p.add_argument("--const_s", type=float, default=0.005)
    p.add_argument("--orth_init", action="store_true")
    p.add_argument("--layernorm", action="store_true")

    p.add_argument("--task_lags", type=str, default="32,64,128,192,256")
    p.add_argument("--task_coeffs", type=str, default="0.6,0.5,0.4,0.32,0.26")
    p.add_argument("--noise_std", type=float, default=0.3)

    # Task variant: fixed-lag (default) or heavy-tailed-lag (truncated Pareto on lags).
    p.add_argument("--task_variant", type=str, default="fixed",
                   choices=["fixed", "heavy_tail"])
    p.add_argument("--task_alpha", type=float, default=1.0)
    p.add_argument("--task_lag_min", type=int, default=8)
    p.add_argument("--task_lag_max", type=int, default=384)
    p.add_argument("--task_K", type=int, default=8)
    p.add_argument("--task_coeff_base", type=float, default=0.6)
    p.add_argument("--task_coeff_decay", type=float, default=0.85)
    p.add_argument("--task_lag_seed", type=int, default=20260410)

    p.add_argument("--diag_batch_size", type=int, default=256)
    p.add_argument("--checkpoint_every", type=int, default=50)

    # alpha estimation
    p.add_argument("--alpha_n_grad_batches_ckpt", type=int, default=256)
    p.add_argument("--alpha_grad_batch_size", type=int, default=256)
    p.add_argument("--alpha_use_grad_clip", action="store_true")
    p.add_argument("--alpha_n_directions", type=int, default=5)
    p.add_argument("--alpha_method", type=str, default="ecf",
                   choices=["mcculloch", "ecf"],
                   help="Alpha estimation method: ecf (default) or mcculloch")
    p.add_argument("--min_samples_alpha", type=int, default=500,
                   help="Minimum samples for reliable alpha estimation")

    # tau estimation
    p.add_argument("--tau_fit_lag_min", type=int, default=64)
    p.add_argument("--tau_fit_lag_max", type=int, default=256)
    p.add_argument("--tau_fit_num_lags", type=int, default=24)
    p.add_argument("--tau_ccdf_qmin", type=float, default=0.75)
    p.add_argument("--tau_ccdf_qmax", type=float, default=0.995)

    # bootstrap + phase classification
    p.add_argument("--beta_bootstrap_B", type=int, default=2000,
                   help="Number of bootstrap resamples for beta uncertainty (default: 2000)")
    p.add_argument("--beta_bootstrap_ci", type=float, default=0.90,
                   help="Confidence level for bootstrap stability interval (default: 0.90)")
    p.add_argument("--phase_r2_threshold", type=float, default=0.90,
                   help="Tail-fit R^2 threshold for phase classification (default: 0.90)")
    p.add_argument("--power_window_beta_min", type=float, default=0.10,
                   help="Minimum envelope exponent for a visible power-law window (default: 0.10)")
    p.add_argument("--power_window_min_points", type=int, default=8,
                   help="Minimum lag-grid points in the power-law window (default: 8)")
    p.add_argument("--power_window_min_fraction", type=float, default=0.05,
                   help="Minimum fraction of lag-grid points in the power-law window (default: 0.05)")

    # experiment data / deferred analysis saves
    p.add_argument("--save_checkpoint_ccdf", action="store_true")
    p.add_argument("--save_final_envelope", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--save_analysis_checkpoint", action="store_true",
                   help="Save final model+optimizer state for plot_only analysis")
    p.add_argument("--save_model_checkpoints", action="store_true")

    # lag grid (for envelopes)
    p.add_argument("--lag_min", type=int, default=4)
    p.add_argument("--lag_max", type=int, default=256)
    p.add_argument("--num_lags", type=int, default=128)

    # device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])

    # plotting
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--skip_plot", action="store_true", help="Skip plotting step")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip simulation, only aggregate + plot (assumes seeds already ran)")
    p.add_argument("--tau_cap", type=float, default=1e6, help="tau cap for plotting (forwarded to plot scripts)")
    p.add_argument("--min_r2", type=float, default=None, help="min_r2 for phase_summary plots")
    p.add_argument("--min_beta_r2", type=float, default=None, help="min_beta_r2 for tau_spectrum plots")

    return p.parse_args()


def build_common_args(args) -> Dict:
    """Build the common simulation arguments dict from parsed args."""
    d = {
        "Nseq_train": args.Nseq_train,
        "Nseq_diag": args.Nseq_diag,
        "T": args.T,
        "D": args.D,
        "H": args.H,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "inject_alpha_noise": args.inject_alpha_noise,
        "inject_alpha": args.inject_alpha,
        "inject_grad_seed_offset": args.inject_grad_seed_offset,
        "task_lags": args.task_lags,
        "task_coeffs": args.task_coeffs,
        "noise_std": args.noise_std,
        "task_variant": args.task_variant,
        "task_alpha": args.task_alpha,
        "task_lag_min": args.task_lag_min,
        "task_lag_max": args.task_lag_max,
        "task_K": args.task_K,
        "task_coeff_base": args.task_coeff_base,
        "task_coeff_decay": args.task_coeff_decay,
        "task_lag_seed": args.task_lag_seed,
        "diag_batch_size": args.diag_batch_size,
        "checkpoint_every": args.checkpoint_every,
        "alpha_n_grad_batches_ckpt": args.alpha_n_grad_batches_ckpt,
        "alpha_grad_batch_size": args.alpha_grad_batch_size,
        "alpha_n_directions": args.alpha_n_directions,
        "alpha_method": args.alpha_method,
        "min_samples_alpha": args.min_samples_alpha,
        "tau_fit_lag_min": args.tau_fit_lag_min,
        "tau_fit_lag_max": args.tau_fit_lag_max,
        "tau_fit_num_lags": args.tau_fit_num_lags,
        "tau_ccdf_qmin": args.tau_ccdf_qmin,
        "tau_ccdf_qmax": args.tau_ccdf_qmax,
        "lag_min": args.lag_min,
        "lag_max": args.lag_max,
        "num_lags": args.num_lags,
        "beta_bootstrap_B": args.beta_bootstrap_B,
        "beta_bootstrap_ci": args.beta_bootstrap_ci,
        "phase_r2_threshold": args.phase_r2_threshold,
        "power_window_beta_min": args.power_window_beta_min,
        "power_window_min_points": args.power_window_min_points,
        "power_window_min_fraction": args.power_window_min_fraction,
        "device": args.device,
        # boolean flags
        "orth_init": args.orth_init,
        "layernorm": args.layernorm,
        "alpha_use_grad_clip": args.alpha_use_grad_clip,
        "save_checkpoint_ccdf": args.save_checkpoint_ccdf,
        "save_final_envelope": args.save_final_envelope,
        "save_analysis_checkpoint": args.save_analysis_checkpoint,
        "save_model_checkpoints": args.save_model_checkpoints,
    }
    return d


def _load_run_config_for_plot_only(args):
    """Use the saved run config so analysis uses the exact simulation setup."""
    opt_dir = os.path.join(args.outdir, args.optimizer)
    cfg_candidates = [
        os.path.join(opt_dir, "phase_trajectory_config.json"),
        # Backward compatibility for result folders produced before the
        # role-based script rename.
        os.path.join(opt_dir, "main_exp1_config.json"),
    ]
    cfg_path = next((p for p in cfg_candidates if os.path.isfile(p)), cfg_candidates[0])
    if not args.plot_only or not os.path.isfile(cfg_path):
        return args
    with open(cfg_path) as jf:
        cfg = json.load(jf)
    preserve = {
        "outdir", "plot_only", "skip_plot", "dpi", "tau_cap",
        "min_r2", "min_beta_r2",
    }
    for key, value in cfg.items():
        if key not in preserve and hasattr(args, key):
            setattr(args, key, value)
    log(f"[plot_only] loaded run configuration from {cfg_path}")
    return args


def main():
    args = parse_args()
    args = _load_run_config_for_plot_only(args)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds specified.")

    opt_dir = os.path.join(args.outdir, args.optimizer)
    os.makedirs(opt_dir, exist_ok=True)

    all_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    log(f"Phase-trajectory orchestrator")
    log(f"Seeds: {seeds}")
    log(f"Models: {all_models}")
    log(f"Output: {opt_dir}")

    # Save orchestration config.  Preserve the simulation manifest during
    # plot_only analysis; write analysis settings separately.
    config_name = "phase_trajectory_analysis_config.json" if args.plot_only else "phase_trajectory_config.json"
    with open(os.path.join(opt_dir, config_name), "w") as jf:
        json.dump(vars(args), jf, indent=2)

    # ================================================================
    # STEP 1: Run simulations (per seed) — unified runner
    # ================================================================
    seed_dirs = []
    common_args = build_common_args(args)
    logs_dir = os.path.join(opt_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for seed in seeds:
        seed_tag = f"seed_{seed:04d}"
        seed_dir = os.path.join(opt_dir, seed_tag)
        seed_dirs.append(seed_dir)

        w_seed = args.w_seed_base + seed

        if not args.plot_only:
            run_log = os.path.join(logs_dir, f"{seed_tag}_run.log")
            run_extra = {
                "models": args.models,
                "const_s": args.const_s,
            }
            log(f"[seed={seed}] Running all models -> {seed_dir}")
            ok = run_simulation(
                "run_phase_trajectory.py",
                seed_dir, common_args, run_extra,
                seed=seed, w_seed=w_seed, log_file=run_log
            )
            if ok:
                log(f"[seed={seed}] OK")
            else:
                log(f"[seed={seed}] FAILED — see {run_log}")

    if args.plot_only:
        log("Running deferred final-envelope analysis from saved checkpoints ...")
        for seed in seeds:
            seed_tag = f"seed_{seed:04d}"
            seed_dir = os.path.join(opt_dir, seed_tag)
            w_seed = args.w_seed_base + seed
            analysis_log = os.path.join(logs_dir, f"{seed_tag}_analysis.log")
            run_extra = {
                "models": args.models,
                "const_s": args.const_s,
                "analysis_only": True,
            }
            ok = run_simulation(
                "run_phase_trajectory.py",
                seed_dir, common_args, run_extra,
                seed=seed, w_seed=w_seed, log_file=analysis_log,
            )
            if ok:
                log(f"[seed={seed}] analysis OK")
            else:
                log(f"[seed={seed}] analysis FAILED — see {analysis_log}")

    # ================================================================
    # STEP 2: Aggregate phase trajectories
    # ================================================================
    log("Aggregating phase trajectories across seeds ...")

    agg_dir = os.path.join(opt_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    for model_name in all_models:
        agg = aggregate_trajectories(seed_dirs, model_name)
        if not agg:
            log(f"  [agg] {model_name}: no data found across seeds")
            continue

        n_seeds = agg.get("n_seeds", 0)
        log(f"  [agg] {model_name}: {n_seeds} seeds, {len(agg['epoch'])} epochs")

        mdir = os.path.join(agg_dir, model_name)
        os.makedirs(mdir, exist_ok=True)

        # Save full aggregated CSV (with _mean and _se columns)
        save_aggregated_trajectory(agg, os.path.join(mdir, "phase_trajectory_aggregated.csv"))

        # Save a standard phase_trajectory.csv (means only) for plotting scripts
        create_aggregated_phase_trajectory(agg, os.path.join(mdir, "phase_trajectory.csv"))

    # ================================================================
    # STEP 2b: Aggregate learning curves across seeds
    # ================================================================
    log("Aggregating learning curves across seeds ...")
    for model_name in all_models:
        lc_agg = aggregate_learning_curves(seed_dirs, model_name)
        if not lc_agg:
            log(f"  [lc-agg] {model_name}: no learning curve data found")
            continue
        n_s = lc_agg.get("n_seeds", 0)
        log(f"  [lc-agg] {model_name}: {n_s} seeds, {len(lc_agg['epoch'])} epochs")
        mdir = os.path.join(agg_dir, model_name)
        os.makedirs(mdir, exist_ok=True)
        save_aggregated_learning_curve(lc_agg, os.path.join(mdir, "learning_curve_aggregated.csv"))

    # ================================================================
    # STEP 2c: Copy raw checkpoint artifacts + aggregate final-only outputs
    # ================================================================
    # Raw checkpoint directories are copied from one exemplar seed for audit plots.
    # Final envelope / phase artifacts are present only after the deferred
    # plot_only analysis pass; when present, aggregate them across seeds below.
    log("Copying exemplar checkpoint data and aggregating final artifacts ...")
    last_seed_dir = seed_dirs[-1] if seed_dirs else None
    if last_seed_dir:
        CHECKPOINT_DIRS = [
            "checkpoint_alpha",
            "checkpoint_taus",
            "checkpoint_tau_ccdf",
            "checkpoint_tau_tail",
            "checkpoint_beta_bootstrap",
        ]

        for model_name in all_models:
            mdir_agg = os.path.join(agg_dir, model_name)
            # New layout: models are directly under seed_dir/<model>/
            src_model_dir = os.path.join(last_seed_dir, model_name)
            if not os.path.isdir(src_model_dir):
                continue

            # Copy checkpoint directories (remove stale first)
            for ckpt_name in CHECKPOINT_DIRS:
                src = os.path.join(src_model_dir, ckpt_name)
                dst = os.path.join(mdir_agg, ckpt_name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)

            aggregate_final_artifacts(seed_dirs, model_name, mdir_agg)

    # ================================================================
    # STEP 2d: Summary table (final-epoch diagnostics per model)
    # ================================================================
    log("Generating summary table (final-epoch diagnostics) ...")
    summary_cols = ["model", "n_seeds",
                    "beta_hat_mean", "beta_hat_se",
                    "beta_median_mean", "beta_median_se",
                    "p_beta_lt1_mean", "p_beta_lt1_se",
                    "beta_env_mean", "beta_env_se",
                    "alpha_hat_mean", "alpha_hat_se",
                    "tau_mean_mean", "tau_mean_se",
                    "delta_zeta_mean", "delta_zeta_se"]
    summary_rows = []
    for model_name in all_models:
        agg = aggregate_trajectories(seed_dirs, model_name)
        if not agg or "epoch" not in agg or len(agg["epoch"]) == 0:
            continue
        # Use the last epoch
        n_s = agg.get("n_seeds", 0)
        row = [model_name, n_s]
        for col in ["beta_hat", "beta_median", "p_beta_lt1", "beta_env", "alpha_hat", "tau_mean", "delta_zeta"]:
            m_key = f"{col}_mean"
            se_key = f"{col}_se"
            if m_key in agg and len(agg[m_key]) > 0:
                row.append(float(agg[m_key][-1]))
                row.append(float(agg[se_key][-1]))
            else:
                row.append(float("nan"))
                row.append(float("nan"))
        summary_rows.append(row)

    summary_path = os.path.join(agg_dir, "summary_table.csv")
    write_csv(summary_path, summary_cols, summary_rows)
    log(f"  Summary table saved to {summary_path}")

    # Print summary to stdout
    log(f"  {'Model':<12} {'n':>4}  {'beta_hat':>8}  {'b_med':>8}  {'P(b<1)':>8}  {'beta_env':>8}  {'alpha_hat':>8}  {'tau_mean':>10}  {'Dzeta':>8}")
    log(f"  {'-'*12} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    for r in summary_rows:
        model_name, n_s = r[0], r[1]
        bh_m, bh_se = r[2], r[3]
        bmed_m, bmed_se = r[4], r[5]
        pbl_m, pbl_se = r[6], r[7]
        be_m, be_se = r[8], r[9]
        ah_m, ah_se = r[10], r[11]
        tm_m, tm_se = r[12], r[13]
        dz_m, dz_se = r[14], r[15]
        log(f"  {model_name:<12} {n_s:>4}  {bh_m:>5.2f}+/-{bh_se:.2f}  {bmed_m:>5.2f}+/-{bmed_se:.2f}  {pbl_m:>5.2f}+/-{pbl_se:.2f}  {be_m:>5.2f}+/-{be_se:.2f}  {ah_m:>5.2f}+/-{ah_se:.2f}  {tm_m:>7.1f}+/-{tm_se:.1f}  {dz_m:>5.2f}+/-{dz_se:.2f}")

    # ================================================================
    # STEP 2e: Aggregate threshold-crossing results across seeds
    # ================================================================
    log("Aggregating threshold-crossing results ...")
    for model_name in all_models:
        tcross_per_seed = []
        for sd in seed_dirs:
            tc_path = os.path.join(sd, model_name,
                                   f"{model_name}_threshold_crossing.json")
            if os.path.isfile(tc_path):
                with open(tc_path) as f:
                    tcross_per_seed.append(json.load(f))

        if not tcross_per_seed:
            continue

        observed_crossings = [tc for tc in tcross_per_seed if tc.get("crossed")]
        left_censored = [tc for tc in tcross_per_seed if tc.get("left_censored")]
        right_censored = [tc for tc in tcross_per_seed if tc.get("right_censored")]

        agg_tcross = {
            "model": model_name,
            "n_seeds": len(tcross_per_seed),
            "n_crossed": len(observed_crossings),
            "n_observed_crossings": len(observed_crossings),
            "n_left_censored": len(left_censored),
            "n_right_censored": len(right_censored),
            "per_seed": tcross_per_seed,
        }

        if observed_crossings:
            steps = [tc["t_cross_step"] for tc in observed_crossings]
            alphas = [tc["alpha_at_cross"] for tc in observed_crossings
                      if tc.get("alpha_at_cross") is not None]
            betas = [tc["beta_env_at_cross"] for tc in observed_crossings
                     if tc.get("beta_env_at_cross") is not None]
            observed_mean = float(np.mean(steps))
            observed_se = (
                float(np.std(steps, ddof=1) / np.sqrt(len(steps)))
                if len(steps) > 1 else 0.0
            )
            agg_tcross["observed_t_cross_step_mean"] = observed_mean
            agg_tcross["observed_t_cross_step_se"] = observed_se
            # Backward-compatible aliases; these refer to observed crossings only.
            agg_tcross["t_cross_step_mean"] = observed_mean
            agg_tcross["t_cross_step_se"] = observed_se
            if alphas:
                agg_tcross["alpha_at_observed_cross_mean"] = float(np.mean(alphas))
                agg_tcross["alpha_at_cross_mean"] = float(np.mean(alphas))
            if betas:
                agg_tcross["beta_env_at_observed_cross_mean"] = float(np.mean(betas))
                agg_tcross["beta_env_at_cross_mean"] = float(np.mean(betas))
            log(
                f"  {model_name}: observed={len(observed_crossings)}, "
                f"left-censored={len(left_censored)}, "
                f"right-censored={len(right_censored)}, "
                f"mean observed step={observed_mean:.0f}"
            )
        else:
            log(
                f"  {model_name}: no observed crossings, "
                f"left-censored={len(left_censored)}, "
                f"right-censored={len(right_censored)}"
            )

        mdir_agg = os.path.join(agg_dir, model_name)
        os.makedirs(mdir_agg, exist_ok=True)
        tc_out = os.path.join(mdir_agg, f"{model_name}_threshold_crossing.json")
        with open(tc_out, "w") as f:
            json.dump(agg_tcross, f, indent=2, default=str)

    # ================================================================
    # STEP 3: Plot (multi-seed aware)
    # ================================================================
    if not args.skip_plot:
        plots_dir = os.path.join(opt_dir, "plots")

        extra_plot_args = {}
        if args.tau_cap is not None:
            extra_plot_args["tau_cap"] = args.tau_cap
        if args.min_r2 is not None:
            extra_plot_args["min_r2"] = args.min_r2
        if args.min_beta_r2 is not None:
            extra_plot_args["min_beta_r2"] = args.min_beta_r2

        log(f"Running multi-seed plotting pipeline on {agg_dir} -> {plots_dir}")
        run_plotting_multiseed(agg_dir, plots_dir, args.dpi, extra_plot_args)

    log("Done.")


if __name__ == "__main__":
    main()
