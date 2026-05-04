#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-Collapse — Experiment 3 orchestration script
=================================================

Runs the full Stochastic Forcing Ablation experiment pipeline for multiple seeds:
  1) For each seed: for each condition, run the unified run_exp2.py
  2) After all seeds: aggregate phase trajectories across seeds (mean ± stderr)
  3) Run the plotting pipeline on the aggregated results

Supports warm-start ablation: train normally for --warmup_epochs, then apply
the intervention. Pass --warmup_epochs 0 (default) for from-scratch ablation.

Usage:
  # From-scratch ablation (default)
  python main_exp2.py --outdir results/exp3_forcing --seeds 42,123,321

  # Warm-start ablation (250 epochs normal, then intervene)
  python main_exp2.py --outdir results/exp3_forcing --seeds 42,123,321 --warmup_epochs 250

This script calls run_exp2.py as a subprocess.

Directory layout produced:
  <outdir>/<optimizer>/
      seed_0042/<model>/condition_baseline/...
      seed_0042/<model>/condition_batch_ablation_2048/...
      seed_0042/<model>/condition_batch_ablation_4096/...
      ...
      aggregated/<model>/condition_baseline/phase_trajectory.csv
      aggregated/<model>/condition_batch_ablation_2048/phase_trajectory.csv
      ...
      plots/...                                                       (final plots)
"""

import argparse, os, sys, subprocess, json, csv, math, shutil
from datetime import datetime
from typing import List, Dict, Tuple

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
    _consensus_label,
    _nanmean_no_warn,
)
from diagnostics import localize_threshold_bracket

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


def _condition_dir_name(condition_name: str, condition_value: str) -> str:
    """
    Build the condition directory name matching the runner's naming convention.

    Runner uses:
      baseline        -> condition_baseline
      batch_ablation  -> condition_batch_ablation_{int(value)}
      clip_ablation   -> condition_clip_ablation_{value:.4f} (trailing zeros stripped)
      winsorize       -> condition_winsorize_ablation_{int(value)}
    """
    if condition_name == "baseline":
        return "condition_baseline"
    elif condition_name == "batch_ablation":
        return f"condition_batch_ablation_{int(float(condition_value))}"
    elif condition_name == "clip_ablation":
        formatted = f"{float(condition_value):.4f}".rstrip('0').rstrip('.')
        return f"condition_clip_ablation_{formatted}"
    elif condition_name == "winsorize_ablation":
        return f"condition_winsorize_ablation_{int(float(condition_value))}"
    else:
        return f"condition_{condition_name}_{condition_value}"


def find_file_for_condition(seed_dirs: List[str], model_name: str, condition_name: str,
                             condition_value: str, filename: str) -> List[str]:
    """
    Find a file across all seed directories for a specific (model, condition, value).

    Looks in: seed_dir/<model>/<condition_dir>/<filename>

    Returns list of file paths that exist.
    """
    cond_dir = _condition_dir_name(condition_name, condition_value)
    filepaths = []
    for seed_dir in seed_dirs:
        filepath = os.path.join(seed_dir, model_name, cond_dir, filename)
        if os.path.exists(filepath):
            filepaths.append(filepath)
    return filepaths


def aggregate_trajectories_for_condition(seed_dirs: List[str], model_name: str,
                                         condition_name: str, condition_value: str) -> Dict:
    """
    Aggregate phase trajectories for a specific (model, condition, value) across seeds.

    Similar to seed_utils.aggregate_trajectories but looks in condition-specific subdirs.
    """
    filepaths = find_file_for_condition(seed_dirs, model_name, condition_name,
                                        condition_value, "phase_trajectory.csv")

    if not filepaths:
        return {}

    data_per_seed = []
    for fp in filepaths:
        try:
            traj = load_phase_trajectory_csv(fp)
            if traj:
                data_per_seed.append(traj)
        except Exception as e:
            log(f"  [warn] failed to load {fp}: {e}")

    if not data_per_seed:
        return {}

    n_seeds = len(data_per_seed)
    has_step = any(
        np.any(np.isfinite(np.asarray(seed_data.get("step", []), dtype=float)))
        for seed_data in data_per_seed
    )
    index_col = "step" if has_step else "epoch"

    index_sets = []
    for seed_data in data_per_seed:
        idx = np.asarray(seed_data.get(index_col, []), dtype=float)
        idx = idx[np.isfinite(idx)].astype(int)
        index_sets.append(set(idx.tolist()))

    common_index = sorted(set.intersection(*index_sets)) if index_sets else []
    if not common_index:
        common_index = sorted(set.union(*index_sets)) if index_sets else []

    index_values = np.array(common_index, dtype=int)
    n_points = len(index_values)

    # Aggregate numeric columns
    agg = {index_col: index_values, "n_seeds": n_seeds}

    # Preserve both epoch and step columns for downstream plotting.
    for aux_col in ("epoch", "step"):
        aux_mat = np.full((n_seeds, n_points), np.nan, dtype=float)
        for si, seed_data in enumerate(data_per_seed):
            idx = np.asarray(seed_data.get(index_col, []), dtype=float)
            aux = np.asarray(seed_data.get(aux_col, []), dtype=float)
            pos = {int(v): i for i, v in enumerate(index_values)}
            for j, key_val in enumerate(idx):
                if np.isfinite(key_val) and int(key_val) in pos and j < aux.size and np.isfinite(aux[j]):
                    aux_mat[si, pos[int(key_val)]] = aux[j]
        aux_mean, _ = _nanmean_no_warn(aux_mat)
        agg[aux_col] = np.rint(aux_mean).astype(int) if np.all(np.isfinite(aux_mean)) else aux_mean

    for col in NUMERIC_COLS:
        col_values = np.full((n_seeds, n_points), np.nan, dtype=float)
        for si, seed_data in enumerate(data_per_seed):
            idx = np.asarray(seed_data.get(index_col, []), dtype=float)
            vals = np.asarray(seed_data.get(col, []), dtype=float)
            pos = {int(v): i for i, v in enumerate(index_values)}
            for j, key_val in enumerate(idx):
                if np.isfinite(key_val) and int(key_val) in pos and j < vals.size:
                    col_values[si, pos[int(key_val)]] = vals[j]

        col_mean, col_se = _nanmean_no_warn(col_values)
        agg[f"{col}_mean"] = col_mean.tolist()
        agg[f"{col}_se"] = col_se.tolist()

    # Keep string fields only when they agree across seeds; otherwise mark mixed.
    for col in STRING_COLS:
        vals = [""] * n_points
        pos = {int(v): i for i, v in enumerate(index_values)}
        for key_val in index_values:
            labels = []
            for seed_data in data_per_seed:
                idx = np.asarray(seed_data.get(index_col, []), dtype=float)
                str_vals = seed_data.get(col, [""] * len(idx))
                for j, idx_val in enumerate(idx):
                    if np.isfinite(idx_val) and int(idx_val) == int(key_val) and j < len(str_vals):
                        value = str(str_vals[j]).strip()
                        if value:
                            labels.append(value)
                        break
            vals[pos[int(key_val)]] = _consensus_label(labels)["label"]
        agg[col] = vals

    return agg


def aggregate_learning_curves_for_condition(seed_dirs: List[str], model_name: str,
                                            condition_name: str, condition_value: str) -> Dict:
    """
    Aggregate learning curves for a specific (model, condition, value) across seeds.
    """
    filepaths = find_file_for_condition(seed_dirs, model_name, condition_name,
                                        condition_value, f"{model_name}_learning_curve.csv")

    if not filepaths:
        return {}

    data_per_seed = []
    for fp in filepaths:
        try:
            lc = load_learning_curve_csv(fp)
            if lc:
                data_per_seed.append(lc)
        except Exception as e:
            log(f"  [warn] failed to load learning curve {fp}: {e}")

    if not data_per_seed:
        return {}

    n_seeds = len(data_per_seed)
    has_step = any(
        np.any(np.isfinite(np.asarray(seed_data.get("step", []), dtype=float)))
        for seed_data in data_per_seed
    )
    index_col = "step" if has_step else "epoch"

    index_sets = []
    for seed_data in data_per_seed:
        idx = np.asarray(seed_data.get(index_col, []), dtype=float)
        idx = idx[np.isfinite(idx)].astype(int)
        index_sets.append(set(idx.tolist()))

    common_index = sorted(set.intersection(*index_sets)) if index_sets else []
    if not common_index:
        common_index = sorted(set.union(*index_sets)) if index_sets else []

    index_values = np.array(common_index, dtype=int)
    n_points = len(index_values)

    agg = {index_col: index_values, "n_seeds": n_seeds}

    for aux_col in ("epoch", "step"):
        aux_mat = np.full((n_seeds, n_points), np.nan, dtype=float)
        for si, seed_data in enumerate(data_per_seed):
            idx = np.asarray(seed_data.get(index_col, []), dtype=float)
            aux = np.asarray(seed_data.get(aux_col, []), dtype=float)
            pos = {int(v): i for i, v in enumerate(index_values)}
            for j, key_val in enumerate(idx):
                if np.isfinite(key_val) and int(key_val) in pos and j < aux.size and np.isfinite(aux[j]):
                    aux_mat[si, pos[int(key_val)]] = aux[j]
        aux_mean, _ = _nanmean_no_warn(aux_mat)
        agg[aux_col] = np.rint(aux_mean).astype(int) if np.all(np.isfinite(aux_mean)) else aux_mean

    train_loss_vals = np.full((n_seeds, n_points), np.nan, dtype=float)
    for si, seed_data in enumerate(data_per_seed):
        idx = np.asarray(seed_data.get(index_col, []), dtype=float)
        vals = np.asarray(seed_data.get("train_loss", []), dtype=float)
        pos = {int(v): i for i, v in enumerate(index_values)}
        for j, key_val in enumerate(idx):
            if np.isfinite(key_val) and int(key_val) in pos and j < vals.size:
                train_loss_vals[si, pos[int(key_val)]] = vals[j]

    loss_mean, loss_se = _nanmean_no_warn(train_loss_vals)
    agg["train_loss_mean"] = loss_mean.tolist()
    agg["train_loss_se"] = loss_se.tolist()

    return agg


def aggregate_final_artifacts_for_condition(seed_dirs: List[str], model_name: str,
                                            condition_name: str, condition_value: str,
                                            outdir: str):
    """
    Aggregate final-only artifacts across seeds for a single ablation condition.
    """
    os.makedirs(outdir, exist_ok=True)
    cond_dir = _condition_dir_name(condition_name, condition_value)

    csv_specs = [
        ("_envelope.csv", ["ell"]),
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
            src = os.path.join(seed_dir, model_name, cond_dir, f"{model_name}{suffix}")
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
            src = os.path.join(seed_dir, model_name, cond_dir, f"{model_name}{suffix}")
            if os.path.exists(src):
                paths.append(src)
        agg = agg_fn(paths)
        if agg is not None:
            write_json(dst, agg)
        elif os.path.exists(dst):
            os.remove(dst)


def save_aggregated_trajectory(agg: Dict, outpath: str):
    """Save aggregated trajectory to CSV."""
    if not agg or "epoch" not in agg:
        return
    epochs = agg["epoch"]
    header = ["epoch", "step"]
    for col in NUMERIC_COLS:
        header.append(f"{col}_mean")
        header.append(f"{col}_se")

    rows = []
    for i in range(len(epochs)):
        step_val = agg["step"][i] if "step" in agg else np.nan
        row = [
            int(round(epochs[i])) if np.isfinite(epochs[i]) else "",
            int(round(step_val)) if np.isfinite(step_val) else "",
        ]
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
                row.append(int(round(epochs[i])) if np.isfinite(epochs[i]) else "")
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

def save_aggregated_learning_curve(agg: Dict, outpath: str):
    """Save aggregated learning curve to CSV."""
    if not agg or "epoch" not in agg:
        return
    epochs = agg["epoch"]
    header = ["epoch", "step", "train_loss_mean", "train_loss_se"]
    rows = []
    for i in range(len(epochs)):
        step_val = agg["step"][i] if "step" in agg else np.nan
        rows.append([
            int(round(epochs[i])) if np.isfinite(epochs[i]) else "",
            int(round(step_val)) if np.isfinite(step_val) else "",
            float(agg["train_loss_mean"][i]),
            float(agg["train_loss_se"][i]),
        ])
    write_csv(outpath, header, rows)


# ============================================================
# Step 3: Run plotting
# ============================================================

def run_plotting_multiseed(agg_dir: str, outdir: str, dpi: int, extra_plot_args: Dict):
    """Run plot_exp2_ablation.py on the aggregated results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_script = os.path.join(script_dir, "plot_exp2_ablation.py")

    if not os.path.exists(plot_script):
        log(f"[WARN] Exp2 plot driver not found: {plot_script}")
        return False

    cmd = [sys.executable, plot_script, "--agg_dir", agg_dir, "--outdir", outdir, "--dpi", str(dpi)]

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
# Build experiment grid
# ============================================================

def build_condition_ablation_grid(args) -> List[Tuple[str, str]]:
    """
    Build list of (condition_name, ablation_value) tuples.

    - baseline: no ablation values
    - batch_ablation: one per value in batch_ablation_values
    - clip_ablation: one per value in clip_ablation_values
    - winsorize_ablation: one per value in winsorize_ablation_values
    """
    grid = []

    conditions = [c.strip().lower() for c in args.conditions.split(",") if c.strip()]

    for cond in conditions:
        if cond == "baseline":
            grid.append(("baseline", "baseline"))
        elif cond == "batch_ablation":
            values = [v.strip() for v in args.batch_ablation_values.split(",") if v.strip()]
            for val in values:
                grid.append(("batch_ablation", val))
        elif cond == "clip_ablation":
            values = [v.strip() for v in args.clip_ablation_values.split(",") if v.strip()]
            for val in values:
                grid.append(("clip_ablation", val))
        elif cond == "winsorize_ablation":
            values = [v.strip() for v in args.winsorize_ablation_values.split(",") if v.strip()]
            for val in values:
                grid.append(("winsorize_ablation", val))

    return grid


# ============================================================
# Parse args & main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anti-Collapse Exp2: multi-seed orchestration (forcing ablation)")

    # output
    p.add_argument("--outdir", type=str, required=True,
                   help="Root output directory (e.g., results/exp3_forcing)")

    # seeds
    p.add_argument("--seeds", type=str, default="42,123,321",
                   help="Comma-separated list of seeds (e.g., 42,123,321)")
    p.add_argument("--w_seed_base", type=int, default=1000,
                   help="Base for projection direction seeds (w_seed = w_seed_base + seed)")

    # model selection (only models that naturally anti-collapse)
    p.add_argument("--models", type=str, default="diag,gru,lstm",
                   help="Comma-separated list of models to ablate")

    # ablation conditions and values
    p.add_argument("--conditions", type=str, default="baseline,batch_ablation,clip_ablation,winsorize_ablation",
                   help="Comma-separated list of ablation conditions")
    p.add_argument("--batch_ablation_values", type=str, default="2048,4096,8192",
                   help="Comma-separated batch ablation values")
    p.add_argument("--clip_ablation_values", type=str, default="0.1,0.01,0.001",
                   help="Comma-separated clip ablation values")
    p.add_argument("--winsorize_ablation_values", type=str, default="95,90,80",
                   help="Comma-separated winsorize ablation values")

    # warm-start
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Number of normal training epochs before applying ablation. "
                        "0 = from-scratch ablation (default). "
                        "Positive value = warm-start: train normally for this many epochs, "
                        "then apply the intervention for the remaining epochs.")

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

    p.add_argument("--const_s", type=float, default=0.005)
    p.add_argument("--orth_init", action="store_true")
    p.add_argument("--layernorm", action="store_true")

    p.add_argument("--task_lags", type=str, default="32,64,128,192,256")
    p.add_argument("--task_coeffs", type=str, default="0.6,0.5,0.4,0.32,0.26")
    p.add_argument("--noise_std", type=float, default=0.3)
    p.add_argument("--task_variant", type=str, default="fixed",
                   choices=["fixed", "heavy_tail"],
                   help="fixed uses --task_lags/--task_coeffs; heavy_tail samples truncated-Pareto lags")
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

    # envelope + ccdf saves
    p.add_argument("--save_checkpoint_ccdf", action="store_true")
    p.add_argument("--save_final_envelope", action="store_true")

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
        "device": args.device,
        # warm-start
        "warmup_epochs": args.warmup_epochs,
        # boolean flags
        "orth_init": args.orth_init,
        "layernorm": args.layernorm,
        "alpha_use_grad_clip": args.alpha_use_grad_clip,
        "save_checkpoint_ccdf": args.save_checkpoint_ccdf,
        "save_final_envelope": args.save_final_envelope,
    }
    return d


def main():
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds specified.")

    opt_dir = os.path.join(args.outdir, args.optimizer)
    os.makedirs(opt_dir, exist_ok=True)

    log(f"Experiment 3 orchestrator (Stochastic Forcing Ablation)")
    log(f"Seeds: {seeds}")
    log(f"Output: {opt_dir}")
    if args.warmup_epochs > 0:
        log(f"Mode: WARM-START (warmup={args.warmup_epochs} epochs)")
    else:
        log(f"Mode: FROM-SCRATCH ablation")

    # Save orchestration config
    with open(os.path.join(opt_dir, "main_exp2_config.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)

    # Build experiment grid
    cond_ablation_grid = build_condition_ablation_grid(args)
    log(f"Ablation conditions: {cond_ablation_grid}")

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    log(f"Models: {models}")

    # Group grid by condition name for efficient subprocess calls
    # run_exp2.py handles multiple ablation_values per condition in one call
    cond_groups: Dict[str, List[str]] = {}
    for cond_name, cond_value in cond_ablation_grid:
        if cond_name not in cond_groups:
            cond_groups[cond_name] = []
        if cond_value != "baseline":
            cond_groups[cond_name].append(cond_value)

    # ================================================================
    # STEP 1: Run simulations (per seed, per condition)
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
            # For each condition group, call run_exp2.py once
            for cond_name, cond_values in cond_groups.items():
                runner_outdir = seed_dir
                runner_log = os.path.join(logs_dir, f"{seed_tag}_{cond_name}.log")

                runner_extra = {
                    "models": ",".join(models),
                    "condition": cond_name,
                    "const_s": args.const_s,
                }

                # For non-baseline conditions, pass all ablation values
                if cond_values:
                    runner_extra["ablation_values"] = ",".join(cond_values)

                log(f"[seed={seed}] Running {cond_name} "
                    f"values=[{','.join(cond_values) if cond_values else 'N/A'}] "
                    f"warmup={args.warmup_epochs} -> {runner_outdir}")
                ok = run_simulation(
                    "run_exp2.py",
                    runner_outdir, common_args, runner_extra,
                    seed=seed, w_seed=w_seed, log_file=runner_log
                )
                if ok:
                    log(f"[seed={seed}] {cond_name} OK")
                else:
                    log(f"[seed={seed}] {cond_name} FAILED — see {runner_log}")

    # ================================================================
    # STEP 2: Aggregate phase trajectories
    # ================================================================
    log("Aggregating phase trajectories across seeds ...")

    agg_dir = os.path.join(opt_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    for model_name in models:
        mdir_agg = os.path.join(agg_dir, model_name)
        os.makedirs(mdir_agg, exist_ok=True)

        for cond_name, cond_value in cond_ablation_grid:
            agg = aggregate_trajectories_for_condition(seed_dirs, model_name, cond_name, cond_value)
            if not agg:
                log(f"  [agg] {model_name}/{cond_name}_{cond_value}: no data found across seeds")
                continue

            n_seeds = agg.get("n_seeds", 0)
            log(f"  [agg] {model_name}/{cond_name}_{cond_value}: {n_seeds} seeds, {len(agg['epoch'])} epochs")

            cdir = os.path.join(mdir_agg, _condition_dir_name(cond_name, cond_value))
            os.makedirs(cdir, exist_ok=True)

            # Save full aggregated CSV (with _mean and _se columns)
            save_aggregated_trajectory(agg, os.path.join(cdir, "phase_trajectory_aggregated.csv"))

            # Save a standard phase_trajectory.csv (means only) for plotting scripts
            create_aggregated_phase_trajectory(agg, os.path.join(cdir, "phase_trajectory.csv"))

    # ================================================================
    # STEP 2b: Aggregate learning curves across seeds
    # ================================================================
    log("Aggregating learning curves across seeds ...")
    for model_name in models:
        mdir_agg = os.path.join(agg_dir, model_name)
        os.makedirs(mdir_agg, exist_ok=True)

        for cond_name, cond_value in cond_ablation_grid:
            lc_agg = aggregate_learning_curves_for_condition(seed_dirs, model_name, cond_name, cond_value)
            if not lc_agg:
                log(f"  [lc-agg] {model_name}/{cond_name}_{cond_value}: no learning curve data found")
                continue
            n_s = lc_agg.get("n_seeds", 0)
            log(f"  [lc-agg] {model_name}/{cond_name}_{cond_value}: {n_s} seeds, {len(lc_agg['epoch'])} epochs")
            cdir = os.path.join(mdir_agg, _condition_dir_name(cond_name, cond_value))
            os.makedirs(cdir, exist_ok=True)
            save_aggregated_learning_curve(lc_agg, os.path.join(cdir, "learning_curve_aggregated.csv"))

    # ================================================================
    # STEP 2c: Copy raw checkpoint artifacts + aggregate final-only outputs
    # ================================================================
    # Raw checkpoint directories are copied from one exemplar seed for audit plots.
    # Final envelope / phase artifacts are aggregated across seeds below.
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

        for model_name in models:
            mdir_agg = os.path.join(agg_dir, model_name)

            for cond_name, cond_value in cond_ablation_grid:
                src_model_cond_dir = os.path.join(last_seed_dir, model_name, _condition_dir_name(cond_name, cond_value))
                if not os.path.isdir(src_model_cond_dir):
                    continue

                cdir_agg = os.path.join(mdir_agg, _condition_dir_name(cond_name, cond_value))
                os.makedirs(cdir_agg, exist_ok=True)

                # Copy checkpoint directories (remove stale first)
                for ckpt_name in CHECKPOINT_DIRS:
                    src = os.path.join(src_model_cond_dir, ckpt_name)
                    dst = os.path.join(cdir_agg, ckpt_name)
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)

                aggregate_final_artifacts_for_condition(
                    seed_dirs,
                    model_name,
                    cond_name,
                    cond_value,
                    cdir_agg,
                )

    # ================================================================
    # STEP 2d: Summary table (final-epoch diagnostics per condition)
    # ================================================================
    log("Generating summary table (final-epoch diagnostics) ...")
    summary_cols = ["model", "condition", "warmup_epochs", "n_seeds",
                    "beta_hat_mean", "beta_hat_se",
                    "beta_median_mean", "beta_median_se",
                    "p_beta_lt1_mean", "p_beta_lt1_se",
                    "beta_env_mean", "beta_env_se",
                    "alpha_hat_mean", "alpha_hat_se",
                    "tau_mean_mean", "tau_mean_se"]
    summary_rows = []
    for model_name in models:
        for cond_name, cond_value in cond_ablation_grid:
            agg = aggregate_trajectories_for_condition(seed_dirs, model_name, cond_name, cond_value)
            if not agg or "epoch" not in agg or len(agg["epoch"]) == 0:
                continue
            # Use the last epoch
            n_s = agg.get("n_seeds", 0)
            cond_label = f"{cond_name}_{cond_value}"
            row = [model_name, cond_label, args.warmup_epochs, n_s]
            for col in ["beta_hat", "beta_median", "p_beta_lt1", "beta_env", "alpha_hat", "tau_mean"]:
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
    log(f"  {'Model':<12} {'Condition':<30} {'warmup':>6} {'n':>4}  {'beta_hat':>8}  {'b_med':>8}  {'P(b<1)':>8}  {'beta_env':>8}  {'alpha_hat':>8}  {'tau_mean':>10}")
    log(f"  {'-'*12} {'-'*30} {'-'*6} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in summary_rows:
        model_name, cond_label, wu, n_s = r[0], r[1], r[2], r[3]
        bh_m, bh_se = r[4], r[5]
        bmed_m, bmed_se = r[6], r[7]
        pbl_m, pbl_se = r[8], r[9]
        be_m, be_se = r[10], r[11]
        ah_m, ah_se = r[12], r[13]
        tm_m, tm_se = r[14], r[15]
        log(f"  {model_name:<12} {cond_label:<30} {wu:>6} {n_s:>4}  {bh_m:>5.2f}+/-{bh_se:.2f}  {bmed_m:>5.2f}+/-{bmed_se:.2f}  {pbl_m:>5.2f}+/-{pbl_se:.2f}  {be_m:>5.2f}+/-{be_se:.2f}  {ah_m:>5.2f}+/-{ah_se:.2f}  {tm_m:>7.1f}+/-{tm_se:.1f}")

    # ================================================================
    # STEP 2e: Threshold localization in intervention space
    # ================================================================
    log("Localizing threshold brackets in intervention space ...")

    # Group conditions by ablation type
    ablation_types = {}
    for cond_name, cond_value in cond_ablation_grid:
        if cond_name == "baseline":
            continue
        ablation_types.setdefault(cond_name, []).append(cond_value)

    # For batch_ablation, higher value = stronger suppression
    # For clip_ablation, lower value = stronger suppression
    # For winsorize_ablation, lower value = stronger suppression
    higher_means_stronger_map = {
        "batch_ablation": True,
        "clip_ablation": False,
        "winsorize_ablation": False,
    }

    threshold_brackets = {}
    for model_name in models:
        threshold_brackets[model_name] = {}
        for abl_type, abl_values in ablation_types.items():
            # Collect final phase labels per ablation value
            condition_results = []
            for val in abl_values:
                # Read final phase from aggregated or seed-level results
                cond_dir_name = _condition_dir_name(abl_type, val)
                final_phase_path = os.path.join(
                    agg_dir, model_name, cond_dir_name,
                    f"{model_name}_final_phase.json"
                )
                if os.path.isfile(final_phase_path):
                    with open(final_phase_path) as f:
                        fp = json.load(f)
                    phase_label = fp.get("phase_label", "")
                    majority_phase_label = fp.get("majority_phase_label", "")
                    phase_majority_fraction = fp.get("phase_majority_fraction")
                else:
                    # Fall back to last checkpoint label from aggregated trajectory
                    agg = aggregate_trajectories_for_condition(
                        seed_dirs, model_name, abl_type, val
                    )
                    if agg and "phase_label" in agg and len(agg["phase_label"]) > 0:
                        phase_label = agg["phase_label"][-1]
                    else:
                        phase_label = ""
                    majority_phase_label = ""
                    phase_majority_fraction = None
                condition_results.append({
                    "condition_value": float(val),
                    "phase_label": phase_label,
                    "majority_phase_label": majority_phase_label,
                    "phase_majority_fraction": phase_majority_fraction,
                })

            bracket = localize_threshold_bracket(
                condition_results,
                intervention_key="condition_value",
                phase_key="phase_label",
                higher_means_stronger=higher_means_stronger_map.get(abl_type, True),
            )
            threshold_brackets[model_name][abl_type] = bracket

            if bracket["bracket_found"]:
                majority_note = " [strict majority]" if bracket.get("used_majority_vote") else ""
                log(f"  {model_name}/{abl_type}: bracket found{majority_note} — "
                    f"last AC at {bracket['last_anti_collapsed']}, "
                    f"first collapsed at {bracket['first_collapsed']}, "
                    f"width={bracket['bracket_width']:.4g}")
            else:
                majority_note = " [strict majority]" if bracket.get("used_majority_vote") else ""
                log(f"  {model_name}/{abl_type}: no full bracket{majority_note} — "
                    f"status={bracket.get('status', 'unknown')}")

    # Save threshold brackets
    brackets_path = os.path.join(agg_dir, "threshold_brackets.json")
    with open(brackets_path, "w") as f:
        json.dump(threshold_brackets, f, indent=2, default=str)
    log(f"  Threshold brackets saved to {brackets_path}")

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
