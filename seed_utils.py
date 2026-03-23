#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for multi-seed experiment pipeline
=====================================================

Common functions for CSV I/O, file discovery, and cross-seed aggregation.
Used by main_exp1.py, plot scripts, and simulation scripts.
"""

import os, csv, re
from typing import List, Dict, Optional

import numpy as np

# ============================================================
# CSV I/O
# ============================================================

def write_csv(path: str, header: List[str], rows: List[List]):
    """Write CSV with auto-mkdir for parent directory."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def append_csv_row(path: str, row: List):
    """Append a single row to an existing CSV."""
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def safe_read_csv_named(path: str):
    """Load CSV with named columns via np.genfromtxt. Returns None on failure."""
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if data.size == 0:
            return None
        if data.shape == ():
            data = np.array([data])
        return data
    except Exception:
        return None


# ============================================================
# Phase trajectory loading
# ============================================================

# Canonical column order — dual-method alpha (ECF + McCulloch) + reliability + envelope-beta
PHASE_TRAJ_COLS = [
    "epoch",
    "step",
    "alpha_hat", "alpha_ecf", "alpha_mcculloch",
    "sigma_alpha_hat", "alpha_hat_std", "alpha_hat_se", "alpha_agreement",
    "beta_hat", "beta_r2",
    "tau_mean", "tau_q90", "tau_q99",
    "tau_fit_r2_mean", "tau_fit_n_valid",
    "fit_lags_min", "fit_lags_max",
    "alpha_reliable", "alpha_method", "n_samples",
    "beta_env", "beta_env_r2",
]

# Numeric columns (everything except index columns and string columns)
NUMERIC_COLS = [c for c in PHASE_TRAJ_COLS if c not in ("epoch", "step", "alpha_method")]

# String columns that should not be aggregated with mean/SE
STRING_COLS = ["alpha_method"]


def load_phase_trajectory_csv(path: str) -> Optional[Dict]:
    """
    Load phase_trajectory.csv into a dict of arrays.

    Returns dict with arrays for each column in PHASE_TRAJ_COLS (sorted by epoch).
    Missing columns filled with NaN (numeric) or "" (string).
    Returns None if file not found or unreadable.
    """
    data = safe_read_csv_named(path)
    if data is None:
        return None
    names = data.dtype.names or ()
    if "epoch" not in names:
        return None

    out = {}
    for c in PHASE_TRAJ_COLS:
        if c in names:
            if c in STRING_COLS:
                out[c] = np.array(data[c], dtype=str)
            else:
                out[c] = np.array(data[c], dtype=float)
        else:
            if c in STRING_COLS:
                out[c] = np.full(data.shape[0], "", dtype=object)
            else:
                out[c] = np.full(data.shape[0], np.nan, dtype=float)

    step_arr = np.asarray(out["step"], dtype=float)
    if np.any(np.isfinite(step_arr)):
        sort_key = np.where(np.isfinite(step_arr), step_arr, np.inf)
    else:
        sort_key = np.asarray(out["epoch"], dtype=float)
    order = np.argsort(sort_key)
    for c in PHASE_TRAJ_COLS:
        out[c] = out[c][order]
    out["epoch"] = out["epoch"].astype(int) if out["epoch"].dtype.kind == 'f' else out["epoch"]
    if np.all(np.isfinite(out["step"])):
        out["step"] = out["step"].astype(int)

    return out


def load_learning_curve_csv(path: str) -> Optional[Dict]:
    """Load a learning curve CSV into dict with epoch, optional step, and train_loss arrays."""
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if data.size == 0:
            return None
        if data.shape == ():
            data = np.array([data])
        names = data.dtype.names or ()
        if "epoch" not in names or "train_loss" not in names:
            return None
        epoch = np.array(data["epoch"], dtype=int)
        step = np.array(data["step"], dtype=float) if "step" in names else np.full(epoch.shape, np.nan, dtype=float)
        loss = np.array(data["train_loss"], dtype=float)
        mask = np.isfinite(epoch) & np.isfinite(loss)
        epoch, step, loss = epoch[mask], step[mask], loss[mask]
        if np.any(np.isfinite(step)):
            order = np.argsort(np.where(np.isfinite(step), step, np.inf))
        else:
            order = np.argsort(epoch)
        result = {"epoch": epoch[order], "train_loss": loss[order]}
        if np.any(np.isfinite(step)):
            step_sorted = step[order]
            result["step"] = step_sorted.astype(int) if np.all(np.isfinite(step_sorted)) else step_sorted
        return result
    except Exception:
        return None


# ============================================================
# File discovery in seed directories
# ============================================================

_SEED_RE = re.compile(r"^seed_\d+$")


def find_file_in_seed_dir(
    seed_dir: str,
    model_name: str,
    filename: str,
    search_subdirs: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Find a file in a seed directory by searching standard subdirectories.

    Searches:
      seed_dir/baselines/<model_name>/<filename>
      seed_dir/lstmgru/<model_name>/<filename>
      (or custom search_subdirs if provided)

    Returns first match or None.
    """
    if search_subdirs is None:
        # "" = direct layout (seed_dir/<model>/), used by unified runners
        # "baselines"/"lstmgru" = legacy layout from old split runners
        search_subdirs = ["", "baselines", "lstmgru"]

    for sub in search_subdirs:
        if sub:
            p = os.path.join(seed_dir, sub, model_name, filename)
        else:
            p = os.path.join(seed_dir, model_name, filename)
        if os.path.exists(p):
            return p
    return None


def find_model_dir_in_seed(
    seed_dir: str,
    model_name: str,
    search_subdirs: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Find the model output directory within a seed directory.

    Returns the first existing directory path, or None.
    """
    if search_subdirs is None:
        # "" = direct layout (seed_dir/<model>/), used by unified runners
        # "baselines"/"lstmgru" = legacy layout from old split runners
        search_subdirs = ["", "baselines", "lstmgru"]

    for sub in search_subdirs:
        if sub:
            d = os.path.join(seed_dir, sub, model_name)
        else:
            d = os.path.join(seed_dir, model_name)
        if os.path.isdir(d):
            return d
    return None


# ============================================================
# Cross-seed aggregation
# ============================================================

def aggregate_trajectories(
    seed_dirs: List[str],
    model_name: str,
) -> Dict:
    """
    Load phase_trajectory.csv from each seed directory for a given model,
    align by epoch, compute mean ± stderr across seeds.

    Returns dict with arrays: epoch, <col>_mean, <col>_se for each numeric col,
    plus n_seeds. Returns empty dict if no data found.
    """
    all_trajs = []
    for sd in seed_dirs:
        p = find_file_in_seed_dir(sd, model_name, "phase_trajectory.csv")
        if p is not None:
            tr = load_phase_trajectory_csv(p)
            if tr is not None:
                all_trajs.append(tr)

    if not all_trajs:
        return {}

    # find common epoch grid (intersection of all seeds)
    epoch_sets = [set(tr["epoch"].tolist()) for tr in all_trajs]
    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        # fallback: use union, with NaN for missing
        common_epochs = sorted(set.union(*epoch_sets))

    epochs = np.array(common_epochs, dtype=int)
    n_seeds = len(all_trajs)
    n_epochs = len(epochs)

    result = {"epoch": epochs, "n_seeds": n_seeds}

    for col in NUMERIC_COLS:
        mat = np.full((n_seeds, n_epochs), np.nan, dtype=float)
        for si, tr in enumerate(all_trajs):
            ep_to_idx = {int(e): i for i, e in enumerate(epochs)}
            for j, e in enumerate(tr["epoch"]):
                if int(e) in ep_to_idx:
                    mat[si, ep_to_idx[int(e)]] = tr[col][j]

        col_mean = np.nanmean(mat, axis=0)
        with np.errstate(invalid="ignore"):
            col_std = np.nanstd(mat, axis=0, ddof=1)
        n_valid = np.sum(np.isfinite(mat), axis=0)
        col_se = col_std / np.sqrt(np.maximum(n_valid, 1))

        result[f"{col}_mean"] = col_mean
        result[f"{col}_se"] = col_se

    # String columns: take from first seed (not aggregatable)
    for col in STRING_COLS:
        if all_trajs and col in all_trajs[0]:
            # Use first seed's values aligned to common epochs
            first = all_trajs[0]
            ep_to_idx_first = {int(e): j for j, e in enumerate(first["epoch"])}
            vals = []
            for e in epochs:
                if int(e) in ep_to_idx_first:
                    vals.append(str(first[col][ep_to_idx_first[int(e)]]))
                else:
                    vals.append("")
            result[f"{col}"] = vals

    # Step is metadata for plotting / cross-condition comparison, not an aggregate.
    if all_trajs and "step" in all_trajs[0]:
        first = all_trajs[0]
        ep_to_idx_first = {int(e): j for j, e in enumerate(first["epoch"])}
        steps = []
        for e in epochs:
            if int(e) in ep_to_idx_first:
                step_val = first["step"][ep_to_idx_first[int(e)]]
                steps.append(float(step_val) if np.isfinite(step_val) else np.nan)
            else:
                steps.append(np.nan)
        steps = np.array(steps, dtype=float)
        result["step"] = steps.astype(int) if np.all(np.isfinite(steps)) else steps

    return result


def aggregate_learning_curves(
    seed_dirs: List[str],
    model_name: str,
) -> Dict:
    """
    Load <model>_learning_curve.csv from each seed, align by epoch,
    compute mean ± SE across seeds.
    """
    all_lcs = []
    for sd in seed_dirs:
        p = find_file_in_seed_dir(sd, model_name, f"{model_name}_learning_curve.csv")
        if p is not None:
            lc = load_learning_curve_csv(p)
            if lc is not None:
                all_lcs.append(lc)

    if not all_lcs:
        return {}

    epoch_sets = [set(lc["epoch"].tolist()) for lc in all_lcs]
    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        common_epochs = sorted(set.union(*epoch_sets))

    epochs = np.array(common_epochs, dtype=int)
    n_seeds = len(all_lcs)
    n_epochs = len(epochs)

    mat = np.full((n_seeds, n_epochs), np.nan, dtype=float)
    for si, lc in enumerate(all_lcs):
        ep_to_idx = {int(e): i for i, e in enumerate(epochs)}
        for j, e in enumerate(lc["epoch"]):
            if int(e) in ep_to_idx:
                mat[si, ep_to_idx[int(e)]] = lc["train_loss"][j]

    loss_mean = np.nanmean(mat, axis=0)
    with np.errstate(invalid="ignore"):
        loss_std = np.nanstd(mat, axis=0, ddof=1)
    n_valid = np.sum(np.isfinite(mat), axis=0)
    loss_se = loss_std / np.sqrt(np.maximum(n_valid, 1))

    result = {"epoch": epochs, "train_loss_mean": loss_mean, "train_loss_se": loss_se, "n_seeds": n_seeds}

    if all_lcs and "step" in all_lcs[0]:
        first = all_lcs[0]
        ep_to_idx_first = {int(e): j for j, e in enumerate(first["epoch"])}
        steps = []
        for e in epochs:
            if int(e) in ep_to_idx_first:
                step_val = first["step"][ep_to_idx_first[int(e)]]
                steps.append(float(step_val) if np.isfinite(step_val) else np.nan)
            else:
                steps.append(np.nan)
        steps = np.array(steps, dtype=float)
        result["step"] = steps.astype(int) if np.all(np.isfinite(steps)) else steps

    return result
