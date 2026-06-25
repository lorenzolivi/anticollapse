#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for multi-seed experiment pipeline
=====================================================

Common functions for CSV I/O, file discovery, and cross-seed aggregation.
Used by main_phase_trajectory.py, plot scripts, and simulation scripts.
"""

import os, csv, re, json
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

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


def safe_load_json(path: str):
    """Load JSON file, returning None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, obj: Dict):
    """Write JSON with auto-mkdir for parent directory."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return np.isfinite(v)


def _nanmean_no_warn(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise mean and stderr without all-NaN / ddof warnings.
    """
    mat = np.asarray(mat, dtype=float)
    n_valid = np.sum(np.isfinite(mat), axis=0)
    sums = np.nansum(mat, axis=0)

    mean = np.full(mat.shape[1], np.nan, dtype=float)
    valid_mean = n_valid > 0
    mean[valid_mean] = sums[valid_mean] / n_valid[valid_mean]

    se = np.full(mat.shape[1], np.nan, dtype=float)
    one_sample = n_valid == 1
    se[one_sample] = 0.0
    many_samples = n_valid > 1
    if np.any(many_samples):
        with np.errstate(invalid="ignore"):
            std = np.nanstd(mat[:, many_samples], axis=0, ddof=1)
        se[many_samples] = std / np.sqrt(n_valid[many_samples])
    return mean, se


def _consensus_label(labels: List[str], mixed_label: str = "mixed") -> Dict[str, Any]:
    """
    Summarize a list of categorical labels without over-calling consensus.
    """
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        return {
            "label": "",
            "majority_label": "",
            "counts": {},
            "majority_fraction": float("nan"),
            "n_labeled": 0,
        }
    counts = Counter(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    unanimous = len(counts) == 1
    return {
        "label": majority_label if unanimous else mixed_label,
        "majority_label": majority_label,
        "counts": dict(counts),
        "majority_fraction": float(majority_count / len(labels)),
        "n_labeled": len(labels),
    }


def aggregate_numeric_csvs(paths: List[str], key_cols: List[str]) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """
    Aggregate numeric CSVs sharing the same schema by averaging value columns.
    """
    datasets = []
    for path in paths:
        data = safe_read_csv_named(path)
        if data is not None:
            datasets.append(data)
    if not datasets:
        return None

    names = list(datasets[0].dtype.names or [])
    if not names:
        return None

    key_cols = [c for c in key_cols if c in names]
    if not key_cols:
        key_cols = [names[0]]
    value_cols = [c for c in names if c not in key_cols]

    def _key_value(x):
        try:
            v = float(x)
            if np.isfinite(v) and abs(v - round(v)) < 1e-12:
                return int(round(v))
            return v
        except Exception:
            return str(x)

    row_maps = []
    key_tuples = set()
    for data in datasets:
        row_map = {}
        for row in data:
            key = tuple(_key_value(row[c]) for c in key_cols)
            row_map[key] = row
            key_tuples.add(key)
        row_maps.append(row_map)

    def _sort_key(key_tuple):
        out = []
        for x in key_tuple:
            if isinstance(x, (int, float)):
                out.append((0, float(x)))
            else:
                out.append((1, str(x)))
        return tuple(out)

    rows = []
    for key in sorted(key_tuples, key=_sort_key):
        row_out: List[Any] = list(key)
        for col in value_cols:
            vals = []
            for row_map in row_maps:
                row = row_map.get(key)
                if row is None:
                    continue
                try:
                    v = float(row[col])
                except Exception:
                    continue
                if np.isfinite(v):
                    vals.append(v)
            row_out.append(float(np.mean(vals)) if vals else float("nan"))
        rows.append(row_out)

    return key_cols + value_cols, rows


def _aggregate_numeric_fields(records: List[Dict], fields: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in fields:
        vals = [float(r[field]) for r in records if field in r and _is_finite_number(r[field])]
        out[field] = float(np.mean(vals)) if vals else None
    return out


def _aggregate_crossover_records(cd_records: List[Dict]) -> Optional[Dict[str, Any]]:
    """Aggregate a list of per-seed crossover_diagnostic dicts.

    Delegates the consensus of the categorical `mode` to `_consensus_label`,
    then averages numeric fields only across seeds sharing the majority mode
    (so anti-collapsed and collapsed statistics are never mixed).  Pass flags
    report both the majority boolean and the fraction of passing seeds.
    Returns None if no valid records were passed.
    """
    if not cd_records:
        return None

    mode_info = _consensus_label(
        [r.get("mode", "") for r in cd_records],
        mixed_label="mixed",
    )
    majority_mode = mode_info["majority_label"]
    cd_majority = [r for r in cd_records if r.get("mode") == majority_mode]

    out: Dict[str, Any] = {
        "valid": True,
        "n_seeds": len(cd_records),
        "mode": mode_info["label"],
        "majority_mode": majority_mode,
        "mode_counts": mode_info["counts"],
    }

    if majority_mode == "anti_collapsed":
        numeric_fields = [
            "ell_star", "n_below", "n_above",
            "power_window_beta", "power_window_beta_min",
            "power_window_local_beta", "power_window_local_r2",
            "power_window_n_points", "power_window_min_required_points",
            "runs_power_window_n_runs", "runs_power_window_z", "runs_power_window_p",
            "runs_below_n_runs", "runs_below_z", "runs_below_p",
            "sign_above_n_neg", "sign_above_n_total", "sign_above_p",
        ]
        out.update(_aggregate_numeric_fields(cd_majority, numeric_fields))
        for flag in (
            "power_law_window_pass",
            "power_window_beta_pass",
            "power_window_size_pass",
            "runs_power_window_pass",
            "runs_below_pass",
            "sign_above_pass",
        ):
            flags = [bool(r.get(flag)) for r in cd_majority if flag in r]
            out[flag + "_fraction"] = (
                float(sum(flags)) / len(flags) if flags else None
            )
            out[flag] = bool(sum(flags) > len(flags) / 2) if flags else False
    elif majority_mode == "collapsed":
        numeric_fields = [
            "r2_exp", "runs_exp_n_runs", "runs_exp_z", "runs_exp_p",
            "power_window_beta", "power_window_beta_min",
            "power_window_local_beta", "power_window_local_r2",
            "power_window_n_points", "power_window_min_required_points",
            "runs_power_window_n_runs", "runs_power_window_z", "runs_power_window_p",
        ]
        out.update(_aggregate_numeric_fields(cd_majority, numeric_fields))
        for flag in (
            "power_law_window_pass",
            "power_window_beta_pass",
            "power_window_size_pass",
            "runs_power_window_pass",
            "runs_exp_pass",
        ):
            flags = [bool(r.get(flag)) for r in cd_majority if flag in r]
            out[flag + "_fraction"] = (
                float(sum(flags)) / len(flags) if flags else None
            )
            out[flag] = bool(sum(flags) > len(flags) / 2) if flags else False

    return out


def _aggregate_envelope_fit_records(records: List[Dict]) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    out: Dict[str, Any] = {"n_seeds": len(records)}
    for key, fields in (
        ("exp", ["a", "b", "r2", "tau_env"]),
        ("power", ["c", "d", "r2"]),
        ("tempered", ["a", "d_log", "b_ell", "beta_env", "tau_max", "r2", "ell_star"]),
    ):
        sub_records = [r.get(key, {}) for r in records if isinstance(r.get(key), dict)]
        if sub_records:
            out[key] = _aggregate_numeric_fields(sub_records, fields)

    aic_records = [r.get("aic", {}) for r in records if isinstance(r.get("aic"), dict)]
    if aic_records:
        out["aic"] = _aggregate_numeric_fields(aic_records, ["exponential", "power", "tempered"])

    winner_info = _consensus_label(
        [r.get("envelope_winner", "") for r in records],
        mixed_label="mixed",
    )
    out["envelope_winner"] = winner_info["label"]
    out["majority_envelope_winner"] = winner_info["majority_label"]
    out["envelope_winner_counts"] = winner_info["counts"]
    out["envelope_winner_majority_fraction"] = winner_info["majority_fraction"]

    # --- Crossover residual diagnostic (Section 7) ---
    cd_records = [
        r.get("crossover_diagnostic", {}) for r in records
        if isinstance(r.get("crossover_diagnostic"), dict)
        and r.get("crossover_diagnostic", {}).get("valid")
    ]
    cd_agg = _aggregate_crossover_records(cd_records)
    if cd_agg is not None:
        out["crossover_diagnostic"] = cd_agg

    return out


def aggregate_envelope_fit_jsons(paths: List[str]) -> Optional[Dict[str, Any]]:
    records = [safe_load_json(p) for p in paths]
    records = [r for r in records if isinstance(r, dict)]
    return _aggregate_envelope_fit_records(records)


def aggregate_gelr_fit_jsons(paths: List[str]) -> Optional[Dict[str, Any]]:
    records = [safe_load_json(p) for p in paths]
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return None

    out: Dict[str, Any] = {"n_seeds": len(records)}
    for field in ["lambda_rowmean_min", "lambda_rowmean_max", "lambda_rowmean_mean"]:
        vals = [float(r[field]) for r in records if field in r and _is_finite_number(r[field])]
        out[field] = float(np.mean(vals)) if vals else None

    mode_info = _consensus_label([r.get("gelr_mode", "") for r in records], mixed_label="mixed")
    out["gelr_mode"] = mode_info["label"]
    out["majority_gelr_mode"] = mode_info["majority_label"]

    bool_vals = [bool(r.get("used_lag_dependent_rates")) for r in records if "used_lag_dependent_rates" in r]
    if bool_vals:
        out["used_lag_dependent_rates"] = (sum(bool_vals) > (len(bool_vals) / 2))

    recurrent = []
    for r in records:
        mats = r.get("recurrent_matrices", [])
        if isinstance(mats, list):
            recurrent.extend(str(x) for x in mats)
    out["recurrent_matrices"] = sorted(set(recurrent))

    tf_records = [r.get("transport_fit", {}) for r in records if isinstance(r.get("transport_fit"), dict)]
    gf_records = [r.get("gelr_fit", {}) for r in records if isinstance(r.get("gelr_fit"), dict)]
    if tf_records:
        out["transport_fit"] = _aggregate_envelope_fit_records(tf_records)
    if gf_records:
        out["gelr_fit"] = _aggregate_envelope_fit_records(gf_records)
    return out


def aggregate_final_phase_jsons(paths: List[str]) -> Optional[Dict[str, Any]]:
    records = [safe_load_json(p) for p in paths]
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return None

    out: Dict[str, Any] = {"aggregation_mode": "seed_summary", "n_seeds": len(records)}
    out.update(_aggregate_numeric_fields(
        records,
        [
            "tail_beta_hat", "tail_beta_r2",
            "boot_beta_median", "boot_beta_lo", "boot_beta_hi",
            "boot_p_beta_lt1", "boot_B_effective",
            "zeta_q10", "zeta_q90", "delta_zeta",
            "ell_star", "phase_r2_threshold",
            "power_window_beta_min", "power_window_min_points",
            "power_window_min_fraction",
        ],
    ))

    aic_records = [r.get("aic", {}) for r in records if isinstance(r.get("aic"), dict)]
    if aic_records:
        out["aic"] = _aggregate_numeric_fields(aic_records, ["exponential", "power", "tempered"])

    phase_info = _consensus_label([r.get("phase_label", "") for r in records], mixed_label="mixed")
    out["phase_label"] = phase_info["label"]
    out["majority_phase_label"] = phase_info["majority_label"]
    out["phase_counts"] = phase_info["counts"]
    out["phase_majority_fraction"] = phase_info["majority_fraction"]

    winner_info = _consensus_label([r.get("envelope_winner", "") for r in records], mixed_label="mixed")
    out["envelope_winner"] = winner_info["label"]
    out["majority_envelope_winner"] = winner_info["majority_label"]
    out["envelope_winner_counts"] = winner_info["counts"]

    # Crossover residual diagnostic — reuse the envelope-fit aggregator logic
    # by delegating to _aggregate_crossover_records.  This keeps one code path
    # for the diagnostic summary regardless of which JSON it travels in.
    cd_records = [
        r.get("crossover_diagnostic", {}) for r in records
        if isinstance(r.get("crossover_diagnostic"), dict)
        and r.get("crossover_diagnostic", {}).get("valid")
    ]
    cd_agg = _aggregate_crossover_records(cd_records)
    if cd_agg is not None:
        out["crossover_diagnostic"] = cd_agg
    return out


# ============================================================
# Phase trajectory loading
# ============================================================

# Canonical column order — dual-method alpha (ECF + McCulloch) + reliability +
# envelope-beta + bootstrap β̂ stability interval + checkpoint phase label
PHASE_TRAJ_COLS = [
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

# Numeric columns (everything except index columns and string columns)
NUMERIC_COLS = [c for c in PHASE_TRAJ_COLS
                if c not in ("epoch", "step", "alpha_method", "phase_label")]

# String columns that should not be aggregated with mean/SE
STRING_COLS = ["alpha_method", "phase_label"]


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

        col_mean, col_se = _nanmean_no_warn(mat)

        result[f"{col}_mean"] = col_mean
        result[f"{col}_se"] = col_se

    # String columns: keep only unanimous cross-seed labels; otherwise mark mixed.
    for col in STRING_COLS:
        vals = []
        for e in epochs:
            labels = []
            for tr in all_trajs:
                ep_arr = np.asarray(tr.get("epoch", []), dtype=int)
                matches = np.where(ep_arr == int(e))[0]
                if matches.size == 0:
                    continue
                value = str(tr[col][matches[0]]).strip()
                if value:
                    labels.append(value)
            vals.append(_consensus_label(labels)["label"])
        result[col] = vals

    # Step is plotting metadata; aggregate it across seeds like other numeric columns.
    if any("step" in tr for tr in all_trajs):
        step_mat = np.full((n_seeds, n_epochs), np.nan, dtype=float)
        for si, tr in enumerate(all_trajs):
            if "step" not in tr:
                continue
            ep_to_idx = {int(e): i for i, e in enumerate(epochs)}
            for j, e in enumerate(tr["epoch"]):
                if int(e) in ep_to_idx:
                    step_val = tr["step"][j]
                    if np.isfinite(step_val):
                        step_mat[si, ep_to_idx[int(e)]] = step_val
        step_mean, _ = _nanmean_no_warn(step_mat)
        result["step"] = np.rint(step_mean).astype(int) if np.all(np.isfinite(step_mean)) else step_mean

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

    loss_mean, loss_se = _nanmean_no_warn(mat)

    result = {"epoch": epochs, "train_loss_mean": loss_mean, "train_loss_se": loss_se, "n_seeds": n_seeds}

    if any("step" in lc for lc in all_lcs):
        step_mat = np.full((n_seeds, n_epochs), np.nan, dtype=float)
        for si, lc in enumerate(all_lcs):
            if "step" not in lc:
                continue
            ep_to_idx = {int(e): i for i, e in enumerate(epochs)}
            for j, e in enumerate(lc["epoch"]):
                if int(e) in ep_to_idx:
                    step_val = lc["step"][j]
                    if np.isfinite(step_val):
                        step_mat[si, ep_to_idx[int(e)]] = step_val
        step_mean, _ = _nanmean_no_warn(step_mat)
        result["step"] = np.rint(step_mean).astype(int) if np.all(np.isfinite(step_mean)) else step_mean

    return result
