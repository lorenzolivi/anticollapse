#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc forcing diagnostic on drift-subtracted zeta increments.

This analysis reads saved checkpoint tau spectra, constructs
zeta_q(t) = -log tau_q(t), subtracts a nonparametric estimate of
E[Delta zeta | zeta], and estimates the tail of the residual increments in
the far-left zeta slice. It is intentionally a plotting/analysis-side
diagnostic: no training code or checkpoints are modified.

The diagnostic complements the update-space forcing audit:
  - update-space increments tell us what driver reached slow-relevant
    parameter rows;
  - zeta residuals are the observable closest to the SDE forcing coordinate.

Because this uses checkpoint-to-checkpoint increments, the reported tail is a
finite-Delta-t proxy rather than a native optimizer-step law.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplcache"),
)
try:
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
except OSError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_restoring_drift import (  # noqa: E402
    build_bin_edges,
    choose_late_start_index,
    discover_model_runs,
    load_checkpoint_series,
    tau_to_zeta,
    trimmed_mean,
)


def _load_root_diagnostics_module():
    """Load the root diagnostics.py despite this script living in diagnostics/."""
    path = PROJECT_ROOT / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("anticollapse_root_diagnostics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diagnostics module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT_DIAGNOSTICS = _load_root_diagnostics_module()
estimate_tail_index_calibrated = _ROOT_DIAGNOSTICS.estimate_tail_index_calibrated


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        val = float(x)
        return val if math.isfinite(val) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 1e-12:
        return scale
    q25, q75 = np.percentile(x, [25, 75])
    scale = float((q75 - q25) / 1.349) if q75 > q25 else float("nan")
    return scale if np.isfinite(scale) and scale > 1e-12 else float("nan")


def _standardize_tail_by_unit_from_reference(
    reference_residual: np.ndarray,
    reference_run_id: np.ndarray,
    reference_unit_id: np.ndarray,
    target_residual: np.ndarray,
    target_run_id: np.ndarray,
    target_unit_id: np.ndarray,
    min_unit_samples: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Standardize tail residuals using full-series per-unit robust scales.

    The scale reference deliberately uses the full late-window residual series
    across all zeta values. Estimating scale only on the far-left slice would
    both starve most units and condition on the tail event being tested.
    """
    reference_residual = np.asarray(reference_residual, dtype=np.float64)
    target_residual = np.asarray(target_residual, dtype=np.float64)
    out = np.full_like(target_residual, np.nan, dtype=np.float64)

    ref_finite = np.isfinite(reference_residual)
    global_scale = _robust_scale(reference_residual[ref_finite])
    global_med = (
        float(np.nanmedian(reference_residual[ref_finite]))
        if np.any(ref_finite) else 0.0
    )

    ref_keys = (
        np.asarray(reference_run_id, dtype=np.int64) * 10_000_000
        + np.asarray(reference_unit_id, dtype=np.int64)
    )
    target_keys = (
        np.asarray(target_run_id, dtype=np.int64) * 10_000_000
        + np.asarray(target_unit_id, dtype=np.int64)
    )
    scale_by_key: Dict[int, Tuple[float, float, int, bool]] = {}
    scales = []
    n_reference_units = 0
    n_reference_units_with_unit_scale = 0
    for key in np.unique(ref_keys[ref_finite]):
        sel = ref_finite & (ref_keys == key)
        vals = reference_residual[sel]
        n_reference_units += 1
        if vals.size >= int(min_unit_samples):
            loc = float(np.median(vals))
            scale = _robust_scale(vals)
        else:
            loc = global_med
            scale = float("nan")
        unit_scaled = bool(np.isfinite(scale) and scale > 0.0)
        if not unit_scaled:
            loc = global_med
            scale = global_scale
        else:
            n_reference_units_with_unit_scale += 1
            scales.append(scale)
        scale_by_key[int(key)] = (float(loc), float(scale), int(vals.size), unit_scaled)

    n_unit_scaled = 0
    n_global_fallback = 0
    n_missing_reference = 0
    for key in np.unique(target_keys[np.isfinite(target_residual)]):
        sel = target_keys == key
        loc, scale, _n_ref, unit_scaled = scale_by_key.get(
            int(key),
            (global_med, global_scale, 0, False),
        )
        if int(key) not in scale_by_key:
            n_missing_reference += int(np.count_nonzero(sel))
        if not (np.isfinite(scale) and scale > 0.0):
            continue
        out[sel] = (target_residual[sel] - loc) / scale
        if unit_scaled:
            n_unit_scaled += int(np.count_nonzero(sel))
        else:
            n_global_fallback += int(np.count_nonzero(sel))

    info = {
        "scale_reference": "full_late_window_residuals_all_zeta",
        "global_robust_scale": float(global_scale),
        "min_unit_samples": int(min_unit_samples),
        "n_reference_units": int(n_reference_units),
        "n_reference_units_with_unit_scale": int(n_reference_units_with_unit_scale),
        "n_unit_scaled_tail_samples": int(n_unit_scaled),
        "n_global_fallback_tail_samples": int(n_global_fallback),
        "n_missing_reference_tail_samples": int(n_missing_reference),
        "unit_scale_median": float(np.median(scales)) if scales else float("nan"),
        "unit_scale_q10": float(np.quantile(scales, 0.10)) if scales else float("nan"),
        "unit_scale_q90": float(np.quantile(scales, 0.90)) if scales else float("nan"),
    }
    return out, info


def _fit_nonparametric_drift(
    current: np.ndarray,
    delta: np.ndarray,
    n_bins: int,
    binning: str,
    trim_fraction: float,
    uniform_clip_quantile: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    current = np.asarray(current, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.isfinite(current) & np.isfinite(delta)
    edges = build_bin_edges(current[mask], n_bins=n_bins, binning=binning,
                            uniform_clip_quantile=uniform_clip_quantile)
    bin_ids = np.digitize(current, edges[1:-1], right=False)
    drift = np.full(current.shape, np.nan, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for b in range(max(0, edges.size - 1)):
        sel = mask & (bin_ids == b)
        vals = delta[sel]
        zvals = current[sel]
        if vals.size == 0:
            center = float(0.5 * (edges[b] + edges[b + 1]))
            mu = float("nan")
        else:
            center = float(np.median(zvals))
            mu = trimmed_mean(vals, trim_fraction)
            drift[sel] = mu
        rows.append({
            "bin": int(b),
            "bin_left": float(edges[b]),
            "bin_right": float(edges[b + 1]),
            "zeta_center": center,
            "count": int(vals.size),
            "drift_trimmed_mean": mu,
        })
    return drift, rows


def _apply_binned_drift(current: np.ndarray, drift_rows: List[Dict[str, float]]) -> np.ndarray:
    """Apply a binned drift fit to new current-zeta values.

    Values outside the fitted bin range are assigned to the nearest edge bin.
    This is used only for the scale-reference residuals; the tail statistic
    itself is evaluated on the primary late window used to fit the drift.
    """
    current = np.asarray(current, dtype=np.float64)
    out = np.full(current.shape, np.nan, dtype=np.float64)
    if not drift_rows:
        return out
    left = np.asarray([r["bin_left"] for r in drift_rows], dtype=np.float64)
    right = np.asarray([r["bin_right"] for r in drift_rows], dtype=np.float64)
    vals = np.asarray([r["drift_trimmed_mean"] for r in drift_rows], dtype=np.float64)
    finite_bins = np.isfinite(left) & np.isfinite(right) & np.isfinite(vals)
    if not np.any(finite_bins):
        return out
    left, right, vals = left[finite_bins], right[finite_bins], vals[finite_bins]
    edges = np.concatenate([left[:1], right])
    ids = np.digitize(current, edges[1:-1], right=False)
    ids = np.clip(ids, 0, vals.size - 1)
    mask = np.isfinite(current)
    out[mask] = vals[ids[mask]]
    return out


def _collect_transitions(args) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, object]], List[int], List[int]]:
    runs = discover_model_runs(args.input_dir, args.model, seeds=getattr(args, "seeds", ""))
    if len(runs) < 1:
        raise RuntimeError("No model runs found")

    all_current, all_delta, all_run, all_unit, all_dt = [], [], [], [], []
    scale_current, scale_delta, scale_run, scale_unit, scale_dt = [], [], [], [], []
    run_rows: List[Dict[str, object]] = []
    late_start_epochs: List[int] = []
    scale_start_epochs: List[int] = []

    for ridx, run in enumerate(runs):
        epochs, tau_matrix = load_checkpoint_series(run["model_dir"])
        zeta_matrix = tau_to_zeta(tau_matrix, tau_floor=float(args.tau_floor))
        n_checkpoints, n_units = zeta_matrix.shape
        late_start_idx = choose_late_start_index(
            n_checkpoints,
            late_fraction=float(args.late_fraction),
            min_late_checkpoints=int(args.min_late_checkpoints),
        )
        scale_start_idx = choose_late_start_index(
            n_checkpoints,
            late_fraction=float(args.scale_late_fraction),
            min_late_checkpoints=int(args.min_scale_checkpoints),
        )
        late_start_epochs.append(int(epochs[late_start_idx]))
        scale_start_epochs.append(int(epochs[scale_start_idx]))
        n_pairs = 0
        n_samples = 0
        n_scale_pairs = 0
        n_scale_samples = 0
        for i in range(scale_start_idx, n_checkpoints - 1):
            dt = float(max(1, int(epochs[i + 1] - epochs[i])))
            current = zeta_matrix[i]
            delta = (zeta_matrix[i + 1] - current) / dt
            mask = np.isfinite(current) & np.isfinite(delta)
            if not np.any(mask):
                continue
            units = np.arange(n_units, dtype=np.int64)[mask]
            scale_current.append(current[mask])
            scale_delta.append(delta[mask])
            scale_run.append(np.full(units.shape, ridx, dtype=np.int64))
            scale_unit.append(units)
            scale_dt.append(np.full(units.shape, dt, dtype=np.float64))
            n_scale_pairs += 1
            n_scale_samples += int(np.count_nonzero(mask))
        for i in range(late_start_idx, n_checkpoints - 1):
            dt = float(max(1, int(epochs[i + 1] - epochs[i])))
            current = zeta_matrix[i]
            delta = (zeta_matrix[i + 1] - current) / dt
            mask = np.isfinite(current) & np.isfinite(delta)
            if not np.any(mask):
                continue
            units = np.arange(n_units, dtype=np.int64)[mask]
            all_current.append(current[mask])
            all_delta.append(delta[mask])
            all_run.append(np.full(units.shape, ridx, dtype=np.int64))
            all_unit.append(units)
            all_dt.append(np.full(units.shape, dt, dtype=np.float64))
            n_pairs += 1
            n_samples += int(np.count_nonzero(mask))
        run_rows.append({
            "run_label": run["label"],
            "model_dir": run["model_dir"],
            "late_start_epoch": int(epochs[late_start_idx]),
            "scale_start_epoch": int(epochs[scale_start_idx]),
            "n_transition_pairs": int(n_pairs),
            "n_transition_samples": int(n_samples),
            "n_scale_reference_pairs": int(n_scale_pairs),
            "n_scale_reference_samples": int(n_scale_samples),
            "n_units": int(n_units),
        })

    if not all_current:
        raise RuntimeError("No late checkpoint transitions could be constructed")

    arrays = {
        "current": np.concatenate(all_current),
        "delta": np.concatenate(all_delta),
        "run_id": np.concatenate(all_run),
        "unit_id": np.concatenate(all_unit),
        "dt": np.concatenate(all_dt),
    }
    scale_arrays = {
        "current": np.concatenate(scale_current) if scale_current else arrays["current"],
        "delta": np.concatenate(scale_delta) if scale_delta else arrays["delta"],
        "run_id": np.concatenate(scale_run) if scale_run else arrays["run_id"],
        "unit_id": np.concatenate(scale_unit) if scale_unit else arrays["unit_id"],
        "dt": np.concatenate(scale_dt) if scale_dt else arrays["dt"],
    }
    return arrays, scale_arrays, run_rows, late_start_epochs, scale_start_epochs


def _subsample_for_plot(x: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= max_points:
        return np.sort(x)
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=int(max_points), replace=False)
    return np.sort(x[idx])


def _model_label(model: str) -> str:
    labels = {
        "shared": "SharedGate",
        "diag": "DiagGate",
        "const": "ConstGate",
        "gru": "GRU",
        "lstm": "LSTM",
    }
    return labels.get(str(model).lower(), str(model))


def _plot_logsurvival(samples: np.ndarray, outpath: Path, title: str, model_label: str) -> None:
    x = np.abs(np.asarray(samples, dtype=np.float64))
    x = x[np.isfinite(x)]
    if x.size < 10:
        return
    x = np.sort(x)
    n = x.size
    ccdf = (n - np.arange(n, dtype=np.float64)) / float(n)
    min_y = max(0.5 / float(n), 1e-5)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, ccdf, color="#2b6ab8", linewidth=1.4, label=model_label)
    grid = np.linspace(0.0, max(float(np.quantile(x, 0.995)), 1.0), 300)
    gaussian_tail = np.maximum(2.0 * stats.norm.sf(grid), min_y)
    ax.plot(grid, gaussian_tail, "--", color="#777777",
            linewidth=1.2, label="matched Gaussian tail")
    ax.set_yscale("log")
    ax.set_ylim(min_y, 1.05)
    ax.set_xlabel(r"$|r|$ (robustly standardized)")
    ax.set_ylabel(r"$\Pr(|R|\geq |r|)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_qq(samples: np.ndarray, outpath: Path, title: str, model_label: str, seed: int) -> None:
    x = _subsample_for_plot(samples, max_points=8000, seed=seed)
    if x.size < 10:
        return
    x = x - float(np.median(x))
    scale = _robust_scale(x)
    if np.isfinite(scale) and scale > 0:
        x = x / scale
    n = x.size
    probs = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / float(n)
    q_theory = stats.norm.ppf(probs)
    lo = float(np.nanmin([np.min(q_theory), np.min(x)]))
    hi = float(np.nanmax([np.max(q_theory), np.max(x)]))
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(q_theory, x, s=4, alpha=0.35, color="#2b6ab8", edgecolor="none", label=model_label)
    ax.plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=1.1, label="Gaussian reference")
    ax.set_xlabel("Gaussian quantile")
    ax.set_ylabel("Empirical residual quantile")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Model directory or experiment root containing seed_*/<model>/checkpoint_taus.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="",
                    help="Optional comma-separated seed filter, e.g. 47,83,12.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--late_fraction", type=float, default=0.35)
    ap.add_argument("--min_late_checkpoints", type=int, default=4)
    ap.add_argument("--scale_late_fraction", type=float, default=0.75,
                    help="Wider late-window fraction used only to estimate per-unit robust scales.")
    ap.add_argument("--min_scale_checkpoints", type=int, default=20,
                    help="Minimum checkpoints in the scale-reference window.")
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--binning", choices=["quantile", "uniform"], default="quantile")
    ap.add_argument("--trim_fraction", type=float, default=0.1)
    ap.add_argument("--uniform_clip_quantile", type=float, default=0.01)
    ap.add_argument("--tail_q_low", type=float, default=0.10)
    ap.add_argument("--tau_floor", type=float, default=1e-12)
    ap.add_argument("--standardize", choices=["unit", "global", "none"], default="unit")
    ap.add_argument("--min_unit_samples", type=int, default=20)
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--tail_k_frac", type=float, default=0.08)
    ap.add_argument("--tail_k_min", type=int, default=50)
    ap.add_argument("--tail_bootstrap_B", type=int, default=300)
    ap.add_argument("--tail_ci_B", type=int, default=100)
    ap.add_argument("--tail_substantive_alpha", type=float, default=1.8)
    ap.add_argument("--tail_gaussian_test_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"zeta-residual forcing diagnostic: input={args.input_dir}, model={args.model}")
    arrays, scale_arrays, run_rows, late_start_epochs, scale_start_epochs = _collect_transitions(args)
    current = arrays["current"]
    delta = arrays["delta"]
    dt_values = arrays["dt"]
    drift_hat, drift_rows = _fit_nonparametric_drift(
        current=current,
        delta=delta,
        n_bins=int(args.n_bins),
        binning=str(args.binning),
        trim_fraction=float(args.trim_fraction),
        uniform_clip_quantile=float(args.uniform_clip_quantile),
    )
    residual = delta - drift_hat
    scale_drift_hat = _apply_binned_drift(scale_arrays["current"], drift_rows)
    scale_residual = scale_arrays["delta"] - scale_drift_hat

    tail_cut = float(np.nanquantile(current[np.isfinite(current)], float(args.tail_q_low)))
    tail_mask = np.isfinite(current) & np.isfinite(residual) & (current <= tail_cut)
    residual_tail = residual[tail_mask]
    run_tail = arrays["run_id"][tail_mask]
    unit_tail = arrays["unit_id"][tail_mask]

    standardization_info: Dict[str, float] = {}
    if args.standardize == "unit":
        residual_std, standardization_info = _standardize_tail_by_unit_from_reference(
            reference_residual=scale_residual,
            reference_run_id=scale_arrays["run_id"],
            reference_unit_id=scale_arrays["unit_id"],
            target_residual=residual_tail,
            target_run_id=run_tail,
            target_unit_id=unit_tail,
            min_unit_samples=int(args.min_unit_samples),
        )
    elif args.standardize == "global":
        loc = float(np.nanmedian(residual_tail)) if residual_tail.size else 0.0
        scale = _robust_scale(residual_tail)
        residual_std = (residual_tail - loc) / scale if np.isfinite(scale) and scale > 0 else residual_tail * np.nan
        standardization_info = {"global_robust_scale": float(scale), "global_location": loc}
    else:
        residual_std = residual_tail
        standardization_info = {"global_robust_scale": 1.0}

    residual_std = residual_std[np.isfinite(residual_std)]
    if residual_std.size > int(args.max_samples):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(residual_std.size, size=int(args.max_samples), replace=False)
        residual_for_est = residual_std[idx]
    else:
        residual_for_est = residual_std

    est = estimate_tail_index_calibrated(
        residual_for_est,
        k_frac=float(args.tail_k_frac),
        k_min=int(args.tail_k_min),
        calib_B=int(args.tail_bootstrap_B),
        ci_B=int(args.tail_ci_B),
        substantive_alpha_threshold=float(args.tail_substantive_alpha),
        gaussian_test_alpha=float(args.tail_gaussian_test_alpha),
        seed=int(args.seed),
    )

    metrics = {
        "input_dir": os.path.abspath(args.input_dir),
        "model": str(args.model),
        "n_runs": int(len(run_rows)),
        "late_start_epoch_min": int(min(late_start_epochs)) if late_start_epochs else None,
        "late_start_epoch_max": int(max(late_start_epochs)) if late_start_epochs else None,
        "scale_late_fraction": float(args.scale_late_fraction),
        "scale_start_epoch_min": int(min(scale_start_epochs)) if scale_start_epochs else None,
        "scale_start_epoch_max": int(max(scale_start_epochs)) if scale_start_epochs else None,
        "n_scale_reference_samples_total": int(scale_arrays["current"].size),
        "n_transition_samples_total": int(current.size),
        "dt_unique": [float(x) for x in np.unique(dt_values[np.isfinite(dt_values)]).tolist()],
        "dt_min": float(np.nanmin(dt_values)) if dt_values.size else float("nan"),
        "dt_max": float(np.nanmax(dt_values)) if dt_values.size else float("nan"),
        "dt_is_constant": bool(
            np.unique(dt_values[np.isfinite(dt_values)]).size <= 1
            if dt_values.size else True
        ),
        "scale_dt_unique": [float(x) for x in np.unique(scale_arrays["dt"][np.isfinite(scale_arrays["dt"])]).tolist()],
        "tail_q_low": float(args.tail_q_low),
        "tail_zeta_cut": tail_cut,
        "n_tail_residual_samples": int(residual_tail.size),
        "n_tail_residual_samples_finite_standardized": int(residual_std.size),
        "n_tail_residual_samples_used": int(residual_for_est.size),
        "delta_t_note": (
            "checkpoint-to-checkpoint increments divided by epoch gap; "
            "dt distribution is recorded because nonuniform spacing can "
            "heteroscedastically distort pooled residual tails"
        ),
        "drift_estimator": "nonparametric quantile-bin trimmed mean",
        "standardization": str(args.standardize),
        "standardization_info": standardization_info,
        "tail_estimate": est,
    }

    _write_json(outdir / "zeta_residual_forcing_metrics.json", metrics)
    _write_csv(
        outdir / "zeta_residual_forcing_summary.csv",
        [
            "model", "n_runs", "n_transition_samples_total",
            "tail_q_low", "tail_zeta_cut", "n_tail_residual_samples_used",
            "alpha_eff", "alpha_eff_lo", "alpha_eff_hi", "xi_hat",
            "gaussian_p_value", "gaussian_p_value_floor",
            "gaussian_p_value_at_floor", "gaussian_reject",
            "detectably_heavy", "substantively_heavy", "reliable",
            "k_selected",
        ],
        [[
            args.model, metrics["n_runs"], metrics["n_transition_samples_total"],
            metrics["tail_q_low"], metrics["tail_zeta_cut"], metrics["n_tail_residual_samples_used"],
            est.get("alpha_eff"), est.get("alpha_eff_lo"), est.get("alpha_eff_hi"), est.get("xi_hat"),
            est.get("gaussian_p_value"), est.get("gaussian_p_value_floor"),
            est.get("gaussian_p_value_at_floor"), est.get("gaussian_reject"),
            est.get("detectably_heavy"), est.get("substantively_heavy"), est.get("reliable"),
            est.get("k_selected"),
        ]],
    )
    _write_csv(
        outdir / "zeta_residual_drift_bins.csv",
        ["bin", "bin_left", "bin_right", "zeta_center", "count", "drift_trimmed_mean"],
        [[row[k] for k in ["bin", "bin_left", "bin_right", "zeta_center", "count", "drift_trimmed_mean"]]
         for row in drift_rows],
    )
    _write_csv(
        outdir / "zeta_residual_run_summary.csv",
        [
            "run_label", "late_start_epoch", "scale_start_epoch",
            "n_transition_pairs", "n_transition_samples",
            "n_scale_reference_pairs", "n_scale_reference_samples",
            "n_units", "model_dir",
        ],
        [[
            r["run_label"], r["late_start_epoch"], r["scale_start_epoch"],
            r["n_transition_pairs"], r["n_transition_samples"],
            r["n_scale_reference_pairs"], r["n_scale_reference_samples"],
            r["n_units"], r["model_dir"],
        ]
         for r in run_rows],
    )

    title = "Drift-subtracted far-left zeta residuals"
    model_label = _model_label(args.model)
    _plot_logsurvival(
        residual_for_est,
        outdir / "zeta_residual_logsurvival.png",
        title,
        model_label,
    )
    _plot_qq(
        residual_for_est,
        outdir / "zeta_residual_qq.png",
        title,
        model_label,
        seed=int(args.seed) + 99,
    )

    a = est.get("alpha_eff", float("nan"))
    p = est.get("gaussian_p_value", float("nan"))
    log(
        f"saved {outdir}; n_tail={residual_for_est.size}, "
        f"alpha_eff={a:.4g}, gaussian_p={p:.4g}"
    )


if __name__ == "__main__":
    main()
