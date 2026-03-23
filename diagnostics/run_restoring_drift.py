#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic 3: Restoring-drift validation for the foundational assumptions
==========================================================================

Post-hoc diagnostic for the restoring-drift closure used in the stochastic
model of the log-spectrum. The script reads saved checkpoint tau spectra,
reconstructs

    zeta_q(t) = log(mu_bar_q(t)) = -log(tau_q(t)),

and evaluates two empirical signatures of confinement in the late-training
regime:

For a linearized late-training drift of the form

    F(zeta) ~= a - kappa * zeta,

a restoring drift corresponds to kappa > 0, i.e. a negative slope of
Delta zeta versus current zeta.

1. Conditional drift:
       F_hat(zeta) ~ E[Delta zeta | zeta_t ~= zeta]
   estimated by binning late-training transitions between consecutive
   checkpoints.

2. Moment stabilization:
   the population mean and variance of zeta_q(t) across neurons should level
   off in late training rather than drift indefinitely.

Supported input layouts:
  - A direct model directory containing checkpoint_taus/
  - An experiment root containing seed_*/<model>/
  - An experiment root containing <model>/ (single-seed or aggregated copy)

Outputs:
  <outdir>/
    conditional_drift.csv
    moment_trajectory.csv
    per_seed_moments.csv
    metrics.json
    conditional_drift.png
    late_moments.png
    manifest.json

Example:
  python diagnostics/run_restoring_drift.py \
      --input_dir results/exp1/adamw \
      --model gru \
      --outdir results/exp1/adamw/drift_validation_gru
"""

import argparse
import math
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diag_utils import save_csv, save_json  # noqa: E402
from seed_utils import find_model_dir_in_seed  # noqa: E402


SEED_RE = re.compile(r"^seed_\d+$")
CKPT_TAU_RE = re.compile(r"^ckpt_(\d+)_taus\.(csv|npy)$")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_model_dir(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "checkpoint_taus"))


def infer_display_name(input_dir: str, model: str, runs) -> str:
    if model:
        return model
    if len(runs) == 1:
        return os.path.basename(os.path.abspath(runs[0]["model_dir"]))
    return os.path.basename(os.path.abspath(input_dir))


def discover_model_runs(input_dir: str, model: str):
    input_dir = os.path.abspath(input_dir)

    if is_model_dir(input_dir):
        return [{
            "label": os.path.basename(input_dir.rstrip(os.sep)),
            "model_dir": input_dir,
            "source": "direct-model-dir",
        }]

    seed_dirs = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            full = os.path.join(input_dir, name)
            if os.path.isdir(full) and SEED_RE.match(name):
                seed_dirs.append(full)

    runs = []
    if seed_dirs:
        if not model:
            raise ValueError(
                "Input directory contains seed_* subdirectories; pass --model to "
                "select which model to analyze."
            )
        for seed_dir in seed_dirs:
            model_dir = find_model_dir_in_seed(seed_dir, model)
            if model_dir and is_model_dir(model_dir):
                runs.append({
                    "label": os.path.basename(seed_dir.rstrip(os.sep)),
                    "model_dir": model_dir,
                    "source": "seed-dir",
                })
        if runs:
            return runs

    if model:
        model_child = os.path.join(input_dir, model)
        if is_model_dir(model_child):
            return [{
                "label": os.path.basename(input_dir.rstrip(os.sep)),
                "model_dir": model_child,
                "source": "model-child-fallback",
            }]

    direct_model_children = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            full = os.path.join(input_dir, name)
            if os.path.isdir(full) and is_model_dir(full):
                direct_model_children.append(full)

    if len(direct_model_children) == 1:
        only = direct_model_children[0]
        return [{
            "label": os.path.basename(only.rstrip(os.sep)),
            "model_dir": only,
            "source": "single-model-child",
        }]

    raise ValueError(
        "Could not locate checkpoint_taus data. Pass either a model directory, "
        "or an experiment root plus --model."
    )


def load_tau_file(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        tau = np.load(path)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        return tau

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data is None:
        raise ValueError(f"Could not parse tau CSV: {path}")
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)

    names = data.dtype.names or ()
    if "tau" not in names:
        raise ValueError(f"CSV missing tau column: {path}")

    tau = np.asarray(data["tau"], dtype=np.float64).reshape(-1)
    if "unit_id" in names:
        unit_id = np.asarray(data["unit_id"], dtype=np.int64).reshape(-1)
        order = np.argsort(unit_id)
        tau = tau[order]
    return tau


def load_checkpoint_series(model_dir: str):
    tau_dir = os.path.join(model_dir, "checkpoint_taus")
    if not os.path.isdir(tau_dir):
        raise ValueError(f"Missing checkpoint_taus directory: {model_dir}")

    by_epoch = {}
    for name in os.listdir(tau_dir):
        match = CKPT_TAU_RE.match(name)
        if not match:
            continue
        epoch = int(match.group(1))
        ext = match.group(2)
        score = 1 if ext == "csv" else 0
        current = by_epoch.get(epoch)
        path = os.path.join(tau_dir, name)
        if current is None or score > current["score"]:
            by_epoch[epoch] = {"path": path, "score": score}

    if len(by_epoch) < 2:
        raise ValueError(f"Need at least 2 checkpoint tau files in {tau_dir}")

    epochs = sorted(by_epoch.keys())
    tau_list = []
    n_units = None
    for epoch in epochs:
        tau = load_tau_file(by_epoch[epoch]["path"])
        if n_units is None:
            n_units = int(tau.size)
        elif tau.size != n_units:
            raise ValueError(
                f"Inconsistent hidden size across checkpoints in {model_dir}: "
                f"expected {n_units}, got {tau.size} at epoch {epoch}"
            )
        tau_list.append(tau)

    tau_matrix = np.stack(tau_list, axis=0)
    return np.asarray(epochs, dtype=np.int64), tau_matrix


def tau_to_zeta(tau: np.ndarray, tau_floor: float) -> np.ndarray:
    return -np.log(np.clip(np.asarray(tau, dtype=np.float64), tau_floor, np.inf))


def choose_late_start_index(n_checkpoints: int, late_fraction: float, min_late_checkpoints: int) -> int:
    n_late = int(math.ceil(float(late_fraction) * float(n_checkpoints)))
    n_late = max(int(min_late_checkpoints), n_late)
    n_late = min(n_checkpoints, n_late)
    n_late = max(2, n_late)
    return max(0, n_checkpoints - n_late)


def trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if trim_fraction <= 0.0 or values.size < 4:
        return float(np.mean(values))
    values = np.sort(values)
    k = int(math.floor(trim_fraction * values.size))
    if 2 * k >= values.size:
        return float(np.mean(values))
    return float(np.mean(values[k:values.size - k]))


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    coeff = np.polyfit(x[mask], y[mask], deg=1)
    return float(coeff[0])


def safe_spearmanr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan")
    result = stats.spearmanr(x[mask], y[mask])
    return float(result.correlation), float(result.pvalue)


def summarize_run_transition_metrics(current: np.ndarray,
                                     delta: np.ndarray,
                                     tail_q: float):
    current = np.asarray(current, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.isfinite(current) & np.isfinite(delta)
    current = current[mask]
    delta = delta[mask]
    if current.size == 0:
        return {
            "bulk_center": float("nan"),
            "conditional_slope": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "inward_fraction": float("nan"),
            "left_cut": float("nan"),
            "right_cut": float("nan"),
            "left_delta_median": float("nan"),
            "right_delta_median": float("nan"),
            "n_transition_samples": 0,
            "left_delta": np.zeros(0, dtype=np.float64),
            "right_delta": np.zeros(0, dtype=np.float64),
        }

    bulk_center = float(np.median(current))
    rho, pval = safe_spearmanr(current, delta)
    inward = ((current - bulk_center) * delta) < 0.0
    inward_fraction = float(np.mean(inward)) if inward.size > 0 else float("nan")

    left_cut = float(np.quantile(current, tail_q))
    right_cut = float(np.quantile(current, 1.0 - tail_q))
    left_mask = current <= left_cut
    right_mask = current >= right_cut
    left_delta = delta[left_mask]
    right_delta = delta[right_mask]

    return {
        "bulk_center": bulk_center,
        "conditional_slope": fit_linear_slope(current, delta),
        "spearman_rho": rho,
        "spearman_p": pval,
        "inward_fraction": inward_fraction,
        "left_cut": left_cut,
        "right_cut": right_cut,
        "left_delta_median": float(np.median(left_delta)) if left_delta.size > 0 else float("nan"),
        "right_delta_median": float(np.median(right_delta)) if right_delta.size > 0 else float("nan"),
        "n_transition_samples": int(current.size),
        "left_delta": left_delta,
        "right_delta": right_delta,
    }


def build_bin_edges(zeta: np.ndarray, n_bins: int, binning: str, uniform_clip_quantile: float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=np.float64)
    zeta = zeta[np.isfinite(zeta)]
    if zeta.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    if binning == "quantile":
        edges = np.quantile(zeta, np.linspace(0.0, 1.0, int(n_bins) + 1))
        edges = np.unique(edges)
        if edges.size >= 2:
            return edges

    q = float(np.clip(uniform_clip_quantile, 0.0, 0.49))
    lo = float(np.quantile(zeta, q))
    hi = float(np.quantile(zeta, 1.0 - q))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.min(zeta))
        hi = float(np.max(zeta))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    edges[0] = min(edges[0], float(np.min(zeta))) - 1e-9
    edges[-1] = max(edges[-1], float(np.max(zeta))) + 1e-9
    return edges


def summarize_conditional_drift(zeta: np.ndarray,
                                delta: np.ndarray,
                                n_bins: int,
                                binning: str,
                                trim_fraction: float,
                                uniform_clip_quantile: float):
    zeta = np.asarray(zeta, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.isfinite(zeta) & np.isfinite(delta)
    zeta = zeta[mask]
    delta = delta[mask]
    if zeta.size == 0:
        return [], float("nan")

    bulk_center = float(np.median(zeta))
    edges = build_bin_edges(zeta, n_bins=n_bins, binning=binning,
                            uniform_clip_quantile=uniform_clip_quantile)
    if edges.size < 2:
        return [], bulk_center

    bin_ids = np.digitize(zeta, edges[1:-1], right=False)
    rows = []
    for bi in range(edges.size - 1):
        sel = (bin_ids == bi)
        if not np.any(sel):
            continue
        z = zeta[sel]
        d = delta[sel]
        rows.append({
            "bin_left": float(edges[bi]),
            "bin_right": float(edges[bi + 1]),
            "zeta_center": float(np.median(z)),
            "count": int(z.size),
            "delta_median": float(np.median(d)),
            "delta_trimmed_mean": trimmed_mean(d, trim_fraction),
            "delta_q25": float(np.quantile(d, 0.25)),
            "delta_q75": float(np.quantile(d, 0.75)),
            "zeta_q25": float(np.quantile(z, 0.25)),
            "zeta_q75": float(np.quantile(z, 0.75)),
        })
    return rows, bulk_center


def aggregate_moment_rows(moment_rows):
    if not moment_rows:
        return []

    epochs = sorted({int(row["epoch"]) for row in moment_rows})
    aggregated = []
    for epoch in epochs:
        subset = [row for row in moment_rows if int(row["epoch"]) == epoch]
        mean_vals = np.array([row["zeta_mean"] for row in subset], dtype=np.float64)
        var_vals = np.array([row["zeta_var"] for row in subset], dtype=np.float64)
        med_vals = np.array([row["zeta_median"] for row in subset], dtype=np.float64)
        q10_vals = np.array([row["zeta_q10"] for row in subset], dtype=np.float64)
        q90_vals = np.array([row["zeta_q90"] for row in subset], dtype=np.float64)

        def _se(values):
            n = int(np.count_nonzero(np.isfinite(values)))
            if n <= 1:
                return 0.0
            return float(np.nanstd(values, ddof=1) / math.sqrt(n))

        aggregated.append({
            "epoch": int(epoch),
            "n_runs": int(len(subset)),
            "n_late_runs": int(sum(int(row["is_late"]) for row in subset)),
            "zeta_mean_mean": float(np.nanmean(mean_vals)),
            "zeta_mean_se": _se(mean_vals),
            "zeta_var_mean": float(np.nanmean(var_vals)),
            "zeta_var_se": _se(var_vals),
            "zeta_median_mean": float(np.nanmean(med_vals)),
            "zeta_median_se": _se(med_vals),
            "zeta_q10_mean": float(np.nanmean(q10_vals)),
            "zeta_q90_mean": float(np.nanmean(q90_vals)),
        })
    return aggregated


def plot_conditional_drift(rows, bulk_center: float, display_name: str, outpath: str):
    if not rows:
        return

    x = np.array([row["zeta_center"] for row in rows], dtype=np.float64)
    y = np.array([row["delta_median"] for row in rows], dtype=np.float64)
    y_lo = np.array([row["delta_q25"] for row in rows], dtype=np.float64)
    y_hi = np.array([row["delta_q75"] for row in rows], dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_lo = y_lo[order]
    y_hi = y_hi[order]

    plt.figure(figsize=(7.2, 4.6))
    plt.fill_between(x, y_lo, y_hi, alpha=0.18, color="tab:blue", linewidth=0.0)
    plt.plot(x, y, marker="o", linewidth=1.8, color="tab:blue")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    if np.isfinite(bulk_center):
        plt.axvline(bulk_center, color="tab:red", linestyle=":", linewidth=1.2)
    plt.xlabel(r"current $\zeta$")
    plt.ylabel(r"median $\Delta \zeta / \Delta \mathrm{epoch}$")
    plt.title(f"Late-training conditional drift: {display_name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_moments(moment_agg_rows, late_start_epoch: float, display_name: str, outpath: str):
    if not moment_agg_rows:
        return

    epoch = np.array([row["epoch"] for row in moment_agg_rows], dtype=np.float64)
    mean_mean = np.array([row["zeta_mean_mean"] for row in moment_agg_rows], dtype=np.float64)
    mean_se = np.array([row["zeta_mean_se"] for row in moment_agg_rows], dtype=np.float64)
    var_mean = np.array([row["zeta_var_mean"] for row in moment_agg_rows], dtype=np.float64)
    var_se = np.array([row["zeta_var_se"] for row in moment_agg_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for ax in axes:
        if np.isfinite(late_start_epoch):
            ax.axvspan(late_start_epoch, float(np.max(epoch)), color="0.85", alpha=0.4)
        ax.grid(True, alpha=0.25)

    axes[0].plot(epoch, mean_mean, color="tab:green", linewidth=2.0)
    axes[0].fill_between(epoch, mean_mean - mean_se, mean_mean + mean_se,
                         color="tab:green", alpha=0.18, linewidth=0.0)
    axes[0].set_ylabel(r"$\mathbb{E}_q[\zeta_q(t)]$")
    axes[0].set_title(f"Late-training moment stabilization: {display_name}")

    axes[1].plot(epoch, var_mean, color="tab:purple", linewidth=2.0)
    axes[1].fill_between(epoch, var_mean - var_se, var_mean + var_se,
                         color="tab:purple", alpha=0.18, linewidth=0.0)
    axes[1].set_ylabel(r"$\mathrm{Var}_q(\zeta_q(t))$")
    axes[1].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Model directory or experiment root containing checkpoint_taus data.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to analyze when input_dir contains multiple models or seed_* dirs.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory where diagnostic CSV/JSON/PNG outputs will be written.")
    parser.add_argument("--late_fraction", type=float, default=0.3,
                        help="Fraction of checkpoints treated as late training.")
    parser.add_argument("--min_late_checkpoints", type=int, default=4,
                        help="Minimum number of late checkpoints to retain.")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="Number of bins for conditional-drift estimation.")
    parser.add_argument("--binning", type=str, default="quantile",
                        choices=["quantile", "uniform"],
                        help="Binning scheme for conditional drift.")
    parser.add_argument("--trim_fraction", type=float, default=0.1,
                        help="Trim fraction for the robust mean increment per bin.")
    parser.add_argument("--uniform_clip_quantile", type=float, default=0.005,
                        help="Only used for uniform bins; clips extreme zeta values when choosing the display range.")
    parser.add_argument("--tail_quantile", type=float, default=0.1,
                        help="Quantile used to summarize left and right tail drift signs.")
    parser.add_argument("--tau_floor", type=float, default=1e-30,
                        help="Lower bound used before taking -log(tau).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = discover_model_runs(args.input_dir, args.model)
    display_name = infer_display_name(args.input_dir, args.model, runs)

    log(f"Found {len(runs)} run(s) for {display_name}")

    manifest = {
        "input_dir": os.path.abspath(args.input_dir),
        "model": args.model,
        "display_name": display_name,
        "n_runs": int(len(runs)),
        "runs": runs,
        "late_fraction": float(args.late_fraction),
        "min_late_checkpoints": int(args.min_late_checkpoints),
        "n_bins": int(args.n_bins),
        "binning": str(args.binning),
        "trim_fraction": float(args.trim_fraction),
        "uniform_clip_quantile": float(args.uniform_clip_quantile),
        "tail_quantile": float(args.tail_quantile),
        "tau_floor": float(args.tau_floor),
    }

    tail_q = float(np.clip(args.tail_quantile, 1e-6, 0.49))
    all_current = []
    all_delta = []
    all_left_delta = []
    all_right_delta = []
    moment_rows = []
    run_summaries = []
    late_start_epochs = []
    transition_count = 0

    for run in runs:
        epochs, tau_matrix = load_checkpoint_series(run["model_dir"])
        zeta_matrix = tau_to_zeta(tau_matrix, tau_floor=args.tau_floor)
        n_checkpoints, n_units = zeta_matrix.shape
        late_start_idx = choose_late_start_index(
            n_checkpoints,
            late_fraction=args.late_fraction,
            min_late_checkpoints=args.min_late_checkpoints,
        )
        late_start_epoch = int(epochs[late_start_idx])
        late_start_epochs.append(late_start_epoch)
        run_current = []
        run_delta = []

        for i, epoch in enumerate(epochs):
            zeta = zeta_matrix[i]
            moment_rows.append({
                "run_label": run["label"],
                "epoch": int(epoch),
                "is_late": int(i >= late_start_idx),
                "zeta_mean": float(np.mean(zeta)),
                "zeta_var": float(np.var(zeta, ddof=1)) if n_units > 1 else 0.0,
                "zeta_median": float(np.median(zeta)),
                "zeta_q10": float(np.quantile(zeta, 0.10)),
                "zeta_q90": float(np.quantile(zeta, 0.90)),
            })

        for i in range(late_start_idx, n_checkpoints - 1):
            delta_epoch = float(max(1, int(epochs[i + 1] - epochs[i])))
            current = zeta_matrix[i]
            delta = (zeta_matrix[i + 1] - current) / delta_epoch
            mask = np.isfinite(current) & np.isfinite(delta)
            if np.any(mask):
                current_valid = current[mask]
                delta_valid = delta[mask]
                all_current.append(current_valid)
                all_delta.append(delta_valid)
                run_current.append(current_valid)
                run_delta.append(delta_valid)
                transition_count += 1

        if run_current:
            run_current_all = np.concatenate(run_current)
            run_delta_all = np.concatenate(run_delta)
        else:
            run_current_all = np.zeros(0, dtype=np.float64)
            run_delta_all = np.zeros(0, dtype=np.float64)

        run_transition_metrics = summarize_run_transition_metrics(
            run_current_all,
            run_delta_all,
            tail_q=tail_q,
        )
        if run_transition_metrics["left_delta"].size > 0:
            all_left_delta.append(run_transition_metrics["left_delta"])
        if run_transition_metrics["right_delta"].size > 0:
            all_right_delta.append(run_transition_metrics["right_delta"])

        late_mean = np.mean(zeta_matrix[late_start_idx:], axis=1)
        late_var = np.var(zeta_matrix[late_start_idx:], axis=1, ddof=1) if n_units > 1 else np.zeros(n_checkpoints - late_start_idx)
        late_epochs_run = epochs[late_start_idx:].astype(np.float64)

        run_summaries.append({
            "run_label": run["label"],
            "model_dir": run["model_dir"],
            "n_checkpoints": int(n_checkpoints),
            "n_units": int(n_units),
            "late_start_epoch": late_start_epoch,
            "epoch_min": int(np.min(epochs)),
            "epoch_max": int(np.max(epochs)),
            "n_late_transition_pairs": int(max(0, n_checkpoints - late_start_idx - 1)),
            "n_late_transition_samples": int(run_transition_metrics["n_transition_samples"]),
            "bulk_zeta_median": run_transition_metrics["bulk_center"],
            "conditional_drift_slope": run_transition_metrics["conditional_slope"],
            "kappa_hat": float(-run_transition_metrics["conditional_slope"]),
            "conditional_drift_spearman_rho": run_transition_metrics["spearman_rho"],
            "conditional_drift_spearman_p": run_transition_metrics["spearman_p"],
            "inward_fraction_relative_to_bulk": run_transition_metrics["inward_fraction"],
            "left_tail_zeta_cut": run_transition_metrics["left_cut"],
            "right_tail_zeta_cut": run_transition_metrics["right_cut"],
            "left_tail_delta_median": run_transition_metrics["left_delta_median"],
            "right_tail_delta_median": run_transition_metrics["right_delta_median"],
            "late_mean_slope_per_epoch": fit_linear_slope(late_epochs_run, late_mean),
            "late_var_slope_per_epoch": fit_linear_slope(late_epochs_run, late_var),
        })
        log(
            f"  {run['label']}: checkpoints={n_checkpoints}, units={n_units}, "
            f"late_start_epoch={late_start_epoch}"
        )

    if not all_current:
        raise RuntimeError("No late-training transitions could be constructed from the checkpoint data.")

    current_all = np.concatenate(all_current)
    delta_all = np.concatenate(all_delta)

    drift_rows, bulk_center = summarize_conditional_drift(
        current_all,
        delta_all,
        n_bins=args.n_bins,
        binning=args.binning,
        trim_fraction=args.trim_fraction,
        uniform_clip_quantile=args.uniform_clip_quantile,
    )
    moment_agg_rows = aggregate_moment_rows(moment_rows)

    late_start_epoch_plot = float(max(late_start_epochs)) if late_start_epochs else float("nan")

    save_csv(
        ["bin_left", "bin_right", "zeta_center", "count",
         "delta_median", "delta_trimmed_mean", "delta_q25", "delta_q75",
         "zeta_q25", "zeta_q75"],
        [[
            row["bin_left"], row["bin_right"], row["zeta_center"], row["count"],
            row["delta_median"], row["delta_trimmed_mean"], row["delta_q25"], row["delta_q75"],
            row["zeta_q25"], row["zeta_q75"],
        ] for row in drift_rows],
        os.path.join(args.outdir, "conditional_drift.csv"),
    )

    save_csv(
        ["run_label", "epoch", "is_late", "zeta_mean", "zeta_var", "zeta_median", "zeta_q10", "zeta_q90"],
        [[
            row["run_label"], row["epoch"], row["is_late"], row["zeta_mean"],
            row["zeta_var"], row["zeta_median"], row["zeta_q10"], row["zeta_q90"],
        ] for row in moment_rows],
        os.path.join(args.outdir, "per_seed_moments.csv"),
    )

    save_csv(
        ["epoch", "n_runs", "n_late_runs", "zeta_mean_mean", "zeta_mean_se",
         "zeta_var_mean", "zeta_var_se", "zeta_median_mean", "zeta_median_se",
         "zeta_q10_mean", "zeta_q90_mean"],
        [[
            row["epoch"], row["n_runs"], row["n_late_runs"],
            row["zeta_mean_mean"], row["zeta_mean_se"],
            row["zeta_var_mean"], row["zeta_var_se"],
            row["zeta_median_mean"], row["zeta_median_se"],
            row["zeta_q10_mean"], row["zeta_q90_mean"],
        ] for row in moment_agg_rows],
        os.path.join(args.outdir, "moment_trajectory.csv"),
    )

    plot_conditional_drift(
        drift_rows,
        bulk_center=bulk_center,
        display_name=display_name,
        outpath=os.path.join(args.outdir, "conditional_drift.png"),
    )
    plot_moments(
        moment_agg_rows,
        late_start_epoch=late_start_epoch_plot,
        display_name=display_name,
        outpath=os.path.join(args.outdir, "late_moments.png"),
    )

    pooled_spearman_rho, pooled_spearman_p = safe_spearmanr(current_all, delta_all)
    pooled_inward_fraction = float(np.mean(((current_all - bulk_center) * delta_all) < 0.0))

    late_rows = [row for row in moment_agg_rows if row["epoch"] >= late_start_epoch_plot]
    late_epochs = np.array([row["epoch"] for row in late_rows], dtype=np.float64)
    late_mean = np.array([row["zeta_mean_mean"] for row in late_rows], dtype=np.float64)
    late_var = np.array([row["zeta_var_mean"] for row in late_rows], dtype=np.float64)

    run_conditional_slopes = np.array([row["conditional_drift_slope"] for row in run_summaries], dtype=np.float64)
    run_spearman_rhos = np.array([row["conditional_drift_spearman_rho"] for row in run_summaries], dtype=np.float64)
    run_spearman_ps = np.array([row["conditional_drift_spearman_p"] for row in run_summaries], dtype=np.float64)
    run_inward = np.array([row["inward_fraction_relative_to_bulk"] for row in run_summaries], dtype=np.float64)
    run_left_cuts = np.array([row["left_tail_zeta_cut"] for row in run_summaries], dtype=np.float64)
    run_right_cuts = np.array([row["right_tail_zeta_cut"] for row in run_summaries], dtype=np.float64)
    run_late_mean_slopes = np.array([row["late_mean_slope_per_epoch"] for row in run_summaries], dtype=np.float64)
    run_late_var_slopes = np.array([row["late_var_slope_per_epoch"] for row in run_summaries], dtype=np.float64)
    left_delta_all = np.concatenate(all_left_delta) if all_left_delta else np.zeros(0, dtype=np.float64)
    right_delta_all = np.concatenate(all_right_delta) if all_right_delta else np.zeros(0, dtype=np.float64)

    metrics = {
        "display_name": display_name,
        "n_runs": int(len(runs)),
        "n_transition_pairs": int(transition_count),
        "n_transition_samples": int(current_all.size),
        "bulk_zeta_median": float(bulk_center),
        "conditional_drift_bin_median_slope": fit_linear_slope(
            np.array([row["zeta_center"] for row in drift_rows], dtype=np.float64),
            np.array([row["delta_median"] for row in drift_rows], dtype=np.float64),
        ),
        "conditional_drift_slope": float(np.nanmean(run_conditional_slopes)),
        "conditional_drift_slope_pooled": fit_linear_slope(current_all, delta_all),
        "conditional_drift_slope_mean_across_runs": float(np.nanmean(run_conditional_slopes)),
        "kappa_hat_binned_median": -fit_linear_slope(
            np.array([row["zeta_center"] for row in drift_rows], dtype=np.float64),
            np.array([row["delta_median"] for row in drift_rows], dtype=np.float64),
        ),
        "kappa_hat_pooled": -fit_linear_slope(current_all, delta_all),
        "kappa_hat_mean_across_runs": float(-np.nanmean(run_conditional_slopes)),
        "conditional_drift_spearman_rho": float(np.nanmean(run_spearman_rhos)),
        "conditional_drift_spearman_p": float(np.nanmean(run_spearman_ps)),
        "conditional_drift_spearman_rho_pooled": pooled_spearman_rho,
        "conditional_drift_spearman_p_pooled": pooled_spearman_p,
        "conditional_drift_spearman_rho_mean_across_runs": float(np.nanmean(run_spearman_rhos)),
        "inward_fraction_relative_to_bulk": float(np.nanmean(run_inward)),
        "inward_fraction_relative_to_bulk_pooled": pooled_inward_fraction,
        "inward_fraction_relative_to_bulk_mean_across_runs": float(np.nanmean(run_inward)),
        "left_tail_quantile": tail_q,
        "left_tail_zeta_cut": float(np.nanmedian(run_left_cuts)),
        "right_tail_zeta_cut": float(np.nanmedian(run_right_cuts)),
        "left_tail_delta_median": float(np.median(left_delta_all)) if left_delta_all.size > 0 else float("nan"),
        "right_tail_delta_median": float(np.median(right_delta_all)) if right_delta_all.size > 0 else float("nan"),
        "late_start_epoch_min": float(np.min(late_start_epochs)) if late_start_epochs else float("nan"),
        "late_start_epoch_max": float(np.max(late_start_epochs)) if late_start_epochs else float("nan"),
        "late_mean_slope_per_epoch": float(np.nanmean(run_late_mean_slopes)),
        "late_var_slope_per_epoch": float(np.nanmean(run_late_var_slopes)),
        "late_mean_slope_per_epoch_pooled_common_window": fit_linear_slope(late_epochs, late_mean),
        "late_var_slope_per_epoch_pooled_common_window": fit_linear_slope(late_epochs, late_var),
        "late_mean_slope_per_epoch_mean_across_runs": float(np.nanmean(run_late_mean_slopes)),
        "late_var_slope_per_epoch_mean_across_runs": float(np.nanmean(run_late_var_slopes)),
        "run_summaries": run_summaries,
    }

    save_json(metrics, os.path.join(args.outdir, "metrics.json"))
    save_json(manifest, os.path.join(args.outdir, "manifest.json"))

    log(f"Saved restoring-drift diagnostics to: {args.outdir}")


if __name__ == "__main__":
    main()
