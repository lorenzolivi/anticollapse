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

        def _finite(values):
            return values[np.isfinite(values)]

        def _finite_mean(values):
            finite = _finite(values)
            if finite.size == 0:
                return float("nan")
            return float(np.mean(finite))

        def _se(values):
            values = _finite(values)
            n = int(np.count_nonzero(np.isfinite(values)))
            if n <= 1:
                return 0.0
            return float(np.std(values, ddof=1) / math.sqrt(n))

        aggregated.append({
            "epoch": int(epoch),
            "n_runs": int(len(subset)),
            "n_late_runs": int(sum(int(row["is_late"]) for row in subset)),
            "zeta_mean_mean": _finite_mean(mean_vals),
            "zeta_mean_se": _se(mean_vals),
            "zeta_var_mean": _finite_mean(var_vals),
            "zeta_var_se": _se(var_vals),
            "zeta_median_mean": _finite_mean(med_vals),
            "zeta_median_se": _se(med_vals),
            "zeta_q10_mean": _finite_mean(q10_vals),
            "zeta_q90_mean": _finite_mean(q90_vals),
        })
    return aggregated


def plot_conditional_drift(rows, bulk_center: float, display_name: str, outpath: str):
    if not rows:
        return

    x = np.array([row["zeta_center"] for row in rows], dtype=np.float64)
    y = np.array([row["delta_trimmed_mean"] for row in rows], dtype=np.float64)
    y_lo = np.array([row["delta_q25"] for row in rows], dtype=np.float64)
    y_hi = np.array([row["delta_q75"] for row in rows], dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_lo = y_lo[order]
    y_hi = y_hi[order]

    plt.figure(figsize=(7.2, 4.6))
    plt.fill_between(x, y_lo, y_hi, alpha=0.18, color="tab:blue", linewidth=0.0,
                     label=r"bin IQR of $\Delta\zeta$")
    plt.plot(x, y, marker="o", linewidth=1.8, color="tab:blue",
             label=r"$\hat F(\zeta)$ (trimmed mean)")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    if np.isfinite(bulk_center):
        plt.axvline(bulk_center, color="tab:red", linestyle=":", linewidth=1.2,
                    label=r"bulk median $\tilde\zeta$")
    plt.xlabel(r"current $\zeta$")
    plt.ylabel(r"$\hat F(\zeta)$")
    plt.title(f"Late-training conditional drift: {display_name}")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9, framealpha=0.9)
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


def _fit_constant_vs_linear(z: np.ndarray, d: np.ndarray):
    """
    Compare a constant model d ≈ c against a linear model d ≈ a + b*z
    on the same data. Returns (rss_const, rss_linear, bic_const, bic_linear,
    slope_linear).
    """
    z = np.asarray(z, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    mask = np.isfinite(z) & np.isfinite(d)
    z = z[mask]; d = d[mask]
    n = z.size
    if n < 4:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    c_hat = float(np.mean(d))
    resid_c = d - c_hat
    rss_c = float(np.sum(resid_c * resid_c))

    coeff = np.polyfit(z, d, deg=1)  # [slope, intercept]
    slope = float(coeff[0]); intercept = float(coeff[1])
    resid_l = d - (slope * z + intercept)
    rss_l = float(np.sum(resid_l * resid_l))

    # Gaussian-likelihood BIC with MLE σ²
    def _bic(rss: float, k: int) -> float:
        if not np.isfinite(rss) or rss <= 0.0 or n <= 0:
            return float("nan")
        sigma2 = rss / n
        ll = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
        return float(k * math.log(n) - 2.0 * ll)

    bic_c = _bic(rss_c, k=1)
    bic_l = _bic(rss_l, k=2)
    return rss_c, rss_l, bic_c, bic_l, slope


def compute_tail_saturation_metrics(
    transition_blocks,
    q_low: float,
    trim_fraction: float,
    bootstrap_B: int,
    ci_level: float,
    rng_seed: int,
):
    """
    Targeted empirical check of the far-left-tail closure F(ζ) = κ + o(1).

    transition_blocks: list of (current_zeta_array, delta_array) tuples. Each
        tuple is one late-training checkpoint transition; every entry of the
        array corresponds to a recurrent unit, so each block is the natural
        unit of dependence.

    For each candidate q_low:
      (a) plateau value κ̂_tail : trimmed mean of Δζ on {ζ ≤ quantile_{q_low}(ζ_pool)}
      (b) slope of Δζ vs ζ inside the same slice
      (c) constant-vs-linear model comparison (RSS, BIC) on the slice
      (d) block-bootstrap CI over transition blocks

    The point estimate uses the tail slice defined by the lower q_low-quantile
    of the full pooled ζ. Each bootstrap iteration *re-estimates* the tail
    quantile from its own resampled pool, so the resulting CI captures
    uncertainty in where the slice starts as well as uncertainty in the
    contents of the slice.

    Returns a dict {q_low: {...metrics...}}.
    """
    blocks = [
        (np.asarray(c, dtype=np.float64), np.asarray(d, dtype=np.float64))
        for (c, d) in transition_blocks
        if np.asarray(c).size > 0
    ]
    if not blocks:
        return {}

    # Pool for quantile selection only.
    z_pool = np.concatenate([c for (c, _) in blocks])
    d_pool = np.concatenate([d for (_, d) in blocks])
    mask_pool = np.isfinite(z_pool) & np.isfinite(d_pool)
    z_pool = z_pool[mask_pool]
    d_pool = d_pool[mask_pool]
    if z_pool.size == 0:
        return {}

    q_low = float(np.clip(q_low, 1e-4, 0.5))
    tail_cut = float(np.quantile(z_pool, q_low))

    tail_mask = z_pool <= tail_cut
    n_tail_pool = int(np.count_nonzero(tail_mask))

    if n_tail_pool < 200:
        return {
            "q_low": q_low,
            "tail_zeta_cut": tail_cut,
            "n_tail_samples": n_tail_pool,
            "skipped": True,
            "reason": f"insufficient tail samples ({n_tail_pool} < 200)",
        }

    z_tail = z_pool[tail_mask]
    d_tail = d_pool[tail_mask]

    kappa_tail = trimmed_mean(d_tail, trim_fraction)
    rss_c, rss_l, bic_c, bic_l, slope_tail = _fit_constant_vs_linear(z_tail, d_tail)

    # Block-bootstrap over transition blocks.
    # On each resample we re-estimate the tail cutoff from the resampled
    # pooled ζ, so the CI captures both (i) sampling variability in the
    # slope / plateau given the slice, and (ii) uncertainty in where the
    # slice starts. The point estimates above still use the full-data
    # cutoff tail_cut.
    rng = np.random.default_rng(rng_seed)
    n_blocks = len(blocks)
    B = int(bootstrap_B)
    kappa_samples = np.empty(B, dtype=np.float64)
    slope_samples = np.empty(B, dtype=np.float64)
    delta_bic_samples = np.empty(B, dtype=np.float64)
    tail_cut_samples = np.empty(B, dtype=np.float64)
    n_tail_samples_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_blocks, size=n_blocks)

        # First pass: build the resampled pooled ζ so we can re-estimate
        # the lower q_low-quantile on THIS bootstrap sample.
        z_pool_blocks = []
        d_pool_blocks = []
        for i in idx:
            c_i, d_i = blocks[i]
            m = np.isfinite(c_i) & np.isfinite(d_i)
            if not np.any(m):
                continue
            z_pool_blocks.append(c_i[m])
            d_pool_blocks.append(d_i[m])
        if not z_pool_blocks:
            kappa_samples[b] = np.nan
            slope_samples[b] = np.nan
            delta_bic_samples[b] = np.nan
            tail_cut_samples[b] = np.nan
            n_tail_samples_boot[b] = 0.0
            continue
        z_pool_b = np.concatenate(z_pool_blocks)
        d_pool_b = np.concatenate(d_pool_blocks)
        if z_pool_b.size < 10:
            kappa_samples[b] = np.nan
            slope_samples[b] = np.nan
            delta_bic_samples[b] = np.nan
            tail_cut_samples[b] = np.nan
            n_tail_samples_boot[b] = 0.0
            continue

        tail_cut_b = float(np.quantile(z_pool_b, q_low))
        tail_cut_samples[b] = tail_cut_b

        tm = z_pool_b <= tail_cut_b
        z_b = z_pool_b[tm]
        d_b = d_pool_b[tm]
        n_tail_samples_boot[b] = float(z_b.size)

        if z_b.size < 4:
            kappa_samples[b] = np.nan
            slope_samples[b] = np.nan
            delta_bic_samples[b] = np.nan
            continue
        kappa_samples[b] = trimmed_mean(d_b, trim_fraction)
        _, _, bic_c_b, bic_l_b, slope_b = _fit_constant_vs_linear(z_b, d_b)
        slope_samples[b] = slope_b
        delta_bic_samples[b] = (bic_l_b - bic_c_b)

    alpha_ci = (1.0 - float(ci_level)) / 2.0
    def _pct_ci(arr: np.ndarray):
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return float("nan"), float("nan")
        lo = float(np.quantile(valid, alpha_ci))
        hi = float(np.quantile(valid, 1.0 - alpha_ci))
        return lo, hi

    kappa_lo, kappa_hi = _pct_ci(kappa_samples)
    slope_lo, slope_hi = _pct_ci(slope_samples)
    dbic_lo, dbic_hi = _pct_ci(delta_bic_samples)
    tail_cut_lo, tail_cut_hi = _pct_ci(tail_cut_samples)

    # Report the distribution of per-iteration tail cutoffs so the
    # reader can see how stable the slice boundary is across resamples.
    tail_cut_bs_mean = float(np.nanmean(tail_cut_samples)) if np.any(np.isfinite(tail_cut_samples)) else float("nan")
    tail_cut_bs_median = float(np.nanmedian(tail_cut_samples)) if np.any(np.isfinite(tail_cut_samples)) else float("nan")
    n_tail_bs_median = float(np.nanmedian(n_tail_samples_boot)) if np.any(np.isfinite(n_tail_samples_boot)) else float("nan")

    # Decisions
    positivity_pass = bool(np.isfinite(kappa_lo) and kappa_lo > 0.0)
    flatness_pass = bool(
        np.isfinite(slope_lo) and np.isfinite(slope_hi)
        and slope_lo < 0.0 < slope_hi
    )
    # Prefer constant model if ΔBIC = BIC_linear - BIC_constant >= 0.
    constant_preferred = bool(
        np.isfinite(bic_c) and np.isfinite(bic_l) and (bic_l - bic_c) >= 0.0
    )

    return {
        "q_low": q_low,
        "tail_zeta_cut": tail_cut,
        "tail_zeta_cut_ci_lo": tail_cut_lo,
        "tail_zeta_cut_ci_hi": tail_cut_hi,
        "tail_zeta_cut_bs_mean": tail_cut_bs_mean,
        "tail_zeta_cut_bs_median": tail_cut_bs_median,
        "n_tail_samples": n_tail_pool,
        "n_tail_samples_bs_median": n_tail_bs_median,
        "n_blocks": n_blocks,
        "kappa_tail": kappa_tail,
        "kappa_tail_ci_lo": kappa_lo,
        "kappa_tail_ci_hi": kappa_hi,
        "slope_tail": slope_tail,
        "slope_tail_ci_lo": slope_lo,
        "slope_tail_ci_hi": slope_hi,
        "rss_constant": rss_c,
        "rss_linear": rss_l,
        "bic_constant": bic_c,
        "bic_linear": bic_l,
        "delta_bic_linear_minus_constant": (bic_l - bic_c) if np.isfinite(bic_c) and np.isfinite(bic_l) else float("nan"),
        "delta_bic_ci_lo": dbic_lo,
        "delta_bic_ci_hi": dbic_hi,
        "positivity_pass": positivity_pass,
        "flatness_pass": flatness_pass,
        "constant_preferred": constant_preferred,
        "bootstrap_B": B,
        "ci_level": float(ci_level),
        "trim_fraction": float(trim_fraction),
        "skipped": False,
    }


def plot_combined_drift_with_tail(
    drift_rows,
    bulk_center: float,
    tail_metrics_primary: dict,
    display_name: str,
    outpath: str,
):
    """
    Overlay the binned conditional drift F̂(ζ) with a shaded tail slice
    (ζ ≤ tail_zeta_cut) and a horizontal dashed line at κ̂_tail with its
    bootstrap CI band.
    """
    if not drift_rows:
        return
    x = np.array([row["zeta_center"] for row in drift_rows], dtype=np.float64)
    y = np.array([row["delta_trimmed_mean"] for row in drift_rows], dtype=np.float64)
    y_lo = np.array([row["delta_q25"] for row in drift_rows], dtype=np.float64)
    y_hi = np.array([row["delta_q75"] for row in drift_rows], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]; y = y[order]; y_lo = y_lo[order]; y_hi = y_hi[order]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.fill_between(x, y_lo, y_hi, alpha=0.15, color="tab:blue", linewidth=0.0,
                    label=r"bin IQR of $\Delta\zeta$")
    ax.plot(x, y, marker="o", linewidth=1.8, color="tab:blue",
            label=r"$\hat F(\zeta)$ (trimmed mean)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    if np.isfinite(bulk_center):
        ax.axvline(bulk_center, color="tab:red", linestyle=":", linewidth=1.2,
                   label=r"bulk median $\tilde\zeta$")

    if tail_metrics_primary and not tail_metrics_primary.get("skipped", True):
        tail_cut = tail_metrics_primary["tail_zeta_cut"]
        kappa = tail_metrics_primary["kappa_tail"]
        klo = tail_metrics_primary["kappa_tail_ci_lo"]
        khi = tail_metrics_primary["kappa_tail_ci_hi"]

        x_left = float(min(np.min(x), tail_cut)) - 0.05 * (np.max(x) - np.min(x) + 1e-9)
        ax.axvspan(x_left, tail_cut, color="tab:orange", alpha=0.12,
                   label=r"tail slice $\zeta \leq \zeta_{q_{\mathrm{low}}}$")
        ax.axvline(tail_cut, color="tab:orange", linestyle="--", linewidth=1.1)
        ax.hlines(kappa, x_left, tail_cut, color="tab:orange", linestyle="-", linewidth=2.2,
                  label=r"$\hat\kappa_{\mathrm{tail}}$")
        if np.isfinite(klo) and np.isfinite(khi):
            ax.fill_between([x_left, tail_cut], [klo, klo], [khi, khi],
                            color="tab:orange", alpha=0.25, linewidth=0.0)

    ax.set_xlabel(r"current $\zeta$")
    ax.set_ylabel(r"$\hat F(\zeta)$")
    ax.set_title(f"Far-tail closure diagnostic: {display_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
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

    # Far-tail saturation diagnostic (targeted check of the F(ζ)=κ+o(1) closure)
    parser.add_argument("--tail_q_low_primary", type=float, default=0.10,
                        help="Primary lower quantile defining the far-tail slice {ζ ≤ quantile_{q_low}(ζ)}.")
    parser.add_argument("--tail_q_low_sweep", type=str, default="0.05,0.10,0.15",
                        help="Comma-separated sweep of q_low values for robustness.")
    parser.add_argument("--tail_trim_fraction", type=float, default=0.1,
                        help="Trim fraction used for the robust κ̂_tail estimate.")
    parser.add_argument("--tail_bootstrap_B", type=int, default=2000,
                        help="Number of block-bootstrap resamples for the far-tail CI.")
    parser.add_argument("--tail_ci_level", type=float, default=0.90,
                        help="Confidence level for the block-bootstrap CI on κ̂_tail and slope.")
    parser.add_argument("--tail_bootstrap_seed", type=int, default=20260410,
                        help="Seed for the block-bootstrap RNG (for reproducibility).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = discover_model_runs(args.input_dir, args.model)
    display_name = infer_display_name(args.input_dir, args.model, runs)

    log(f"Found {len(runs)} run(s) for {display_name}")
    low_n_runs_warning = None
    if len(runs) < 3:
        low_n_runs_warning = (
            f"n_runs={len(runs)} < 3: the cross-run variance/SE estimates are "
            f"unreliable, and the drift-closure validation figure should NOT "
            f"be treated as publication-quality with this few seeds."
        )
        log("=" * 72)
        log("WARNING: " + low_n_runs_warning)
        log("=" * 72)

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
            zeta_finite = zeta[np.isfinite(zeta)]
            n_finite = int(zeta_finite.size)
            if n_finite == 0:
                z_mean = z_var = z_med = z_q10 = z_q90 = float("nan")
            else:
                z_mean = float(np.mean(zeta_finite))
                z_var = float(np.var(zeta_finite, ddof=1)) if n_finite > 1 else 0.0
                z_med = float(np.median(zeta_finite))
                z_q10 = float(np.quantile(zeta_finite, 0.10))
                z_q90 = float(np.quantile(zeta_finite, 0.90))
            moment_rows.append({
                "run_label": run["label"],
                "epoch": int(epoch),
                "is_late": int(i >= late_start_idx),
                "zeta_mean": z_mean,
                "zeta_var": z_var,
                "zeta_median": z_med,
                "zeta_q10": z_q10,
                "zeta_q90": z_q90,
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

        late_zeta = zeta_matrix[late_start_idx:]
        late_mean_vals = []
        late_var_vals = []
        for zeta in late_zeta:
            zeta_finite = zeta[np.isfinite(zeta)]
            if zeta_finite.size == 0:
                late_mean_vals.append(float("nan"))
                late_var_vals.append(float("nan"))
            else:
                late_mean_vals.append(float(np.mean(zeta_finite)))
                late_var_vals.append(
                    float(np.var(zeta_finite, ddof=1))
                    if zeta_finite.size > 1
                    else 0.0
                )
        late_mean = np.asarray(late_mean_vals, dtype=np.float64)
        late_var = np.asarray(late_var_vals, dtype=np.float64)
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

    # ------------------------------------------------------------------
    # Far-left-tail saturation diagnostic (closure F(ζ) = κ + o(1)).
    # Transition-level blocks: each (current, delta) pair from the main
    # loop above is one checkpoint-transition "block" (all neurons at once),
    # which is the natural unit of dependence for a block-bootstrap CI.
    # ------------------------------------------------------------------
    transition_blocks = list(zip(all_current, all_delta))

    q_low_sweep = []
    for tok in str(args.tail_q_low_sweep).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            q_low_sweep.append(float(tok))
        except ValueError:
            continue
    if float(args.tail_q_low_primary) not in q_low_sweep:
        q_low_sweep.append(float(args.tail_q_low_primary))
    q_low_sweep = sorted(set(q_low_sweep))

    tail_results = {}
    for q_low in q_low_sweep:
        res = compute_tail_saturation_metrics(
            transition_blocks=transition_blocks,
            q_low=q_low,
            trim_fraction=float(args.tail_trim_fraction),
            bootstrap_B=int(args.tail_bootstrap_B),
            ci_level=float(args.tail_ci_level),
            rng_seed=int(args.tail_bootstrap_seed),
        )
        tail_results[f"q_low_{q_low:.3f}"] = res
        if res and not res.get("skipped", False):
            log(
                f"  tail saturation q_low={q_low:.3f}: "
                f"κ̂_tail={res['kappa_tail']:.3e} "
                f"[{res['kappa_tail_ci_lo']:.3e}, {res['kappa_tail_ci_hi']:.3e}], "
                f"slope={res['slope_tail']:.3e} "
                f"[{res['slope_tail_ci_lo']:.3e}, {res['slope_tail_ci_hi']:.3e}], "
                f"ΔBIC(lin-const)={res['delta_bic_linear_minus_constant']:.2f}, "
                f"positivity={res['positivity_pass']}, flatness={res['flatness_pass']}, "
                f"constant_preferred={res['constant_preferred']}"
            )
            log(
                f"    tail cutoff ζ_{{q_low}}={res['tail_zeta_cut']:.3e} "
                f"(bootstrap CI [{res['tail_zeta_cut_ci_lo']:.3e}, {res['tail_zeta_cut_ci_hi']:.3e}], "
                f"bs median {res['tail_zeta_cut_bs_median']:.3e}, "
                f"n_tail pool={res['n_tail_samples']}, bs median n_tail={res['n_tail_samples_bs_median']:.0f})"
            )
        else:
            log(f"  tail saturation q_low={q_low:.3f}: skipped ({res.get('reason', 'n/a')})")

    primary_key = f"q_low_{float(args.tail_q_low_primary):.3f}"
    tail_primary = tail_results.get(primary_key, {})

    save_json(
        {
            "n_runs": int(len(runs)),
            "low_n_runs_warning": low_n_runs_warning,
            "q_low_primary": float(args.tail_q_low_primary),
            "q_low_sweep": q_low_sweep,
            "tail_trim_fraction": float(args.tail_trim_fraction),
            "tail_bootstrap_B": int(args.tail_bootstrap_B),
            "tail_ci_level": float(args.tail_ci_level),
            "results": tail_results,
            "primary": tail_primary,
        },
        os.path.join(args.outdir, "tail_saturation.json"),
    )

    # Flat CSV for easy plotting/tabulation
    tail_csv_rows = []
    for k in sorted(tail_results.keys()):
        r = tail_results[k]
        if not r:
            continue
        tail_csv_rows.append([
            r.get("q_low", float("nan")),
            r.get("tail_zeta_cut", float("nan")),
            r.get("tail_zeta_cut_ci_lo", float("nan")),
            r.get("tail_zeta_cut_ci_hi", float("nan")),
            r.get("tail_zeta_cut_bs_mean", float("nan")),
            r.get("tail_zeta_cut_bs_median", float("nan")),
            int(r.get("n_tail_samples", 0)),
            r.get("n_tail_samples_bs_median", float("nan")),
            int(r.get("n_blocks", 0)),
            r.get("kappa_tail", float("nan")),
            r.get("kappa_tail_ci_lo", float("nan")),
            r.get("kappa_tail_ci_hi", float("nan")),
            r.get("slope_tail", float("nan")),
            r.get("slope_tail_ci_lo", float("nan")),
            r.get("slope_tail_ci_hi", float("nan")),
            r.get("bic_constant", float("nan")),
            r.get("bic_linear", float("nan")),
            r.get("delta_bic_linear_minus_constant", float("nan")),
            int(bool(r.get("positivity_pass", False))),
            int(bool(r.get("flatness_pass", False))),
            int(bool(r.get("constant_preferred", False))),
            int(bool(r.get("skipped", False))),
        ])
    save_csv(
        ["q_low", "tail_zeta_cut", "tail_zeta_cut_ci_lo", "tail_zeta_cut_ci_hi",
         "tail_zeta_cut_bs_mean", "tail_zeta_cut_bs_median",
         "n_tail_samples", "n_tail_samples_bs_median", "n_blocks",
         "kappa_tail", "kappa_tail_ci_lo", "kappa_tail_ci_hi",
         "slope_tail", "slope_tail_ci_lo", "slope_tail_ci_hi",
         "bic_constant", "bic_linear", "delta_bic_linear_minus_constant",
         "positivity_pass", "flatness_pass", "constant_preferred", "skipped"],
        tail_csv_rows,
        os.path.join(args.outdir, "tail_saturation.csv"),
    )

    plot_combined_drift_with_tail(
        drift_rows,
        bulk_center=bulk_center,
        tail_metrics_primary=tail_primary,
        display_name=display_name,
        outpath=os.path.join(args.outdir, "conditional_drift_with_tail.png"),
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
