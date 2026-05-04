#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-seed plotting driver — phase-trajectory engine
====================================================

Reads aggregated results (phase_trajectory_aggregated.csv with _mean/_se columns)
and produces dynamics plots WITH ±1 SE error bands.

For final-checkpoint plots (histograms, CCDFs, alpha stable fit), it delegates
to the standard single-seed plotting scripts (which use the last-seed checkpoint
data copied into the aggregated directory).

Typically called by main_exp1.py after aggregation, but can also be run standalone:

    python plot_exp1_all_multiseed.py \
        --agg_dir results/exp2_phase_full/adamw/aggregated \
        --outdir  results/exp2_phase_full/adamw/plots \
        --dpi 600 --tau_cap 1e6
"""

import os, sys, subprocess, argparse, csv, datetime
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helpers
# ============================================================

def safe_read_csv_named(path: str):
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


def find_model_dirs(agg_dir: str):
    """
    In the aggregated directory, model dirs are direct children:
      agg_dir/const/, agg_dir/diag/, agg_dir/gru/, ...

    Each must contain phase_trajectory_aggregated.csv (multi-seed)
    OR at minimum phase_trajectory.csv (single-seed fallback).

    Returns list of dicts: {"dir": ..., "name": ..., "label": ...}
    """
    agg_dir = os.path.abspath(agg_dir)
    out = []
    if not os.path.isdir(agg_dir):
        return out

    for entry in sorted(os.listdir(agg_dir)):
        mdir = os.path.join(agg_dir, entry)
        if not os.path.isdir(mdir):
            continue
        # skip known non-model dirs
        if entry in ("plots", "plots_exp1", "logs", "__pycache__"):
            continue
        # must have at least a phase trajectory
        has_agg = os.path.exists(os.path.join(mdir, "phase_trajectory_aggregated.csv"))
        has_std = os.path.exists(os.path.join(mdir, "phase_trajectory.csv"))
        if has_agg or has_std:
            out.append({"dir": mdir, "name": entry, "label": entry})

    return out


def load_aggregated_trajectory(model_dir: str):
    """
    Load phase_trajectory_aggregated.csv (columns: epoch, <col>_mean, <col>_se).
    Falls back to phase_trajectory.csv (no SE) if aggregated not found.
    Returns dict with keys: epoch, and for each metric: <col>_mean, <col>_se.
    """
    agg_path = os.path.join(model_dir, "phase_trajectory_aggregated.csv")
    std_path = os.path.join(model_dir, "phase_trajectory.csv")

    data = safe_read_csv_named(agg_path)
    if data is not None:
        names = data.dtype.names or ()
        if "epoch" not in names:
            data = None

    is_aggregated = data is not None

    if data is None:
        # fallback to standard (no SE)
        data = safe_read_csv_named(std_path)
        if data is None:
            return None
        names = data.dtype.names or ()
        if "epoch" not in names:
            return None

    epoch = np.array(data["epoch"], dtype=int)
    order = np.argsort(epoch)
    epoch = epoch[order]

    result = {"epoch": epoch, "is_aggregated": is_aggregated}

    # columns we care about for dynamics plots
    METRICS = [
        "alpha_hat", "sigma_alpha_hat", "alpha_hat_std", "alpha_hat_se",
        "beta_hat", "beta_r2",
        "beta_median", "beta_lo", "beta_hi", "p_beta_lt1",
        "beta_bootstrap_B_eff",
        "tau_mean", "tau_q90", "tau_q99",
        "tau_fit_r2_mean", "tau_fit_n_valid",
    ]

    names = data.dtype.names or ()

    for col in METRICS:
        if is_aggregated:
            mean_key = f"{col}_mean"
            se_key = f"{col}_se"
            if mean_key in names:
                result[f"{col}_mean"] = np.array(data[mean_key], dtype=float)[order]
            else:
                result[f"{col}_mean"] = np.full(epoch.shape, np.nan)
            if se_key in names:
                result[f"{col}_se"] = np.array(data[se_key], dtype=float)[order]
            else:
                result[f"{col}_se"] = np.full(epoch.shape, np.nan)
        else:
            # standard format: column is the mean, no SE
            if col in names:
                result[f"{col}_mean"] = np.array(data[col], dtype=float)[order]
            else:
                result[f"{col}_mean"] = np.full(epoch.shape, np.nan)
            result[f"{col}_se"] = np.full(epoch.shape, np.nan)

    return result


def load_aggregated_learning_curve(model_dir: str, model_name: str):
    """
    Load learning_curve_aggregated.csv (columns: epoch, train_loss_mean, train_loss_se).
    Falls back to <model>_learning_curve.csv (no SE).
    """
    agg_path = os.path.join(model_dir, "learning_curve_aggregated.csv")
    std_path = os.path.join(model_dir, f"{model_name}_learning_curve.csv")

    data = safe_read_csv_named(agg_path)
    if data is not None:
        names = data.dtype.names or ()
        if "epoch" in names and "train_loss_mean" in names:
            epoch = np.array(data["epoch"], dtype=int)
            order = np.argsort(epoch)
            mean = np.array(data["train_loss_mean"], dtype=float)[order]
            se_col = "train_loss_se"
            se = np.array(data[se_col], dtype=float)[order] if se_col in names else np.full_like(mean, np.nan)
            return {"epoch": epoch[order], "mean": mean, "se": se, "is_aggregated": True}

    # fallback
    data = safe_read_csv_named(std_path)
    if data is None:
        return None
    names = data.dtype.names or ()
    if "epoch" not in names or "train_loss" not in names:
        return None
    epoch = np.array(data["epoch"], dtype=int)
    loss = np.array(data["train_loss"], dtype=float)
    mask = np.isfinite(epoch) & np.isfinite(loss)
    epoch, loss = epoch[mask], loss[mask]
    order = np.argsort(epoch)
    return {"epoch": epoch[order], "mean": loss[order], "se": np.full_like(loss[order], np.nan), "is_aggregated": False}


# ============================================================
# Dynamics plotting (with ±SE bands)
# ============================================================

def plot_dynamics_with_se(entries, outdir, dpi):
    """
    Phase dynamics plots with ±1 SE error bands.
    """
    traj_by_label = {}
    for e in entries:
        tr = load_aggregated_trajectory(e["dir"])
        if tr is not None:
            traj_by_label[e["label"]] = tr

    if not traj_by_label:
        print("[multiseed] No phase trajectories found, skipping dynamics plots.")
        return

    labels = sorted(traj_by_label.keys())

    # ---- alpha_hat(t) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["alpha_hat_mean"]
        se = tr["alpha_hat_se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], y[m] - se[m], y[m] + se[m],
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\alpha}(t)$")
        plt.title(r"Gradient tail index dynamics (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_alpha_hat_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- beta_hat(t) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["beta_hat_mean"]
        se = tr["beta_hat_se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], y[m] - se[m], y[m] + se[m],
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\beta}(t)$")
        plt.title(r"Time-scale tail exponent dynamics (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_beta_hat_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- beta_median(t) with bootstrap 90% stability interval ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["beta_median_mean"]
        blo = tr.get("beta_lo_mean", np.full_like(ep, np.nan, dtype=float))
        bhi = tr.get("beta_hi_mean", np.full_like(ep, np.nan, dtype=float))
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            # Bootstrap stability interval (not cross-seed SE — within-run uncertainty)
            boot_m = m & np.isfinite(blo) & np.isfinite(bhi)
            if np.any(boot_m):
                plt.fill_between(ep[boot_m], blo[boot_m], bhi[boot_m],
                                 alpha=0.15, color=line.get_color())
    if any_data:
        plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label=r"$\beta=1$")
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\beta}_{\mathrm{median}}(t)$")
        plt.title(r"Bootstrap $\hat{\beta}$ dynamics (median $\pm$ 90% stability interval)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_beta_median_bootstrap_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- P(beta < 1) dynamics ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr.get("p_beta_lt1_mean", np.full_like(ep, np.nan, dtype=float))
        se = tr.get("p_beta_lt1_se", np.full_like(ep, np.nan, dtype=float))
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], np.maximum(y[m] - se[m], 0),
                                 np.minimum(y[m] + se[m], 1),
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        plt.xlabel("epoch")
        plt.ylabel(r"$P(\hat{\beta}<1)$")
        plt.ylim(-0.05, 1.05)
        plt.title(r"Bootstrap probability $P(\hat{\beta}<1)$ over training")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_p_beta_lt1_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- tau_mean(t) (LOG SCALE) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["tau_mean_mean"]
        se = tr["tau_mean_se"]
        m = np.isfinite(ep) & np.isfinite(y) & (y > 0)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                lo = np.maximum(y[m] - se[m], 1e-6)
                hi = y[m] + se[m]
                plt.fill_between(ep[m], lo, hi, alpha=0.18, color=line.get_color())
    if any_data:
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel(r"$\mathbb{E}[\tau](t)$")
        plt.title(r"Mean $\tau$ dynamics — log scale (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_tau_mean_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- tau_q99(t) (LOG SCALE) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["tau_q99_mean"]
        se = tr["tau_q99_se"]
        m = np.isfinite(ep) & np.isfinite(y) & (y > 0)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                lo = np.maximum(y[m] - se[m], 1e-6)
                hi = y[m] + se[m]
                plt.fill_between(ep[m], lo, hi, alpha=0.18, color=line.get_color())
    if any_data:
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel(r"$\tau_{0.99}(t)$")
        plt.title(r"$\tau_{0.99}$ dynamics — log scale (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_tau_q99_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- beta_r2(t) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["beta_r2_mean"]
        se = tr["beta_r2_se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], y[m] - se[m], y[m] + se[m],
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.xlabel("epoch")
        plt.ylabel(r"$R^2$")
        plt.ylim(-0.05, 1.05)
        plt.title(r"CCDF tail-fit quality over time (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_beta_r2_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- tau_fit_r2_mean(t) (slope fit quality) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["tau_fit_r2_mean_mean"]
        se = tr["tau_fit_r2_mean_se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], y[m] - se[m], y[m] + se[m],
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.xlabel("epoch")
        plt.ylabel(r"mean $R^2$ (slope fits)")
        plt.ylim(-0.05, 1.05)
        plt.title(r"Per-neuron slope-fit quality (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_tau_slope_r2_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- Phase trajectory in (alpha, beta) ----
    plt.figure(figsize=(6.8, 5.2))
    any_traj = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        a = tr["alpha_hat_mean"]
        b_med = tr.get("beta_median_mean", np.full_like(a, np.nan, dtype=float))
        b_hat = tr["beta_hat_mean"]
        b = np.where(np.isfinite(b_med), b_med, b_hat)
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            continue
        any_traj = True
        plt.plot(a[m], b[m], marker="o", markersize=3.0, linewidth=1.6, label=label)
        # annotate final
        plt.text(a[m][-1], b[m][-1], f" {label}", fontsize=8, va="center")
        # direction arrows
        a_m, b_m = a[m], b[m]
        n_pts = a_m.size
        for frac in [0.33, 0.66]:
            idx_arr = min(int(frac * n_pts), n_pts - 2)
            if 0 <= idx_arr < n_pts - 1:
                plt.annotate("", xy=(a_m[idx_arr + 1], b_m[idx_arr + 1]),
                             xytext=(a_m[idx_arr], b_m[idx_arr]),
                             arrowprops=dict(arrowstyle="->",
                                             color=plt.gca().lines[-1].get_color(),
                                             lw=1.5))
    if any_traj:
        plt.xlabel(r"$\hat{\alpha}(t)$")
        plt.ylabel(r"$\hat{\beta}_{\mathrm{med}}(t)$")
        plt.title(r"Phase trajectory $(\hat{\alpha}, \hat{\beta}_{\mathrm{med}})$ — multi-seed mean")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_phase_trajectory_alpha_beta.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- sigma_alpha_hat(t) ----
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for label in labels:
        tr = traj_by_label[label]
        ep = tr["epoch"]
        y = tr["sigma_alpha_hat_mean"]
        se = tr["sigma_alpha_hat_se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if np.any(m):
            any_data = True
            line, = plt.plot(ep[m], y[m], linewidth=2.0, label=label)
            if np.any(np.isfinite(se[m]) & (se[m] > 0)):
                plt.fill_between(ep[m], y[m] - se[m], y[m] + se[m],
                                 alpha=0.18, color=line.get_color())
    if any_data:
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\sigma}_{\alpha}(t)$")
        plt.title(r"Gradient projection scale dynamics (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_sigma_alpha_hat_vs_epoch.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[multiseed] Saved dynamics plots (with SE bands) to: {outdir}")


def plot_learning_curves_with_se(entries, outdir, dpi, ylog=False):
    """Learning curves with ±1 SE error bands."""
    plt.figure(figsize=(7.6, 4.8))
    any_data = False

    for e in entries:
        lc = load_aggregated_learning_curve(e["dir"], e["name"])
        if lc is None:
            continue
        ep = lc["epoch"]
        y = lc["mean"]
        se = lc["se"]
        m = np.isfinite(ep) & np.isfinite(y)
        if not np.any(m):
            continue
        any_data = True
        line, = plt.plot(ep[m], y[m], linewidth=1.8, label=e["label"])
        if np.any(np.isfinite(se[m]) & (se[m] > 0)):
            lo = np.maximum(y[m] - se[m], 1e-12) if ylog else y[m] - se[m]
            hi = y[m] + se[m]
            plt.fill_between(ep[m], lo, hi, alpha=0.18, color=line.get_color())

    if any_data:
        if ylog:
            plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("train loss (MSE)")
        plt.title(r"Learning curves (mean $\pm$ 1 SE)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ms_learning_curves_train_loss.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    if any_data:
        print(f"[multiseed] Saved learning curve plot to: {outdir}")


# ============================================================
# Delegate final-checkpoint plots to single-seed scripts
# ============================================================

SINGLE_SEED_SCRIPTS = [
    "plot_exp1_envelopes.py",
    "plot_exp1_tau_spectrum.py",
    "plot_exp1_alpha_grad.py",
    "plot_exp1_phase_summary.py",
    # learning_curves handled by our multiseed plotter above
]


def run_single_seed_scripts(agg_dir, outdir, dpi, extra_args):
    """
    Run the standard single-seed plotting scripts on the aggregated directory.
    These handle final-checkpoint histograms, CCDFs, alpha stable fits, etc.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ok, fail = [], []
    for script in SINGLE_SEED_SCRIPTS:
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            print(f"[multiseed] [SKIP] Missing script: {script_path}")
            fail.append(script)
            continue

        cmd = [sys.executable, script_path, "--indir", agg_dir, "--outdir", outdir, "--dpi", str(dpi)]

        # forward per-script args
        if script == "plot_exp1_tau_spectrum.py":
            if "tau_cap" in extra_args:
                cmd += ["--tau_cap", str(extra_args["tau_cap"])]
            if "min_beta_r2" in extra_args:
                cmd += ["--min_beta_r2", str(extra_args["min_beta_r2"])]
            if "hist_bins" in extra_args:
                cmd += ["--hist_bins", str(extra_args["hist_bins"])]
        if script == "plot_exp1_alpha_grad.py":
            if "bins" in extra_args:
                cmd += ["--bins", str(extra_args["bins"])]
        if script == "plot_exp1_phase_summary.py":
            if "min_r2" in extra_args:
                cmd += ["--min_r2", str(extra_args["min_r2"])]
        if script == "plot_exp1_envelopes.py":
            if "debug" in extra_args:
                cmd += ["--debug", str(extra_args["debug"])]

        print(f"[multiseed] [RUN] {' '.join(cmd)}", flush=True)
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)
            ok.append(script)
        except subprocess.CalledProcessError as e:
            fail.append(script)
            print(f"[multiseed] [FAIL] {script}: returncode={e.returncode}")
            if e.stderr:
                for line in e.stderr.strip().split("\n")[:5]:
                    print(f"  {line}")

    print(f"[multiseed] Single-seed scripts: ok={len(ok)} fail={len(fail)}")
    if fail:
        print(f"[multiseed] Failed: {', '.join(fail)}")

    return ok, fail


# ============================================================
# Step registry
# ============================================================

ALL_STEPS = [
    "dynamics",
    "learning_curves",
    "envelopes",
    "tau_spectrum",
    "alpha_grad",
    "phase_summary",
]

STEP_DESCRIPTIONS = {
    "dynamics":        "Multi-seed dynamics plots (alpha, beta, tau, etc. with ±SE bands)",
    "learning_curves": "Multi-seed learning curves with ±SE bands",
    "envelopes":       "Envelope scaling plots (delegates to plot_exp1_envelopes.py)",
    "tau_spectrum":    "Tau spectrum / CCDF plots (delegates to plot_exp1_tau_spectrum.py)",
    "alpha_grad":      "Alpha gradient distribution plots (delegates to plot_exp1_alpha_grad.py)",
    "phase_summary":   "Phase summary plots (delegates to plot_exp1_phase_summary.py)",
}

# Map step names to single-seed scripts (for delegated steps)
STEP_TO_SCRIPT = {
    "envelopes":     "plot_exp1_envelopes.py",
    "tau_spectrum":  "plot_exp1_tau_spectrum.py",
    "alpha_grad":    "plot_exp1_alpha_grad.py",
    "phase_summary": "plot_exp1_phase_summary.py",
}


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Multi-seed plotting driver for Experiment 1")
    ap.add_argument("--agg_dir", required=True,
                    help="Aggregated results directory (e.g., results/exp2_phase_full/adamw/aggregated)")
    ap.add_argument("--outdir", default=None,
                    help="Plot output directory (default: sibling plots/ dir)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylog", type=int, default=0, help="If 1, log-scale y on learning curves")

    # step control
    ap.add_argument("--skip", type=str, default=None,
                    help="Comma-separated steps to skip (e.g., envelopes,tau_spectrum)")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated steps to run (skips all others)")
    ap.add_argument("--list", action="store_true",
                    help="Print available steps and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="Show which steps would run and exit")

    # forwarded to single-seed scripts
    ap.add_argument("--tau_cap", type=float, default=None)
    ap.add_argument("--min_r2", type=float, default=None)
    ap.add_argument("--min_beta_r2", type=float, default=None)
    ap.add_argument("--hist_bins", type=int, default=None)
    ap.add_argument("--bins", type=int, default=None)
    ap.add_argument("--debug", type=int, default=None)
    ap.add_argument("--show_unreliable", action="store_true",
                    help="Forward --show_unreliable to alpha_grad plotter")

    args = ap.parse_args()

    # --list: print steps and exit
    if args.list:
        print("Available plotting steps:")
        for step in ALL_STEPS:
            desc = STEP_DESCRIPTIONS.get(step, "")
            print(f"  {step:20s}  {desc}")
        return

    # Determine which steps to run
    if args.only:
        steps_to_run = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in steps_to_run if s not in ALL_STEPS]
        if unknown:
            print(f"[multiseed] WARNING: unknown steps in --only: {unknown}")
        steps_to_run = [s for s in steps_to_run if s in ALL_STEPS]
    else:
        steps_to_run = list(ALL_STEPS)

    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
        unknown = skip_set - set(ALL_STEPS)
        if unknown:
            print(f"[multiseed] WARNING: unknown steps in --skip: {unknown}")
        steps_to_run = [s for s in steps_to_run if s not in skip_set]

    # --dry_run: show plan and exit
    if args.dry_run:
        print(f"[multiseed] Dry run — steps that would execute:")
        for step in steps_to_run:
            desc = STEP_DESCRIPTIONS.get(step, "")
            print(f"  {step:20s}  {desc}")
        skipped = [s for s in ALL_STEPS if s not in steps_to_run]
        if skipped:
            print(f"[multiseed] Skipped steps: {skipped}")
        return

    agg_dir = os.path.abspath(args.agg_dir)
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    else:
        outdir = os.path.join(os.path.dirname(agg_dir), "plots")
    os.makedirs(outdir, exist_ok=True)

    print(f"[multiseed] agg_dir = {agg_dir}")
    print(f"[multiseed] outdir  = {outdir}")
    print(f"[multiseed] steps   = {steps_to_run}")

    entries = find_model_dirs(agg_dir)
    if not entries:
        raise RuntimeError(f"No model dirs found under: {agg_dir}")

    print(f"[multiseed] Found {len(entries)} model dirs: {[e['label'] for e in entries]}")

    # Build extra_args dict for forwarding to single-seed scripts
    extra_args = {}
    if args.tau_cap is not None:
        extra_args["tau_cap"] = args.tau_cap
    if args.min_r2 is not None:
        extra_args["min_r2"] = args.min_r2
    if args.min_beta_r2 is not None:
        extra_args["min_beta_r2"] = args.min_beta_r2
    if args.hist_bins is not None:
        extra_args["hist_bins"] = args.hist_bins
    if args.bins is not None:
        extra_args["bins"] = args.bins
    if args.debug is not None:
        extra_args["debug"] = args.debug
    if args.show_unreliable:
        extra_args["show_unreliable"] = True

    # Execute steps in order
    for step in steps_to_run:
        print(f"\n[multiseed] === Step: {step} ===")

        if step == "dynamics":
            plot_dynamics_with_se(entries, outdir, args.dpi)

        elif step == "learning_curves":
            plot_learning_curves_with_se(entries, outdir, args.dpi, ylog=(args.ylog == 1))

        elif step in STEP_TO_SCRIPT:
            # Delegate to single-seed script
            script_name = STEP_TO_SCRIPT[step]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, script_name)

            if not os.path.exists(script_path):
                print(f"[multiseed] [SKIP] Missing script: {script_path}")
                continue

            cmd = [sys.executable, script_path, "--indir", agg_dir, "--outdir", outdir, "--dpi", str(args.dpi)]

            # forward per-script args
            if step == "tau_spectrum":
                if "tau_cap" in extra_args:
                    cmd += ["--tau_cap", str(extra_args["tau_cap"])]
                if "min_beta_r2" in extra_args:
                    cmd += ["--min_beta_r2", str(extra_args["min_beta_r2"])]
                if "hist_bins" in extra_args:
                    cmd += ["--hist_bins", str(extra_args["hist_bins"])]
            elif step == "alpha_grad":
                if "bins" in extra_args:
                    cmd += ["--bins", str(extra_args["bins"])]
                if extra_args.get("show_unreliable"):
                    cmd += ["--show_unreliable"]
            elif step == "phase_summary":
                if "min_r2" in extra_args:
                    cmd += ["--min_r2", str(extra_args["min_r2"])]
            elif step == "envelopes":
                if "debug" in extra_args:
                    cmd += ["--debug", str(extra_args["debug"])]

            print(f"[multiseed] [RUN] {' '.join(cmd)}", flush=True)
            try:
                r = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if r.stdout.strip():
                    for line in r.stdout.strip().split("\n"):
                        print(f"  {line}")
                print(f"[multiseed] [OK] {script_name}")
            except subprocess.CalledProcessError as e:
                print(f"[multiseed] [FAIL] {script_name}: returncode={e.returncode}")
                if e.stderr:
                    for line in e.stderr.strip().split("\n")[:5]:
                        print(f"  {line}")

        else:
            print(f"[multiseed] Unknown step: {step}")

    print(f"\n[multiseed] All done. Plots in: {outdir}")


if __name__ == "__main__":
    main()
