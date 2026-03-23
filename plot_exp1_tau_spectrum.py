#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tau-spectrum plotting — improved version
=========================================
Key changes vs. previous version:
  - Log-scale on tau dynamics plots (tau_mean, tau_q99) to handle 1e12 outliers
  - Robust median-based statistics alongside mean for trajectory plots
  - tau histogram uses log-scale x-axis + optional cap for outlier filtering
  - Separate "valid neuron fraction" dynamics plot
  - CCDF plots unchanged (they were already log-log)
  - tau_cap parameter: neurons with tau >= tau_cap are treated as "non-decaying"
    and excluded from histogram / reported separately
"""

import os, argparse, json, csv
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def safe_read_csv_named(path: str):
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if data.size == 0:
            return None
        return data
    except Exception:
        return None

def read_ccdf_csv(path: str):
    data = safe_read_csv_named(path)
    if data is None:
        return None
    if data.shape == ():
        data = np.array([data])
    if ("tau" not in data.dtype.names) or ("ccdf" not in data.dtype.names):
        return None
    tau = np.array(data["tau"], dtype=float)
    ccdf = np.array(data["ccdf"], dtype=float)
    return tau, ccdf

def write_csv(path: str, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def load_phase_trajectory(traj_csv: str):
    """
    Loads phase_trajectory.csv and returns dict of arrays sorted by epoch.
    Requires at least 'epoch'. Other fields optional.
    """
    data = safe_read_csv_named(traj_csv)
    if data is None:
        return None

    rows = np.array([data]) if data.shape == () else data
    names = rows.dtype.names or ()
    if "epoch" not in names:
        return None

    ep = np.array(rows["epoch"], dtype=float)
    m = np.isfinite(ep)
    if not np.any(m):
        return None
    rows = rows[m]
    ep = ep[m]

    order = np.argsort(ep)
    rows = rows[order]
    ep = ep[order].astype(int)

    def col(name, default=np.nan):
        if name not in (rows.dtype.names or ()):
            return np.full(ep.shape, default, dtype=float)
        arr = np.array(rows[name], dtype=float)
        return arr

    return {
        "epoch": ep,
        "beta_hat": col("beta_hat"),
        "beta_r2": col("beta_r2"),
        "tau_mean": col("tau_mean"),
        "tau_q90": col("tau_q90"),
        "tau_q99": col("tau_q99"),
        "tau_fit_r2_mean": col("tau_fit_r2_mean"),
        "tau_fit_n_valid": col("tau_fit_n_valid"),
    }

def load_final_epoch_from_phase_trajectory(traj_csv: str):
    """
    Returns (epoch, row_dict) from LAST epoch in phase_trajectory.csv.
    """
    tr = load_phase_trajectory(traj_csv)
    if tr is None or tr["epoch"].size == 0:
        return None
    i = int(np.argmax(tr["epoch"]))
    epoch = int(tr["epoch"][i])
    row = {k: (float(tr[k][i]) if k != "epoch" else epoch) for k in tr.keys()}
    return epoch, row

# ------------------------------------------------------------
# Recursive discovery
# ------------------------------------------------------------

def find_model_dirs(indir: str):
    """
    Recursively find model output directories.

    Accept if any of:
      - phase_trajectory.csv (dynamic)
      - checkpoint_taus/ckpt_XXXX_taus.npy
      - checkpoint_tau_tail/ckpt_XXXX_tau_tail_fit.json
      - static <name>_taus.npy / <name>_tau_ccdf.csv / <name>_tau_tail_fit.json

    Skips seed_XXXX directories to avoid duplicate discovery when called
    from the optimizer-level directory instead of aggregated/.
    """
    out = []
    indir = os.path.abspath(indir)
    import re
    _SEED_RE = re.compile(r"^seed_\d+$")

    for root, dirs, files in os.walk(indir):
        folder = os.path.basename(root)

        # Skip seed directories to avoid duplicate model discovery
        dirs[:] = [d for d in dirs if not _SEED_RE.match(d)]

        has_dynamic = "phase_trajectory.csv" in files
        has_ckpt = (
            os.path.isdir(os.path.join(root, "checkpoint_taus")) or
            os.path.isdir(os.path.join(root, "checkpoint_tau_tail")) or
            os.path.isdir(os.path.join(root, "checkpoint_tau_ccdf"))
        )
        has_static = (
            f"{folder}_taus.npy" in files or
            f"{folder}_tau_ccdf.csv" in files or
            f"{folder}_tau_tail_fit.json" in files
        )

        if has_dynamic or has_ckpt or has_static:
            out.append({"dir": root, "name": folder, "label": folder})

    uniq = {d["dir"]: d for d in out}
    out = list(uniq.values())
    out.sort(key=lambda d: d["label"])
    return out

def pick_final_ckpt_tag(model_dir: str, model_name: str) -> Optional[str]:
    """
    Prefer final checkpoint epoch from phase_trajectory.csv.
    If absent, pick max ckpt_XXXX in checkpoint_taus folder.
    Returns ckpt_tag like 'ckpt_0500' or None.
    """
    traj = os.path.join(model_dir, "phase_trajectory.csv")
    if os.path.exists(traj):
        r = load_final_epoch_from_phase_trajectory(traj)
        if r is not None and r[0] is not None:
            return f"ckpt_{int(r[0]):04d}"

    # fallback: scan checkpoint_taus
    tdir = os.path.join(model_dir, "checkpoint_taus")
    if os.path.isdir(tdir):
        tags = []
        for fn in os.listdir(tdir):
            if fn.startswith("ckpt_") and fn.endswith("_taus.npy"):
                tag = fn.split("_taus.npy")[0]
                tags.append(tag)
        if tags:
            def key(t):
                try:
                    return int(t.split("_")[1])
                except Exception:
                    return -1
            tags.sort(key=key)
            return tags[-1]
    return None

def load_tau_array(model_dir: str, model_name: str, ckpt_tag: Optional[str]):
    """
    Prefer checkpoint taus. Fallback to static taus.
    """
    if ckpt_tag is not None:
        p = os.path.join(model_dir, "checkpoint_taus", f"{ckpt_tag}_taus.npy")
        if os.path.exists(p):
            return np.load(p).astype(float)

    p2 = os.path.join(model_dir, f"{model_name}_taus.npy")
    if os.path.exists(p2):
        return np.load(p2).astype(float)

    return None

def load_ccdf(model_dir: str, model_name: str, ckpt_tag: Optional[str]):
    """
    Prefer checkpoint ccdf if available. Fallback to static ccdf.
    """
    if ckpt_tag is not None:
        p = os.path.join(model_dir, "checkpoint_tau_ccdf", f"{ckpt_tag}_tau_ccdf.csv")
        if os.path.exists(p):
            return read_ccdf_csv(p)

    p2 = os.path.join(model_dir, f"{model_name}_tau_ccdf.csv")
    if os.path.exists(p2):
        return read_ccdf_csv(p2)

    return None

def load_tail_fit(model_dir: str, model_name: str, ckpt_tag: Optional[str]):
    """
    Prefer checkpoint tail fit. Fallback to static tail fit.
    """
    if ckpt_tag is not None:
        p = os.path.join(model_dir, "checkpoint_tau_tail", f"{ckpt_tag}_tau_tail_fit.json")
        if os.path.exists(p):
            return safe_load_json(p) or {}

    p2 = os.path.join(model_dir, f"{model_name}_tau_tail_fit.json")
    if os.path.exists(p2):
        return safe_load_json(p2) or {}

    return {}

# ------------------------------------------------------------
# Robust tau statistics (filters out clamped / non-decaying neurons)
# ------------------------------------------------------------

def robust_tau_stats(tau: np.ndarray, tau_cap: float):
    """
    Given a raw tau array, compute statistics on the 'valid' subset (tau < tau_cap).
    Returns dict with keys: median, mean, q90, q99, n_total, n_valid, frac_capped.
    """
    tau = np.asarray(tau, dtype=float)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    n_total = tau.size
    if n_total == 0:
        return {"median": np.nan, "mean": np.nan, "q90": np.nan, "q99": np.nan,
                "n_total": 0, "n_valid": 0, "frac_capped": 1.0}
    valid = tau[tau < tau_cap]
    n_valid = valid.size
    frac_capped = 1.0 - (n_valid / n_total) if n_total > 0 else 0.0
    if n_valid == 0:
        return {"median": np.nan, "mean": np.nan, "q90": np.nan, "q99": np.nan,
                "n_total": n_total, "n_valid": 0, "frac_capped": frac_capped}
    return {
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "q90": float(np.quantile(valid, 0.90)),
        "q99": float(np.quantile(valid, 0.99)),
        "n_total": n_total,
        "n_valid": n_valid,
        "frac_capped": frac_capped,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Experiment root folder (can contain baselines/ and lstmgru/)")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <indir>/plots_exp1)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--hist_bins", type=int, default=40)
    ap.add_argument("--min_beta_r2", type=float, default=-np.inf,
                    help="Optional: for dynamic beta-related plots, only show points with beta_r2 >= this.")
    ap.add_argument("--tau_cap", type=float, default=1e6,
                    help="Neurons with tau >= tau_cap are treated as non-decaying and excluded from "
                         "histograms. Default 1e6.")
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_exp1")
    os.makedirs(outdir, exist_ok=True)

    entries = find_model_dirs(indir)
    if not entries:
        raise RuntimeError("No models with tau artifacts found (dynamic or static).")

    tau_cap = float(args.tau_cap)

    # ============================================================
    # DYNAMICS FROM phase_trajectory.csv (preferred)
    # ============================================================
    traj_by_label = {}
    for e in entries:
        model_dir, label = e["dir"], e["label"]
        traj_p = os.path.join(model_dir, "phase_trajectory.csv")
        if os.path.exists(traj_p):
            tr = load_phase_trajectory(traj_p)
            if tr is not None and tr["epoch"].size > 0:
                traj_by_label[label] = tr

    if traj_by_label:
        # ---- tau_q99 vs epoch (LOG SCALE) ----
        plt.figure(figsize=(7.6, 4.8))
        any_data = False
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            ep = tr["epoch"]
            y = tr["tau_q99"]
            m = np.isfinite(ep) & np.isfinite(y) & (y > 0)
            if np.any(m):
                any_data = True
                plt.plot(ep[m], y[m], linewidth=2.0, label=label)
        if any_data:
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel(r"$\tau_{0.99}(t)$")
            plt.title(r"Time-scale spectrum dynamics: $\tau_{0.99}(t)$")
            plt.grid(True, alpha=0.25, which="both")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "tau_dynamics_tau_q99_vs_epoch.png"),
                        dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # ---- tau_mean vs epoch (LOG SCALE) ----
        plt.figure(figsize=(7.6, 4.8))
        any_data = False
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            ep = tr["epoch"]
            y = tr["tau_mean"]
            m = np.isfinite(ep) & np.isfinite(y) & (y > 0)
            if np.any(m):
                any_data = True
                plt.plot(ep[m], y[m], linewidth=2.0, label=label)
        if any_data:
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel(r"$\mathbb{E}[\tau](t)$")
            plt.title(r"Time-scale spectrum dynamics: mean $\tau$ (log scale)")
            plt.grid(True, alpha=0.25, which="both")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "tau_dynamics_tau_mean_vs_epoch.png"),
                        dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # ---- tau_fit_r2_mean vs epoch (slope fit quality) ----
        plt.figure(figsize=(7.6, 4.8))
        any_data = False
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            ep = tr["epoch"]
            r2 = tr["tau_fit_r2_mean"]
            m = np.isfinite(ep) & np.isfinite(r2)
            if np.any(m):
                any_data = True
                plt.plot(ep[m], r2[m], linewidth=2.0, label=label)
        if any_data:
            plt.xlabel("epoch")
            plt.ylabel(r"mean $R^2$ (slope fits)")
            plt.ylim(-0.05, 1.05)
            plt.title(r"Per-neuron slope-fit quality over time (mean $R^2$)")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "tau_dynamics_slope_r2_vs_epoch.png"),
                        dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # ---- beta_r2 vs epoch (tail-fit reliability over time) ----
        plt.figure(figsize=(7.6, 4.8))
        any_data = False
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            ep = tr["epoch"]
            r2 = tr["beta_r2"]
            m = np.isfinite(ep) & np.isfinite(r2)
            if np.any(m):
                any_data = True
                plt.plot(ep[m], r2[m], linewidth=2.0, label=label)
        if any_data:
            plt.xlabel("epoch")
            plt.ylabel(r"$R^2$")
            plt.ylim(0.0, 1.01)
            plt.title(r"CCDF tail-fit quality over time ($\hat{\beta}$ regression)")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "tau_dynamics_beta_r2_vs_epoch.png"),
                        dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # Save a long-form CSV (handy for external plotting)
        rows = []
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            for i in range(tr["epoch"].size):
                rows.append([
                    label,
                    int(tr["epoch"][i]),
                    float(tr["tau_mean"][i]),
                    float(tr["tau_q90"][i]),
                    float(tr["tau_q99"][i]),
                    float(tr["beta_hat"][i]),
                    float(tr["beta_r2"][i]),
                    float(tr["tau_fit_r2_mean"][i]),
                    int(tr["tau_fit_n_valid"][i]) if np.isfinite(tr["tau_fit_n_valid"][i]) else 0,
                ])
        write_csv(
            os.path.join(outdir, "tau_dynamics_long.csv"),
            ["label", "epoch", "tau_mean", "tau_q90", "tau_q99",
             "beta_hat", "beta_r2", "tau_fit_r2_mean", "tau_fit_n_valid"],
            rows
        )

    # ============================================================
    # FINAL CHECKPOINT PLOTS
    # ============================================================

    # ---- Histogram overlay (log-scale x-axis, capped) ----
    plt.figure(figsize=(7.4, 4.8))
    any_hist = False
    summary_rows = []

    for e in entries:
        model_dir, model_name, label = e["dir"], e["name"], e["label"]
        ckpt_tag = pick_final_ckpt_tag(model_dir, model_name)
        tau_raw = load_tau_array(model_dir, model_name, ckpt_tag)
        if tau_raw is None:
            continue
        tau_raw = np.asarray(tau_raw, dtype=float)
        stats = robust_tau_stats(tau_raw, tau_cap)
        summary_rows.append([label, model_name, ckpt_tag or "",
                             stats["median"], stats["mean"], stats["q90"], stats["q99"],
                             stats["n_total"], stats["n_valid"], stats["frac_capped"]])

        # filter for histogram: only valid (non-capped) neurons
        tau = tau_raw[np.isfinite(tau_raw) & (tau_raw > 0) & (tau_raw < tau_cap)]
        if tau.size < 3:
            continue
        any_hist = True
        log_tau = np.log10(tau)
        bins = min(int(args.hist_bins), max(10, int(tau.size // 3)))
        plt.hist(log_tau, bins=bins, density=True, alpha=0.3, label=f"{label} (n={tau.size})")

    if any_hist:
        plt.xlabel(r"$\log_{10}\,\tau$")
        plt.ylabel("density")
        plt.title(rf"Time-scale spectrum (final ckpt, $\tau < {tau_cap:.0e}$)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tau_hist.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # Save robust summary CSV
    if summary_rows:
        write_csv(
            os.path.join(outdir, "tau_robust_summary.csv"),
            ["label", "model", "ckpt_tag",
             "tau_median_valid", "tau_mean_valid", "tau_q90_valid", "tau_q99_valid",
             "n_total", "n_valid", "frac_capped"],
            summary_rows
        )

    # ---- CCDF log-log overlay (final, uses ALL taus including capped) ----
    plt.figure(figsize=(7.0, 4.6))
    any_ccdf = False
    for e in entries:
        model_dir, model_name, label = e["dir"], e["name"], e["label"]
        ckpt_tag = pick_final_ckpt_tag(model_dir, model_name)
        cc = load_ccdf(model_dir, model_name, ckpt_tag)
        if cc is None:
            continue
        tau, ccdf = cc
        mask = np.isfinite(tau) & np.isfinite(ccdf) & (tau > 0) & (ccdf > 0)
        if np.any(mask):
            any_ccdf = True
            plt.plot(np.log(tau[mask]), np.log(ccdf[mask]), linewidth=1.5, label=label)

    if any_ccdf:
        plt.xlabel(r"$\log \tau$")
        plt.ylabel(r"$\log \Pr(\tau > x)$")
        plt.title("CCDF of time scales (log-log, final checkpoint)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tau_ccdf.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # ---- CCDF fit window highlight + export summary (final) ----
    rows = []
    plt.figure(figsize=(7.0, 4.6))
    any_ccdf2 = False

    for e in entries:
        model_dir, model_name, label = e["dir"], e["name"], e["label"]
        ckpt_tag = pick_final_ckpt_tag(model_dir, model_name)

        fit = load_tail_fit(model_dir, model_name, ckpt_tag) or {}
        beta = float(fit.get("beta_hat", np.nan))
        r2 = float(fit.get("beta_r2", np.nan))
        x_lo = float(fit.get("x_lo", np.nan))
        x_hi = float(fit.get("x_hi", np.nan))
        n_fit = int(fit.get("n_fit", 0)) if fit.get("n_fit", None) is not None else 0

        rows.append([label, model_name, ckpt_tag or "", beta, r2, x_lo, x_hi, n_fit])

        cc = load_ccdf(model_dir, model_name, ckpt_tag)
        if cc is None:
            continue
        tau, ccdf = cc
        mask = np.isfinite(tau) & np.isfinite(ccdf) & (tau > 0) & (ccdf > 0)
        if not np.any(mask):
            continue

        any_ccdf2 = True
        plt.plot(np.log(tau[mask]), np.log(ccdf[mask]), linewidth=1.5, label=label)

        if np.isfinite(x_lo) and np.isfinite(x_hi) and (x_hi > x_lo):
            wmask = mask & (tau >= x_lo) & (tau <= x_hi)
            if np.any(wmask):
                plt.plot(np.log(tau[wmask]), np.log(ccdf[wmask]), linewidth=3.0)

    if any_ccdf2:
        plt.xlabel(r"$\log \tau$")
        plt.ylabel(r"$\log \Pr(\tau > x)$")
        plt.title("CCDF tail fit window highlight (thick segment, final checkpoint)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tau_ccdf_fit_window.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    if rows:
        write_csv(
            os.path.join(outdir, "tau_tail_fit_summary.csv"),
            ["label", "model", "ckpt_tag", "beta_hat", "beta_r2", "x_lo", "x_hi", "n_fit"],
            rows
        )

    print(f"[OK] Saved tau-spectrum plots (dynamics + final) to: {outdir}")

if __name__ == "__main__":
    main()
