#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic 2: Large-width population concentration
===================================================

Tests whether the empirical spectrum of log-effective learning rates
and the intensive envelope f(ℓ) = (1/H)||μ||_1 converge as H → ∞.

Specifically validates:
  1. Spectral convergence: the empirical distribution of τ_q stabilizes
     as H grows (Wasserstein distance decreases).
  2. Envelope collapse: the intensive envelope becomes width-independent
     (pairwise Pearson r between widths → 1).
  3. Variance scaling: cross-seed variance of f(ℓ) decreases as ~1/H.

Outputs:
  results_width/
    <arch>_H<width>_seed<N>/
      tau_spectrum.npy        — (H,) array of tau values
      envelope.csv            — ℓ, f_actual
      metrics.json            — summary statistics
    convergence_<arch>.csv    — width, wasserstein_to_largest, envelope_r2, variance_f
    summary.csv               — all runs

Usage:
  python run_width_scaling.py [--outdir results_width] [--device auto]
                              [--epochs 200] [--T 300]
"""

import argparse, os, sys, time
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_utils import (
    set_seed, resolve_device, make_dataset, build_model, build_optimizer,
    train_model, save_json, save_csv, envelope_correlation,
    save_run_manifest, load_main_diagnostics_module,
)

import torch
from scipy import stats


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def wasserstein_1d(a, b):
    """1D Wasserstein (earth-mover) distance between two empirical distributions."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(stats.wasserstein_distance(a, b))


def ks_distance(a, b):
    """Kolmogorov-Smirnov statistic between two samples."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, pval = stats.ks_2samp(a, b)
    return float(stat), float(pval)


def run_single(arch, H, seed, args, device):
    """Run one (architecture, width, seed) configuration."""
    tag = f"{arch}_H{H}_seed{seed}"
    rundir = os.path.join(args.outdir, tag)
    os.makedirs(rundir, exist_ok=True)

    log(f"--- {tag} ---")
    set_seed(seed)

    # Fixed data seed across widths (so we compare the same task)
    X, Y, u = make_dataset(
        Nseq=args.Nseq, T=args.T, D=args.D,
        task_lags=[10, 50],
        task_coeffs=[1.0, 0.5],
        noise_std=0.1, seed=seed + 20000)

    model = build_model(arch, args.D, H, const_s=args.const_s)
    model.apply_orthogonal()
    optimizer = build_optimizer(model, args.optimizer, lr=args.lr)

    t0 = time.time()
    losses = train_model(model, optimizer, X, Y, device,
                         epochs=args.epochs, batch_size=args.batch_size,
                         verbose=args.verbose)
    train_time = time.time() - t0
    log(f"  trained in {train_time:.1f}s, final loss={losses[-1]:.6f}")

    diag_device = torch.device("cpu")
    model.to(diag_device)

    # Lag grid
    max_lag = min(args.T // 2, args.max_lag)
    lags = np.unique(np.concatenate([
        np.arange(1, min(17, max_lag)),
        np.geomspace(16, max_lag, 20).astype(int),
    ]))
    lags = lags[lags <= max_lag]
    lags = np.sort(np.unique(lags))
    fit_lags = lags[lags >= args.fit_lag_min]
    if fit_lags.size < 4:
        fit_lags = lags
    main_diag = load_main_diagnostics_module()

    # Main-pipeline transport diagnostics
    log(f"  computing transport diagnostics (H={H}, {len(lags)} lags)...")
    tau, tau_info = main_diag.estimate_tau_spectrum(
        model_name=arch,
        model=model,
        Xdg_cpu=X,
        device=diag_device,
        diag_batch_size=args.batch_size,
        fit_lags=fit_lags,
    )
    n_valid = int(tau_info["n_valid_neurons"])
    f_actual, _ = main_diag.compute_macro_envelope(
        model_name=arch,
        model=model,
        Xdg_cpu=X,
        device=diag_device,
        ells=lags,
        diag_batch_size=args.batch_size,
    )

    # Log-tau for distributional analysis
    log_tau = np.log(tau[np.isfinite(tau) & (tau > 0)])

    save_run_manifest(
        os.path.join(rundir, "config.json"),
        args,
        arch=arch,
        H=H,
        seed=seed,
        device=str(device),
        diag_device=str(diag_device),
        lags=lags.tolist(),
        fit_lags=fit_lags.tolist(),
    )

    # Save
    np.save(os.path.join(rundir, "tau_spectrum.npy"), tau)
    np.save(os.path.join(rundir, "log_tau.npy"), log_tau)

    header_e = ["lag", "f_actual"]
    rows_e = [[int(ell), float(f_actual[j])] for j, ell in enumerate(lags)]
    save_csv(header_e, rows_e, os.path.join(rundir, "envelope.csv"))

    metrics = {
        "arch": arch, "H": H, "seed": seed,
        "optimizer": args.optimizer, "lr": args.lr,
        "epochs": args.epochs, "T": args.T,
        "final_loss": float(losses[-1]),
        "train_time_s": float(train_time),
        "n_valid_neurons": int(n_valid),
        "tau_median": float(np.nanmedian(tau)),
        "tau_mean": float(np.nanmean(tau)),
        "log_tau_std": float(np.std(log_tau)) if len(log_tau) > 0 else np.nan,
        "r2_mean": float(tau_info["r2_mean"]),
        "r2_q10": float(tau_info["r2_q10"]),
        "r2_median": float(tau_info["r2_q50"]),
    }
    save_json(metrics, os.path.join(rundir, "metrics.json"))

    return {
        "tag": tag, "arch": arch, "H": H, "seed": seed,
        "tau": tau, "log_tau": log_tau, "f_actual": f_actual,
        "lags": lags, "metrics": metrics,
    }


def analyze_convergence(results_by_arch, args):
    """
    For each architecture, analyze width convergence:
    1. Wasserstein distance of log-tau distributions to the largest width
    2. Envelope pairwise Pearson r
    3. Variance scaling of f(ℓ) across seeds
    """
    for arch, runs in results_by_arch.items():
        log(f"\n=== Convergence analysis: {arch} ===")

        widths = sorted(set(r["H"] for r in runs))
        seeds = sorted(set(r["seed"] for r in runs))
        max_H = max(widths)

        # Group by width
        by_width = {H: [r for r in runs if r["H"] == H] for H in widths}

        # Reference: largest width, pool all seeds' log-tau
        ref_runs = by_width[max_H]
        ref_log_tau = np.concatenate([r["log_tau"] for r in ref_runs])

        convergence_rows = []
        header = [
            "width", "n_seeds",
            "wasserstein_to_ref", "ks_stat_to_ref", "ks_pval_to_ref",
            "envelope_pearson_to_ref",
            "f_variance_mean", "f_variance_per_lag",
            "tau_median_mean", "tau_median_std",
        ]

        for H in widths:
            h_runs = by_width[H]
            n_s = len(h_runs)

            # Pool log-tau across seeds for this width
            pooled_log_tau = np.concatenate([r["log_tau"] for r in h_runs])

            # Wasserstein to reference
            wd = wasserstein_1d(pooled_log_tau, ref_log_tau) if H != max_H else 0.0
            ks_stat, ks_pval = ks_distance(pooled_log_tau, ref_log_tau) if H != max_H else (0.0, 1.0)

            # Envelope: mean across seeds, then Pearson with reference
            lags = h_runs[0]["lags"]
            # Find common lag grid (should be same, but be safe)
            f_matrix = np.array([r["f_actual"] for r in h_runs])  # (n_seeds, n_lags)
            f_mean = f_matrix.mean(axis=0)
            f_var = f_matrix.var(axis=0, ddof=1) if n_s > 1 else np.zeros_like(f_mean)

            ref_f_matrix = np.array([r["f_actual"] for r in ref_runs])
            ref_f_mean = ref_f_matrix.mean(axis=0)

            # Use common lags (truncate to shorter if needed)
            n_common = min(len(f_mean), len(ref_f_mean))
            corr = envelope_correlation(f_mean[:n_common], ref_f_mean[:n_common])

            # Tau statistics across seeds
            tau_medians = [np.nanmedian(r["tau"]) for r in h_runs]

            row = [
                H, n_s,
                f"{wd:.6f}",
                f"{ks_stat:.6f}", f"{ks_pval:.4f}",
                f"{corr['pearson_r']:.6f}",
                f"{np.mean(f_var):.2e}",
                ";".join(f"{v:.2e}" for v in f_var[:8]),  # first 8 lags
                f"{np.mean(tau_medians):.4f}",
                f"{np.std(tau_medians):.4f}" if len(tau_medians) > 1 else "n/a",
            ]
            convergence_rows.append(row)

            log(f"  H={H:>4d}  W1={wd:.4f}  KS={ks_stat:.4f}  "
                f"env_r={corr['pearson_r']:.4f}  var(f)={np.mean(f_var):.2e}  "
                f"τ_med={np.mean(tau_medians):.2f}±{np.std(tau_medians):.2f}")

        save_csv(header, convergence_rows,
                 os.path.join(args.outdir, f"convergence_{arch}.csv"))

        # --- Variance scaling analysis ---
        # Check if var(f) ~ 1/H
        if len(widths) >= 3:
            log_H = np.log([float(H) for H in widths])
            mean_vars = []
            for H in widths:
                h_runs_h = by_width[H]
                if len(h_runs_h) > 1:
                    f_mat = np.array([r["f_actual"] for r in h_runs_h])
                    mean_vars.append(np.mean(np.var(f_mat, axis=0, ddof=1)))
                else:
                    mean_vars.append(np.nan)

            valid_v = np.array([(not np.isnan(v)) and v > 0 for v in mean_vars])
            if valid_v.sum() >= 3:
                log_v = np.log(np.array(mean_vars)[valid_v])
                log_h = log_H[valid_v]
                A = np.vstack([np.ones_like(log_h), log_h]).T
                coeff, _, _, _ = np.linalg.lstsq(A, log_v, rcond=None)
                slope = float(coeff[1])
                log(f"  Variance scaling: log(var) ~ {slope:.2f} * log(H)  "
                    f"(expect ≈ -1.0 for 1/H scaling)")
                save_json({"arch": arch, "slope": slope,
                           "widths": [int(h) for h in widths],
                           "mean_vars": mean_vars},
                          os.path.join(args.outdir, f"variance_scaling_{arch}.json"))


def main():
    parser = argparse.ArgumentParser(description="Large-width population concentration diagnostic")
    parser.add_argument("--outdir", default="results_width")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--Nseq", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", default="sgd",
                        help="Optimizer (sgd recommended for clean width scaling)")
    parser.add_argument("--const_s", type=float, default=0.005)
    parser.add_argument("--max_lag", type=int, default=140)
    parser.add_argument("--fit_lag_min", type=int, default=16)
    parser.add_argument("--widths", default="16,32,64,128,256,512",
                        help="Comma-separated hidden sizes")
    parser.add_argument("--seeds", default="42,123,321,456,789",
                        help="Comma-separated seeds (≥3 for variance analysis)")
    parser.add_argument("--archs", default="diag,gru",
                        help="Comma-separated architectures")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"Device: {device}")

    seeds = [int(s) for s in args.seeds.split(",")]
    widths = [int(w) for w in args.widths.split(",")]
    archs = [a.strip() for a in args.archs.split(",")]

    all_results = []
    results_by_arch = {a: [] for a in archs}
    t_total = time.time()

    for arch in archs:
        for H in widths:
            for seed in seeds:
                result = run_single(arch, H, seed, args, device)
                all_results.append(result)
                results_by_arch[arch].append(result)

    # --- Summary CSV ---
    header = ["arch", "H", "seed", "final_loss", "n_valid_neurons",
              "tau_median", "r2_median", "train_time_s"]
    rows = []
    for r in all_results:
        m = r["metrics"]
        rows.append([
            m["arch"], m["H"], m["seed"],
            f"{m['final_loss']:.6f}", m["n_valid_neurons"],
            f"{m['tau_median']:.4f}", f"{m['r2_median']:.4f}",
            f"{m['train_time_s']:.1f}",
        ])
    save_csv(header, rows, os.path.join(args.outdir, "summary.csv"))

    # --- Convergence analysis ---
    analyze_convergence(results_by_arch, args)

    elapsed = time.time() - t_total
    log(f"\nDone. {len(all_results)} runs in {elapsed:.0f}s. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
