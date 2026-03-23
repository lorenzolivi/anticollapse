#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic 1: Mixture-of-exponentials validation
=================================================

Tests whether the mixture representation
    f_mix(ℓ) = (1/H) Σ_q exp(-ℓ/τ_q)
faithfully reproduces the transport envelope
    f_transport(ℓ) = (1/H) Σ_q |μ^(q)_{t,ℓ}|
computed from the first-order diagonal expansion.

Under GELR (adaptive optimizers), additionally tests the lag-dependent
Rayleigh-weighted mixture against the actual GELR envelope:
    f_GELR-mix(ℓ) = E_{seq,t} (1/H) Σ_q Λ^(q)_{t,ℓ} exp(-ℓ/τ_q)

Outputs:
  results_mixture/
    <arch>_<opt>_seed<N>/
      envelope.csv          — ℓ, transport/GELR actual and mixture curves
      per_neuron.csv        — q, tau_q, r2, mu_bar, [lambda_rowmean_q]
      neuron_traces.csv     — selected neurons: q, ℓ, log_abs_mu
      metrics.json          — correlations, robust errors, GELR metadata
      config.json           — exact run configuration and lag grid
    summary.csv             — one row per run

Usage:
  python run_mixture.py [--outdir results_mixture] [--device auto]
                        [--epochs 200] [--H 64] [--T 300]
"""

import argparse, os, sys, time
from datetime import datetime
import numpy as np

# Ensure parent dir is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_utils import (
    set_seed, resolve_device, make_dataset, build_model, build_optimizer,
    train_model, compute_per_neuron_envelope, extract_tau_spectrum,
    build_mixture_envelope, compute_gelr_mixture_envelope,
    envelope_correlation, log_rmse, relative_l2_error,
    mean_relative_error_supported, save_json, save_csv, save_run_manifest,
    load_main_diagnostics_module,
)

import torch


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_single(arch, opt_name, seed, args, device):
    """Run one (architecture, optimizer, seed) configuration."""
    tag = f"{arch}_{opt_name}_seed{seed}"
    rundir = os.path.join(args.outdir, tag)
    os.makedirs(rundir, exist_ok=True)

    log(f"--- {tag} ---")
    set_seed(seed)

    # Dataset (use separate seed for data so all runs see different data)
    X, Y, u = make_dataset(
        Nseq=args.Nseq, T=args.T, D=args.D,
        task_lags=[10, 50, 100],
        task_coeffs=[1.0, 0.5, 0.3],
        noise_std=0.1, seed=seed + 10000)

    # Model & optimizer
    model = build_model(arch, args.D, args.H, const_s=args.const_s)
    model.apply_orthogonal()
    optimizer = build_optimizer(model, opt_name, lr=args.lr)

    # Train
    t0 = time.time()
    losses = train_model(model, optimizer, X, Y, device,
                         epochs=args.epochs, batch_size=args.batch_size,
                         verbose=args.verbose)
    train_time = time.time() - t0
    log(f"  trained in {train_time:.1f}s, final loss={losses[-1]:.6f}")

    # Move model to CPU for diagnostics (safe for MPS/CUDA)
    diag_device = torch.device("cpu")
    model.to(diag_device)

    # --- Lag grid ---
    max_lag = min(args.T // 2, args.max_lag)
    lags = np.unique(np.concatenate([
        np.arange(1, min(17, max_lag)),              # fine grid at short lags
        np.geomspace(16, max_lag, 20).astype(int),   # log-spaced medium-long
    ]))
    lags = lags[lags <= max_lag]
    lags = np.sort(np.unique(lags))
    fit_lags = lags[lags >= args.fit_lag_min]
    if fit_lags.size < 4:
        fit_lags = lags
    main_diag = load_main_diagnostics_module()

    # --- Per-neuron envelope f_q(ℓ) ---
    log(f"  computing per-neuron envelope at {len(lags)} lags...")
    f_q = compute_per_neuron_envelope(model, X, diag_device, lags, batch_size=args.batch_size)
    # f_q: (n_lags, H)

    # --- Extract tau spectrum ---
    tau, r2, mu_bar = extract_tau_spectrum(f_q, lags, fit_lag_min=args.fit_lag_min)
    valid_tau = np.isfinite(tau) & (tau > 0) & np.isfinite(r2)
    n_valid = valid_tau.sum()
    log(f"  tau extraction: {n_valid}/{args.H} neurons valid, "
        f"median tau={np.nanmedian(tau):.2f}, median r2={np.nanmedian(r2):.4f}")

    env_compare = main_diag.compute_macro_envelope_comparison(
        model_name=arch,
        model=model,
        optimizer=optimizer,
        Xdg_cpu=X,
        device=diag_device,
        ells=lags,
        diag_batch_size=args.batch_size,
    )
    f_transport_actual = env_compare["f_transport"]
    f_gelr_actual = env_compare["f_gelr"]

    # --- Mixture envelope (transport) ---
    f_mix = build_mixture_envelope(tau, lags, weights=None)

    # --- Metrics: transport actual vs mixture ---
    corr_mix = envelope_correlation(f_transport_actual, f_mix)
    mix_log_rmse = log_rmse(f_transport_actual, f_mix)
    mix_rel_l2 = relative_l2_error(f_transport_actual, f_mix)
    mix_rel_supported = mean_relative_error_supported(f_transport_actual, f_mix)

    gelr_mix = compute_gelr_mixture_envelope(
        model=model,
        optimizer=optimizer,
        X=X,
        device=diag_device,
        lags=lags,
        tau=tau,
        batch_size=args.batch_size,
    )
    f_gelr_mix = gelr_mix["f_gelr_mix"]
    corr_gelr = envelope_correlation(f_gelr_actual, f_gelr_mix)
    gelr_log_rmse = log_rmse(f_gelr_actual, f_gelr_mix)
    gelr_rel_l2 = relative_l2_error(f_gelr_actual, f_gelr_mix)
    gelr_rel_supported = mean_relative_error_supported(f_gelr_actual, f_gelr_mix)

    lambda_rowmean = gelr_mix["lambda_rowmean"]
    lambda_ratio = float(
        np.max(lambda_rowmean) / max(float(np.min(lambda_rowmean)), 1e-30)
    )
    log(f"  transport mixture vs actual: Spearman={corr_mix['spearman_rho']:.4f}, "
        f"Pearson(log)={corr_mix['pearson_r']:.4f}, logRMSE={mix_log_rmse:.4f}")
    log(f"  GELR mixture vs actual: Spearman={corr_gelr['spearman_rho']:.4f}, "
        f"Pearson(log)={corr_gelr['pearson_r']:.4f}, logRMSE={gelr_log_rmse:.4f}")
    log(f"  GELR mode={gelr_mix['gelr_mode']} lag-dependent={gelr_mix['used_lag_dependent_rates']} "
        f"Lambda_rowmean range [{lambda_rowmean.min():.6f}, {lambda_rowmean.max():.6f}]")

    save_run_manifest(
        os.path.join(rundir, "config.json"),
        args,
        arch=arch,
        optimizer=opt_name,
        seed=seed,
        device=str(device),
        diag_device=str(diag_device),
        lags=lags.tolist(),
        fit_lags=fit_lags.tolist(),
        gelr_mode=gelr_mix["gelr_mode"],
        used_lag_dependent_rates=gelr_mix["used_lag_dependent_rates"],
        recurrent_matrices=gelr_mix["recurrent_matrices"],
    )

    # --- Save envelope.csv ---
    header = [
        "lag",
        "f_transport_actual",
        "f_mix",
        "f_gelr_actual",
        "f_gelr_mix",
        "lambda_mean",
        "lambda_std",
    ]
    rows = []
    for j, ell in enumerate(lags):
        rows.append([
            int(ell),
            float(f_transport_actual[j]),
            float(f_mix[j]),
            float(f_gelr_actual[j]),
            float(f_gelr_mix[j]),
            float(env_compare["lambda_mean"][j]),
            float(env_compare["lambda_std"][j]),
        ])
    save_csv(header, rows, os.path.join(rundir, "envelope.csv"))

    # --- Save per_neuron.csv ---
    header_n = ["q", "tau", "r2", "mu_bar"]
    if lambda_rowmean is not None:
        header_n.append("lambda_rowmean")
    rows_n = []
    for q in range(args.H):
        row = [q, float(tau[q]), float(r2[q]), float(mu_bar[q])]
        if lambda_rowmean is not None:
            row.append(float(lambda_rowmean[q]))
        rows_n.append(row)
    save_csv(header_n, rows_n, os.path.join(rundir, "per_neuron.csv"))

    # --- Save neuron traces (5 representative neurons) ---
    # Pick neurons at quantiles of tau distribution
    valid_idx = np.where(valid_tau)[0]
    if len(valid_idx) >= 5:
        quantile_idx = np.quantile(np.arange(len(valid_idx)), [0.1, 0.3, 0.5, 0.7, 0.9]).astype(int)
        sorted_by_tau = valid_idx[np.argsort(tau[valid_idx])]
        representative = sorted_by_tau[quantile_idx]
    else:
        representative = valid_idx[:5]

    header_t = ["q", "lag", "log_abs_mu", "log_exp_fit"]
    rows_t = []
    for q in representative:
        for j, ell in enumerate(lags):
            log_mu = float(np.log(f_q[j, q] + 1e-30))
            log_exp = float(-ell / tau[q]) if np.isfinite(tau[q]) and tau[q] > 0 else np.nan
            rows_t.append([int(q), int(ell), log_mu, log_exp])
    save_csv(header_t, rows_t, os.path.join(rundir, "neuron_traces.csv"))

    # --- Save metrics.json ---
    metrics = {
        "arch": arch, "optimizer": opt_name, "seed": seed,
        "H": args.H, "T": args.T, "epochs": args.epochs, "lr": args.lr,
        "final_loss": float(losses[-1]),
        "train_time_s": float(train_time),
        "n_valid_neurons": int(n_valid),
        "tau_median": float(np.nanmedian(tau)),
        "tau_mean": float(np.nanmean(tau)),
        "tau_q90": float(np.nanquantile(tau, 0.90)) if n_valid > 0 else np.nan,
        "r2_median": float(np.nanmedian(r2)),
        "r2_q10": float(np.nanquantile(r2, 0.10)) if n_valid > 0 else np.nan,
        "mix_vs_transport": corr_mix,
        "mix_log_rmse": mix_log_rmse,
        "mix_rel_l2": mix_rel_l2,
        "mix_rel_err_mean_supported": mix_rel_supported,
        "gelr_mix_vs_gelr": corr_gelr,
        "gelr_log_rmse": gelr_log_rmse,
        "gelr_rel_l2": gelr_rel_l2,
        "gelr_rel_err_mean_supported": gelr_rel_supported,
        "gelr_mode": gelr_mix["gelr_mode"],
        "used_lag_dependent_rates": gelr_mix["used_lag_dependent_rates"],
        "recurrent_matrices": gelr_mix["recurrent_matrices"],
        "lambda_rowmean_mean": float(lambda_rowmean.mean()),
        "lambda_rowmean_std": float(lambda_rowmean.std()),
        "lambda_rowmean_min": float(lambda_rowmean.min()),
        "lambda_rowmean_max": float(lambda_rowmean.max()),
        "lambda_rowmean_ratio_max_min": lambda_ratio,
    }

    save_json(metrics, os.path.join(rundir, "metrics.json"))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Mixture-of-exponentials validation diagnostic")
    parser.add_argument("--outdir", default="results_mixture")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--Nseq", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--const_s", type=float, default=0.005,
                        help="Init gate value for ConstGate/DiagGate")
    parser.add_argument("--max_lag", type=int, default=140)
    parser.add_argument("--fit_lag_min", type=int, default=16,
                        help="Minimum lag for tau slope fitting")
    parser.add_argument("--seeds", default="42,123,321,456,789",
                        help="Comma-separated seeds")
    parser.add_argument("--archs", default="diag,gru,lstm",
                        help="Comma-separated architectures")
    parser.add_argument("--optimizers", default="sgd,adam,rmsprop",
                        help="Comma-separated optimizers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"Device: {device}")

    seeds = [int(s) for s in args.seeds.split(",")]
    archs = [a.strip() for a in args.archs.split(",")]
    optimizers = [o.strip() for o in args.optimizers.split(",")]

    all_metrics = []
    t_total = time.time()

    for arch in archs:
        for opt_name in optimizers:
            for seed in seeds:
                metrics = run_single(arch, opt_name, seed, args, device)
                all_metrics.append(metrics)

    # --- Summary CSV ---
    header = [
        "arch", "optimizer", "seed",
        "final_loss", "n_valid_neurons",
        "tau_median", "r2_median",
        "mix_spearman", "mix_pearson", "mix_log_rmse", "mix_rel_l2", "mix_rel_err_mean_supported",
        "gelr_mix_spearman", "gelr_mix_pearson", "gelr_log_rmse", "gelr_rel_l2", "gelr_rel_err_mean_supported",
        "gelr_mode", "lambda_rowmean_ratio_max_min",
    ]
    rows = []
    for m in all_metrics:
        row = [
            m["arch"], m["optimizer"], m["seed"],
            f"{m['final_loss']:.6f}",
            m["n_valid_neurons"],
            f"{m['tau_median']:.4f}",
            f"{m['r2_median']:.4f}",
            f"{m['mix_vs_transport']['spearman_rho']:.4f}",
            f"{m['mix_vs_transport']['pearson_r']:.4f}",
            f"{m['mix_log_rmse']:.4f}",
            f"{m['mix_rel_l2']:.4f}",
            f"{m['mix_rel_err_mean_supported']:.4f}",
            f"{m['gelr_mix_vs_gelr']['spearman_rho']:.4f}",
            f"{m['gelr_mix_vs_gelr']['pearson_r']:.4f}",
            f"{m['gelr_log_rmse']:.4f}",
            f"{m['gelr_rel_l2']:.4f}",
            f"{m['gelr_rel_err_mean_supported']:.4f}",
            m["gelr_mode"],
            f"{m['lambda_rowmean_ratio_max_min']:.4f}",
        ]
        rows.append(row)

    save_csv(header, rows, os.path.join(args.outdir, "summary.csv"))

    elapsed = time.time() - t_total
    log(f"\nDone. {len(all_metrics)} runs in {elapsed:.0f}s. Results in {args.outdir}/")

    # --- Print summary table ---
    print("\n" + "=" * 90)
    print(f"{'Arch':<6} {'Opt':<8} {'Seed':<6} "
          f"{'ρ_mix':>7} {'r_mix':>7} "
          f"{'ρ_gelr':>7} {'r_gelr':>7} "
          f"{'Λ_ratio':>8} {'τ_med':>8}")
    print("-" * 90)
    for m in all_metrics:
        rho_m = m["mix_vs_transport"]["spearman_rho"]
        r_m = m["mix_vs_transport"]["pearson_r"]
        rho_l = f"{m['gelr_mix_vs_gelr']['spearman_rho']:>7.4f}"
        r_l = f"{m['gelr_mix_vs_gelr']['pearson_r']:>7.4f}"
        lam_r = f"{m['lambda_rowmean_ratio_max_min']:>8.2f}"
        print(f"{m['arch']:<6} {m['optimizer']:<8} {m['seed']:<6} "
              f"{rho_m:>7.4f} {r_m:>7.4f} "
              f"{rho_l} {r_l} "
              f"{lam_r} {m['tau_median']:>8.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
