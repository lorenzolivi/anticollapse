#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic 3 (runner): Far-tail drift-closure validation pipeline
==================================================================

End-to-end pipeline that produces the empirical evidence for the
far-left-tail closure F(ζ) = κ + o(1) used in the anti-collapse paper
(Appendix: app:drift_validation).

This is the *orchestrator* for Diagnostic 3. The core drift analysis lives
in ``run_restoring_drift.py`` (same folder); this script wraps it with the
training step and writes all diagnostic artifacts into the requested output
directory. It never copies figures into the manuscript/Overleaf tree.

Steps:
    1. Train the selected architecture on the heavy-tailed-lag task variant
       using ../main_exp1.py, with dense checkpointing so that the
       late-training tau spectrum is well sampled.
    2. Run ``run_restoring_drift.py`` on the resulting run directory with
       the far-tail saturation diagnostic enabled.
    3. Summarize the multi-projection alpha estimates saved during training:
       per-projection alpha_k, aggregate alpha_hat, and cross-projection
       dispersion.

Usage (run from the diagnostics/ folder):
    python run_drift_validation.py --outdir results_drift_validation
    python run_drift_validation.py --outdir results_drift_validation --skip_train
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime


THIS_DIR = os.path.dirname(os.path.abspath(__file__))                # diagnostics/
PROJECT_ROOT = os.path.dirname(THIS_DIR)                             # repo root
MAIN_EXP1 = os.path.join(PROJECT_ROOT, "main_exp1.py")
RESTORING_DRIFT = os.path.join(THIS_DIR, "run_restoring_drift.py")

CKPT_ALPHA_RE = re.compile(r"^ckpt_(\d+)_alpha_grad\.json$")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, check=True):
    log("[CMD] " + " ".join(str(c) for c in cmd))
    rc = subprocess.call(cmd)
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(str(c) for c in cmd)}")
    return rc


def _quantile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = float(q) * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if out == out else float("nan")


def summarize_alpha_projections(
    input_dir: str,
    model: str,
    outdir: str,
    tailiness_threshold: float = 1.8,
) -> None:
    """Write an explicit multi-projection alpha summary for the validation run.

    The summary reports three layers of evidence about the heavy-tailed
    forcing:

    1. The pooled (global) tail index ``alpha_hat`` (configured estimator,
       usually pooled ECF) together with the bootstrap CI for the pooled
       McCulloch estimate ``alpha_mcc_boot_ci_pooled`` saved by
       ``main_exp1.py``. The pooled estimate answers whether the entire
       projected-gradient cloud is globally heavy-tailed.
    2. The directional median and IQR of the per-projection McCulloch
       estimates ``alpha_hat_per_dir_mcc``. The directional distribution
       answers whether there are projection directions with heavy-tailed
       forcing.
    3. The directional tailiness fraction: the fraction of per-projection
       estimates strictly below ``tailiness_threshold`` (default 1.8).
       This collapses the directional distribution into a single
       interpretable summary that is more stable than the per-projection
       minimum.

    The minimum per-projection value is also reported, but should be read
    as a sensitivity flag rather than as an aggregate.
    """
    rows = []
    final_by_seed = {}

    if not os.path.isdir(input_dir):
        log(f"[alpha-summary] input directory not found: {input_dir}")
        return

    for seed_name in sorted(os.listdir(input_dir)):
        seed_dir = os.path.join(input_dir, seed_name)
        if not os.path.isdir(seed_dir) or not seed_name.startswith("seed_"):
            continue
        alpha_dir = os.path.join(seed_dir, model, "checkpoint_alpha")
        if not os.path.isdir(alpha_dir):
            continue
        for fname in sorted(os.listdir(alpha_dir)):
            m = CKPT_ALPHA_RE.match(fname)
            if not m:
                continue
            epoch = int(m.group(1))
            fpath = os.path.join(alpha_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as exc:
                log(f"[alpha-summary] could not read {fpath}: {exc}")
                continue

            per_dir = [
                _safe_float(v)
                for v in data.get("alpha_hat_per_dir_mcc", [])
            ]
            per_dir = [v for v in per_dir if v == v]
            if per_dir:
                med = _quantile(per_dir, 0.50)
                q25 = _quantile(per_dir, 0.25)
                q75 = _quantile(per_dir, 0.75)
                mean = sum(per_dir) / len(per_dir)
                var = sum((v - mean) ** 2 for v in per_dir) / max(1, len(per_dir) - 1)
                std = var ** 0.5
                cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
                amin = min(per_dir)
                amax = max(per_dir)
                frac_below = sum(1 for v in per_dir if v < tailiness_threshold) / len(per_dir)
            else:
                med = q25 = q75 = mean = std = cv = amin = amax = float("nan")
                frac_below = float("nan")

            mcc_boot_ci = data.get("alpha_mcc_boot_ci_pooled", [None, None]) or [None, None]
            mcc_boot_ci_lo = _safe_float(mcc_boot_ci[0]) if len(mcc_boot_ci) > 0 else float("nan")
            mcc_boot_ci_hi = _safe_float(mcc_boot_ci[1]) if len(mcc_boot_ci) > 1 else float("nan")

            row = {
                "model": model,
                "seed": seed_name,
                "epoch": epoch,
                "alpha_hat": _safe_float(data.get("alpha_hat")),
                "alpha_hat_se": _safe_float(data.get("alpha_hat_se")),
                "alpha_ecf": _safe_float(data.get("alpha_ecf")),
                "alpha_mcculloch": _safe_float(data.get("alpha_mcculloch")),
                "alpha_mcc_pooled": _safe_float(data.get("alpha_mcc_pooled")),
                "alpha_mcc_boot_ci_lo": mcc_boot_ci_lo,
                "alpha_mcc_boot_ci_hi": mcc_boot_ci_hi,
                "alpha_method": str(data.get("alpha_method", "")),
                "alpha_reliable": int(data.get("alpha_reliable", 0) or 0),
                "n_directions": int(data.get("n_directions", len(per_dir)) or len(per_dir)),
                "n_samples": int(data.get("n_samples", 0) or 0),
                "per_direction_estimator": "mcculloch",
                "tailiness_threshold": float(tailiness_threshold),
                "alpha_per_dir_values": json.dumps(per_dir),
                "alpha_per_dir_median": med,
                "alpha_per_dir_q25": q25,
                "alpha_per_dir_q75": q75,
                "alpha_per_dir_iqr": q75 - q25 if q75 == q75 and q25 == q25 else float("nan"),
                "alpha_per_dir_mean": mean,
                "alpha_per_dir_std": std,
                "alpha_per_dir_cv": cv,
                "alpha_per_dir_min": amin,
                "alpha_per_dir_max": amax,
                "alpha_per_dir_frac_below_threshold": frac_below,
            }
            rows.append(row)
            if seed_name not in final_by_seed or epoch > final_by_seed[seed_name]["epoch"]:
                final_by_seed[seed_name] = row

    if not rows:
        log(f"[alpha-summary] no checkpoint alpha files found for model={model}")
        return

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "alpha_projection_summary.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _finite(values):
        return [v for v in values if v == v]

    final_rows = list(final_by_seed.values())
    final_alpha = _finite([r["alpha_hat"] for r in final_rows])
    final_alpha_se = _finite([r["alpha_hat_se"] for r in final_rows])
    final_alpha_mcc_weighted = _finite([r["alpha_mcculloch"] for r in final_rows])
    final_alpha_mcc_pooled = _finite([r["alpha_mcc_pooled"] for r in final_rows])
    final_alpha_ecf = _finite([r["alpha_ecf"] for r in final_rows])
    final_per_dir_median = _finite([r["alpha_per_dir_median"] for r in final_rows])
    final_per_dir_iqr = _finite([r["alpha_per_dir_iqr"] for r in final_rows])
    final_per_dir_min = _finite([r["alpha_per_dir_min"] for r in final_rows])
    final_per_dir_frac = _finite([r["alpha_per_dir_frac_below_threshold"] for r in final_rows])
    final_mcc_ci_lo = _finite([r["alpha_mcc_boot_ci_lo"] for r in final_rows])
    final_mcc_ci_hi = _finite([r["alpha_mcc_boot_ci_hi"] for r in final_rows])

    summary = {
        "model": model,
        "n_checkpoint_records": len(rows),
        "n_final_seed_records": len(final_rows),
        "seeds": sorted(final_by_seed.keys()),
        "tailiness_threshold": float(tailiness_threshold),
        "per_direction_estimator": "mcculloch",
        "aggregate_alpha_estimator": "configured alpha_hat (usually pooled ECF)",
        "estimator_note": (
            "Three-layer reporting convention with explicit semantics. "
            "(i) Pooled ECF (primary global estimate): alpha_hat / "
            "alpha_ecf. The pooled ECF estimator does not currently "
            "propagate a bootstrap CI. The field alpha_hat_se is NOT a "
            "pooled-ECF standard error: in main_exp1.py it is computed "
            "as alpha_mcc_std / sqrt(K), i.e.\\ a cross-direction "
            "McCulloch dispersion proxy, and is retained here only for "
            "audit / cross-checking, not as ECF uncertainty. "
            "(ii) Pooled McCulloch (secondary global estimate with CI): "
            "alpha_mcc_pooled, with bootstrap CI "
            "alpha_mcc_boot_ci_pooled (surfaced as "
            "alpha_mcc_boot_ci_lo / alpha_mcc_boot_ci_hi). The CI "
            "applies specifically to alpha_mcc_pooled, not to "
            "alpha_hat / alpha_ecf and not to alpha_mcculloch. "
            "(iii) Directional tailiness (per-projection McCulloch): "
            "the simple per-direction median is alpha_per_dir_median "
            "(aggregated across seeds as "
            "final_alpha_per_dir_median_median); cross-projection "
            "dispersion is final_cross_projection_iqr_median; the stable "
            "directional summary is "
            "final_alpha_per_dir_frac_below_threshold_median (fraction "
            "of projections with alpha_k < tailiness_threshold). The "
            "per-projection minimum is a sensitivity flag only. "
            "Optional: alpha_mcculloch is a bootstrap-weighted "
            "directional median computed by main_exp1.py and is "
            "DIFFERENT from the simple per-direction median; if used in "
            "tables, label it explicitly as the weighted directional "
            "McCulloch. "
            "Anti-collapse does not require the pooled distribution to "
            "be heavy-tailed; a directional heavy-tailed component can "
            "be sufficient when the existence threshold is small."
        ),
        # Layer (i): pooled ECF point estimate. No bootstrap CI is
        # currently propagated. alpha_hat_se is a cross-direction
        # McCulloch dispersion proxy (see estimator_note), retained here
        # for audit purposes only -- it is NOT a pooled-ECF SE.
        "final_alpha_ecf_median": _quantile(final_alpha_ecf, 0.50),
        "final_alpha_hat_median": _quantile(final_alpha, 0.50),
        "final_alpha_hat_q25": _quantile(final_alpha, 0.25),
        "final_alpha_hat_q75": _quantile(final_alpha, 0.75),
        "final_alpha_hat_iqr": _quantile(final_alpha, 0.75) - _quantile(final_alpha, 0.25)
        if final_alpha else float("nan"),
        "final_alpha_hat_se_median": _quantile(final_alpha_se, 0.50),
        # Layer (ii): pooled McCulloch point estimate with matching
        # bootstrap CI. The CI fields below belong specifically to
        # alpha_mcc_pooled (NOT to alpha_hat, alpha_ecf, or alpha_mcculloch).
        "final_alpha_mcc_pooled_median": _quantile(final_alpha_mcc_pooled, 0.50),
        "final_alpha_mcc_pooled_q25": _quantile(final_alpha_mcc_pooled, 0.25),
        "final_alpha_mcc_pooled_q75": _quantile(final_alpha_mcc_pooled, 0.75),
        "final_alpha_mcc_boot_ci_lo_median": _quantile(final_mcc_ci_lo, 0.50),
        "final_alpha_mcc_boot_ci_hi_median": _quantile(final_mcc_ci_hi, 0.50),
        # Layer (iii): directional tailiness, simple per-direction
        # median. This is the directional-McCulloch summary that
        # belongs in the manuscript table; it is the cross-seed median
        # of the simple per-direction median computed in each row.
        "final_alpha_per_dir_median_median": _quantile(final_per_dir_median, 0.50),
        "final_alpha_per_dir_median_q25": _quantile(final_per_dir_median, 0.25),
        "final_alpha_per_dir_median_q75": _quantile(final_per_dir_median, 0.75),
        "final_cross_projection_iqr_median": _quantile(final_per_dir_iqr, 0.50),
        "final_alpha_per_dir_min_median": _quantile(final_per_dir_min, 0.50),
        "final_alpha_per_dir_frac_below_threshold_median": _quantile(final_per_dir_frac, 0.50),
        # Optional layer: bootstrap-weighted directional McCulloch
        # (alpha_mcculloch in main_exp1.py). This is DIFFERENT from
        # alpha_per_dir_median above; report only if explicitly labeled
        # as the weighted directional median.
        "final_alpha_mcculloch_median_weighted": _quantile(final_alpha_mcc_weighted, 0.50),
        "final_alpha_mcculloch_q25_weighted": _quantile(final_alpha_mcc_weighted, 0.25),
        "final_alpha_mcculloch_q75_weighted": _quantile(final_alpha_mcc_weighted, 0.75),
        "final_alpha_mcculloch_iqr_weighted": _quantile(final_alpha_mcc_weighted, 0.75) - _quantile(final_alpha_mcc_weighted, 0.25)
        if final_alpha_mcc_weighted else float("nan"),
        "final_records": final_rows,
    }
    json_path = os.path.join(outdir, "alpha_projection_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"[alpha-summary] wrote {csv_path}")
    log(f"[alpha-summary] wrote {json_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str,
                   default=os.path.join(THIS_DIR, "results_drift_validation"),
                   help="Root directory for all outputs of this run.")
    p.add_argument("--seeds", type=str, default="42,123,321",
                   help="Comma-separated seed list. DGX launcher overrides this for full validation.")
    p.add_argument("--model", type=str, default="gru",
                   choices=["const", "shared", "diag", "gru", "lstm"],
                   help="Architecture for the validation run (default: gru).")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd", "sgd_momentum"])

    # Training scale
    p.add_argument("--Nseq_train", type=int, default=8000)
    p.add_argument("--Nseq_diag", type=int, default=4000)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Dense checkpointing is important for the tail diagnostic: we need
    # enough late-training transitions to populate the far-left slice.
    p.add_argument("--checkpoint_every", type=int, default=20)

    # Heavy-tailed-lag task
    p.add_argument("--task_alpha", type=float, default=1.0,
                   help="Tail index α_task of the truncated-Pareto lag distribution.")
    p.add_argument("--task_lag_min", type=int, default=8)
    p.add_argument("--task_lag_max", type=int, default=384)
    p.add_argument("--task_K", type=int, default=8)
    p.add_argument("--task_lag_seed", type=int, default=20260410)
    p.add_argument("--noise_std", type=float, default=0.3)

    # Checkpoint diagnostics.  The drift diagnostic itself uses the saved
    # tau spectra, but the training run also records the alpha forcing proxy.
    # Exposing these knobs lets the DGX launcher use the multi-projection
    # protocol described in learnability.tex without changing main_exp1.py.
    p.add_argument("--alpha_n_directions", type=int, default=5,
                   help="Number of independent projection directions for alpha estimation.")
    p.add_argument("--alpha_n_grad_batches_ckpt", type=int, default=256,
                   help="Gradient-projection batches per checkpoint and projection direction.")
    p.add_argument("--alpha_grad_batch_size", type=int, default=256,
                   help="Batch size used when collecting gradient projections.")
    p.add_argument("--alpha_method", type=str, default="ecf",
                   choices=["ecf", "mcculloch"],
                   help="Primary alpha-estimation method used by main_exp1.py.")
    p.add_argument("--min_samples_alpha", type=int, default=500)

    # Tau-spectrum fit range used to construct zeta(t) = -log tau(t).
    p.add_argument("--tau_fit_lag_min", type=int, default=64)
    p.add_argument("--tau_fit_lag_max", type=int, default=256)
    p.add_argument("--tau_fit_num_lags", type=int, default=24)
    p.add_argument("--tau_ccdf_qmin", type=float, default=0.75)
    p.add_argument("--tau_ccdf_qmax", type=float, default=0.995)
    p.add_argument("--beta_bootstrap_B", type=int, default=2000)
    p.add_argument("--beta_bootstrap_ci", type=float, default=0.90)
    p.add_argument("--phase_r2_threshold", type=float, default=0.90)

    # Diagnostic knobs
    p.add_argument("--late_fraction", type=float, default=0.3)
    p.add_argument("--min_late_checkpoints", type=int, default=4)
    p.add_argument("--n_bins", type=int, default=18)
    p.add_argument("--tail_q_low_primary", type=float, default=0.10)
    p.add_argument("--tail_q_low_sweep", type=str, default="0.05,0.10,0.15")
    p.add_argument("--tail_trim_fraction", type=float, default=0.1)
    p.add_argument("--tail_bootstrap_B", type=int, default=2000)
    p.add_argument("--tail_ci_level", type=float, default=0.90)
    p.add_argument("--tail_bootstrap_seed", type=int, default=20260410)

    # Alpha-summary tailiness threshold: fraction of per-projection
    # estimates strictly below this value is reported as the
    # directional-tailiness summary. Default 1.8 follows the convention
    # discussed in the manuscript; 1.9 is a softer alternative.
    p.add_argument("--alpha_tailiness_threshold", type=float, default=1.8)

    # Pipeline control
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and only (re)run the diagnostic on an existing train_dir.")
    p.add_argument("--skip_diagnostic", action="store_true",
                   help="Skip the diagnostic step (useful for pure training reruns).")
    p.add_argument("--latex_figure_dir", type=str, default=None,
                   help="Deprecated no-op. Figures are never copied outside --outdir.")

    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    train_dir = os.path.join(args.outdir, "exp1")
    diag_dir = os.path.join(args.outdir, "drift_diagnostic")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: training (selected seeds, heavy-tailed-lag, dense checkpoints)
    # ------------------------------------------------------------------
    if not args.skip_train:
        if not os.path.exists(MAIN_EXP1):
            raise FileNotFoundError(f"Could not find main_exp1.py at {MAIN_EXP1}")
        train_cmd = [
            sys.executable, MAIN_EXP1,
            "--outdir", train_dir,
            "--seeds", args.seeds,
            "--models", args.model,
            "--optimizer", args.optimizer,
            "--Nseq_train", str(args.Nseq_train),
            "--Nseq_diag", str(args.Nseq_diag),
            "--T", str(args.T),
            "--D", str(args.D),
            "--H", str(args.H),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--grad_clip", str(args.grad_clip),
            "--checkpoint_every", str(args.checkpoint_every),
            "--task_variant", "heavy_tail",
            "--task_alpha", str(args.task_alpha),
            "--task_lag_min", str(args.task_lag_min),
            "--task_lag_max", str(args.task_lag_max),
            "--task_K", str(args.task_K),
            "--task_lag_seed", str(args.task_lag_seed),
            "--noise_std", str(args.noise_std),
            "--alpha_n_directions", str(args.alpha_n_directions),
            "--alpha_n_grad_batches_ckpt", str(args.alpha_n_grad_batches_ckpt),
            "--alpha_grad_batch_size", str(args.alpha_grad_batch_size),
            "--alpha_method", args.alpha_method,
            "--min_samples_alpha", str(args.min_samples_alpha),
            "--tau_fit_lag_min", str(args.tau_fit_lag_min),
            "--tau_fit_lag_max", str(args.tau_fit_lag_max),
            "--tau_fit_num_lags", str(args.tau_fit_num_lags),
            "--tau_ccdf_qmin", str(args.tau_ccdf_qmin),
            "--tau_ccdf_qmax", str(args.tau_ccdf_qmax),
            "--beta_bootstrap_B", str(args.beta_bootstrap_B),
            "--beta_bootstrap_ci", str(args.beta_bootstrap_ci),
            "--phase_r2_threshold", str(args.phase_r2_threshold),
            "--device", args.device,
            "--skip_plot",
        ]
        run(train_cmd)
    else:
        log("Skipping training step (--skip_train).")

    # ------------------------------------------------------------------
    # Step 2: drift diagnostic with far-tail saturation
    # ------------------------------------------------------------------
    # main_exp1.py puts seed directories under <train_dir>/<optimizer>/seed_*/.
    input_dir = os.path.join(train_dir, args.optimizer)
    if not args.skip_diagnostic:
        if not os.path.exists(RESTORING_DRIFT):
            raise FileNotFoundError(f"Could not find run_restoring_drift.py at {RESTORING_DRIFT}")
        diag_cmd = [
            sys.executable, RESTORING_DRIFT,
            "--input_dir", input_dir,
            "--model", args.model,
            "--outdir", diag_dir,
            "--late_fraction", str(args.late_fraction),
            "--min_late_checkpoints", str(args.min_late_checkpoints),
            "--n_bins", str(args.n_bins),
            "--tail_q_low_primary", str(args.tail_q_low_primary),
            "--tail_q_low_sweep", args.tail_q_low_sweep,
            "--tail_trim_fraction", str(args.tail_trim_fraction),
            "--tail_bootstrap_B", str(args.tail_bootstrap_B),
            "--tail_ci_level", str(args.tail_ci_level),
            "--tail_bootstrap_seed", str(args.tail_bootstrap_seed),
        ]
        run(diag_cmd)
    else:
        log("Skipping diagnostic step (--skip_diagnostic).")

    # ------------------------------------------------------------------
    # Step 3: summarize multi-projection alpha estimates
    # ------------------------------------------------------------------
    summarize_alpha_projections(
        input_dir=input_dir,
        model=args.model,
        outdir=diag_dir,
        tailiness_threshold=args.alpha_tailiness_threshold,
    )

    # ------------------------------------------------------------------
    # Step 4: report local figure path only. Do not copy into latex/figures.
    # ------------------------------------------------------------------
    src_fig = os.path.join(diag_dir, "conditional_drift_with_tail.png")
    if args.latex_figure_dir:
        log("[WARN] --latex_figure_dir is deprecated and ignored; "
            "copy figures to Overleaf manually after review.")
    if os.path.exists(src_fig):
        log(f"Final figure at: {src_fig}")
    else:
        log("[WARN] Combined drift figure not found; check the diagnostic log.")

    log("Done.")


if __name__ == "__main__":
    main()
