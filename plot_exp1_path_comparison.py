#!/usr/bin/env python3
"""Path A vs Path B comparison plots for Experiment 1.

Produces the manuscript-bound comparison figures:
  - exp1_drift_plateau.png       : kappa_tail vs q_low sweep, both paths, with 90% CIs.
  - exp1_time_scale_spectrum.png : final log-rate / time-scale spectrum, both paths.

The envelope-at-convergence figure is handled by side-by-side subfigures in the
LaTeX source, using the existing per-path log_envelope_vs_ell.png files mirrored
into exp1_figures/ by plot_all.sh — no script needed for that one.

The Delta-zeta trajectory (exp1_delta_zeta.png) is retained as an audit-only
option: it can be regenerated with --which delta_zeta, but it is not included
in the default --which all set because the manuscript now shows the spectrum
itself rather than only its inter-quantile width.

The forcing-trajectory plot (exp1_forcing_trajectory.png) is retained as an
audit-only option: it can be regenerated with --which forcing, but is NOT
included in the default --which all set because Path A supplies forcing by
construction and the manuscript now reads Experiment 1 through drift,
spectrum, envelope, and phase diagnostics.

The same forcing-vs-epoch audit view is useful in the other experiments,
where it captures non-trivial multi-rung / multi-condition behaviour:
Experiment 2 has its own implementation in
``plot_exp2_phase_ladder.plot_forcing_trajectory`` (one trace per capacity-
ladder rung), and Experiment 3 will reuse the pattern across ablation
conditions. The function ``plot_forcing_trajectory`` below is kept here in
case anyone wants the audit view of the two-path comparison.

Usage:
    python plot_exp1_path_comparison.py
    python plot_exp1_path_comparison.py --which drift
    python plot_exp1_path_comparison.py \
        --path_b results/exp1_constgate_full/adamw \
        --path_a results/exp1_constgate_inject_full/adamw \
        --outdir results/exp1_figures

By default the script reads from the canonical Path B / Path A locations and
writes to results/exp1_figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----- color / style conventions (consistent across the three plots) -----
PATH_B_COLOR = "#0044aa"   # blue
PATH_A_COLOR = "#dd4400"   # orange-red
PATH_B_LABEL = "Path B (no injection)"
PATH_A_LABEL = r"Path A ($\alpha\!=\!1.6$ injection)"


# ----- helpers -----
def load_aggregated_csv(p: Path) -> List[Dict[str, str]]:
    """Load a phase_trajectory_aggregated.csv into a list of dicts."""
    if not p.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {p}")
    with open(p) as f:
        return list(csv.DictReader(f))


def load_tail_saturation(p: Path) -> Dict:
    """Load a tail_saturation.json (drift plateau output)."""
    if not p.exists():
        raise FileNotFoundError(f"tail_saturation.json not found: {p}")
    with open(p) as f:
        return json.load(f)


def find_drift_json(adamw_dir: Path) -> Path:
    """Find tail_saturation.json — prefer drift_const, fall back to drift_constgate."""
    for name in ("drift_const", "drift_constgate"):
        p = adamw_dir / name / "tail_saturation.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"tail_saturation.json not found in {adamw_dir}/drift_const or {adamw_dir}/drift_constgate"
    )


def extract_csv_series(
    rows: List[Dict[str, str]], col_mean: str, col_se: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (x=epoch, y_mean, y_se) arrays from an aggregated CSV."""
    x = np.array([int(r["epoch"]) for r in rows], dtype=int)
    y = np.array([float(r[col_mean]) for r in rows], dtype=float)
    s = np.array([float(r[col_se]) for r in rows], dtype=float)
    return x, y, s


def extract_tail_sweep(
    d: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (q_low, kappa_tail, ci_lo, ci_hi) from a tail_saturation.json."""
    items = sorted(d["results"].items(), key=lambda kv: float(kv[1]["q_low"]))
    q_low = np.array([r["q_low"] for _, r in items], dtype=float)
    k    = np.array([r["kappa_tail"]      for _, r in items], dtype=float)
    lo   = np.array([r["kappa_tail_ci_lo"] for _, r in items], dtype=float)
    hi   = np.array([r["kappa_tail_ci_hi"] for _, r in items], dtype=float)
    return q_low, k, lo, hi


def checkpoint_epoch(p: Path) -> int:
    """Extract the epoch from ckpt_XXXX_taus.csv; return -1 if malformed."""
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def final_tau_files(adamw_dir: Path, model: str = "const") -> List[Path]:
    """Return one final checkpoint tau CSV per seed directory."""
    files: List[Path] = []
    for seed_dir in sorted(adamw_dir.glob("seed_*")):
        tau_dir = seed_dir / model / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=checkpoint_epoch)
        if candidates:
            files.append(candidates[-1])
    if not files:
        raise FileNotFoundError(f"No per-seed checkpoint tau CSVs found under {adamw_dir}")
    return files


def load_final_taus(adamw_dir: Path, model: str = "const") -> np.ndarray:
    """Load and concatenate finite positive final-checkpoint tau values."""
    vals: List[float] = []
    for p in final_tau_files(adamw_dir, model=model):
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(tau) and tau > 0:
                    vals.append(tau)
    if not vals:
        raise ValueError(f"No finite positive tau values found under {adamw_dir}")
    return np.asarray(vals, dtype=float)


def load_phase_annotation(adamw_dir: Path, model: str = "const") -> str:
    """Compact canonical verdict annotation for the spectrum panel."""
    p = adamw_dir / "aggregated" / model / f"{model}_final_phase.json"
    if not p.exists():
        return ""
    with open(p) as f:
        d = json.load(f)
    phase = d.get("majority_phase_label") or d.get("phase_label") or "unknown"
    cd = d.get("crossover_diagnostic", {}) or {}
    power_pass = cd.get("power_law_window_pass")
    power_text = "power-law window present" if power_pass else "no resolved power-law window"
    return f"canonical verdict: {phase}\n{power_text}"


def ccdf_from_samples(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF for finite positive samples."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x = np.sort(x)
    n = x.size
    if n == 0:
        return x, x
    ccdf = (n - np.arange(1, n + 1)) / max(1, n)
    return x, ccdf


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian density used only as a visual light-tail reference."""
    sigma = max(float(sigma), 1e-12)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


# ----- plot 1: forcing trajectory -----
def plot_forcing_trajectory(path_b_dir: Path, path_a_dir: Path, outpath: Path, dpi: int = 400) -> None:
    b_csv = path_b_dir / "aggregated/const/phase_trajectory_aggregated.csv"
    a_csv = path_a_dir / "aggregated/const/phase_trajectory_aggregated.csv"
    b = load_aggregated_csv(b_csv)
    a = load_aggregated_csv(a_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Left panel: pooled ECF alpha across epochs (Path A first, then Path B)
    for label, rows, color in [
        (PATH_A_LABEL, a, PATH_A_COLOR),
        (PATH_B_LABEL, b, PATH_B_COLOR),
    ]:
        x, y, s = extract_csv_series(rows, "alpha_ecf_mean", "alpha_ecf_se")
        axes[0].plot(x, y, color=color, label=label, linewidth=1.5)
        axes[0].fill_between(x, y - s, y + s, color=color, alpha=0.18)
    axes[0].axhline(2.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    axes[0].axhline(1.8, color="gray", linewidth=0.5, linestyle=":",  alpha=0.4)
    axes[0].text(axes[0].get_xlim()[1] * 0.985, 2.0, " Gaussian (α=2)",
                 ha="right", va="bottom", fontsize=8, color="gray")
    axes[0].text(axes[0].get_xlim()[1] * 0.985, 1.8, r" $\alpha_{\mathrm{tail}}=1.8$",
                 ha="right", va="bottom", fontsize=8, color="gray")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Pooled $\hat\alpha_{\mathrm{ECF}}$")
    axes[0].set_ylim(0.5, 2.15)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    axes[0].set_title("Forcing proxy across training")
    axes[0].grid(alpha=0.25)

    # Right panel: estimator calibration audit across epochs (Path A first, then Path B)
    for label, rows, color in [
        (PATH_A_LABEL, a, PATH_A_COLOR),
        (PATH_B_LABEL, b, PATH_B_COLOR),
    ]:
        x, y, s = extract_csv_series(rows, "alpha_reliable_mean", "alpha_reliable_se")
        axes[1].plot(x, y, color=color, label=label, linewidth=1.5)
        axes[1].fill_between(x, y - s, y + s, color=color, alpha=0.18)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\langle\mathrm{alpha\_reliable}\rangle$ across seeds")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="center right", fontsize=9, framealpha=0.9)
    axes[1].set_title("Estimator reliability flag")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ----- plot 2: drift plateau -----
def plot_drift_plateau(path_b_dir: Path, path_a_dir: Path, outpath: Path, dpi: int = 400) -> None:
    b = load_tail_saturation(find_drift_json(path_b_dir))
    a = load_tail_saturation(find_drift_json(path_a_dir))

    q_b, k_b, lo_b, hi_b = extract_tail_sweep(b)
    q_a, k_a, lo_a, hi_a = extract_tail_sweep(a)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    yerr_b = np.array([k_b - lo_b, hi_b - k_b])
    yerr_a = np.array([k_a - lo_a, hi_a - k_a])

    off = 0.0035
    # Path A first, then Path B
    ax.errorbar(
        q_a - off, k_a, yerr=yerr_a, fmt="s",
        color=PATH_A_COLOR, label=PATH_A_LABEL,
        capsize=4, linewidth=1.5, markersize=6,
    )
    ax.errorbar(
        q_b + off, k_b, yerr=yerr_b, fmt="o",
        color=PATH_B_COLOR, label=PATH_B_LABEL,
        capsize=4, linewidth=1.5, markersize=6,
    )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.99, 0, " positive plateau\n required for closure",
            ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel(r"$q_{\mathrm{low}}$ (left-tail quantile cut)")
    ax.set_ylabel(r"$\hat\kappa_{\mathrm{tail}}$ (90% block-bootstrap CI)")
    ax.set_title("Far-left drift plateau — quantile-cut sweep")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ----- plot 3: final time-scale spectrum -----
def plot_time_scale_spectrum(path_b_dir: Path, path_a_dir: Path, outpath: Path, dpi: int = 400) -> None:
    tau_b = load_final_taus(path_b_dir)
    tau_a = load_final_taus(path_a_dir)
    z_b = -np.log(tau_b)
    z_a = -np.log(tau_a)
    annotation = load_phase_annotation(path_a_dir) or load_phase_annotation(path_b_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Left: p(zeta), the empirical stationary law in the SDE coordinate and
    # the natural partner of the drift diagnostic F(zeta). The Gaussian curve
    # is a same-mean/same-variance visual reference, not a goodness-of-fit
    # claim; in a positive-control anti-collapsed run this panel should show a
    # populated far-left tail rather than merely isolated outliers.
    z_all = np.concatenate([z_a, z_b])
    lo, hi = np.quantile(z_all, [0.005, 0.995])
    pad = 0.08 * max(hi - lo, 1e-6)
    bins = np.linspace(lo - pad, hi + pad, 34)
    x_grid = np.linspace(bins[0], bins[-1], 400)

    for label, z, color in [
        (PATH_A_LABEL, z_a, PATH_A_COLOR),
        (PATH_B_LABEL, z_b, PATH_B_COLOR),
    ]:
        axes[0].hist(
            z, bins=bins, density=True, histtype="step",
            linewidth=2.0, color=color, label=f"{label} empirical",
        )
        mu = float(np.mean(z))
        sigma = float(np.std(z, ddof=1))
        axes[0].plot(
            x_grid, normal_pdf(x_grid, mu, sigma),
            color=color, linestyle="--", linewidth=1.4,
            label=f"{label} Gaussian reference",
        )
        q10, q90 = np.quantile(z, [0.10, 0.90])
        axes[0].axvspan(q10, q90, color=color, alpha=0.07)

    axes[0].set_xlabel(r"$\zeta=-\log\tau$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Final log-rate spectrum")
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    axes[0].grid(alpha=0.25)

    # Right: tau CCDF on log-log axes, the tail object entering the
    # spectrum-to-envelope Tauberian correspondence. Anti-collapse should show
    # a resolved straight scaling window with slope -beta; a few rare large
    # time scales without such a window are finite-sample contamination, not a
    # power-law spectrum.
    for label, tau, color in [
        (PATH_A_LABEL, tau_a, PATH_A_COLOR),
        (PATH_B_LABEL, tau_b, PATH_B_COLOR),
    ]:
        xs, ys = ccdf_from_samples(tau)
        mask = ys > 0
        axes[1].plot(xs[mask], ys[mask], color=color, linewidth=1.8, label=label)
        q90, q99 = np.quantile(tau, [0.90, 0.99])
        axes[1].axvspan(q90, q99, color=color, alpha=0.06)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Time scale $\tau$")
    axes[1].set_ylabel(r"Empirical $\mathbb{P}(T\geq\tau)$")
    axes[1].set_title(r"Final $\tau$-spectrum tail")
    axes[1].legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    if annotation:
        axes[1].text(
            0.97, 0.97, annotation,
            transform=axes[1].transAxes,
            ha="right", va="top", fontsize=8.0, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.85),
        )
    axes[1].grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ----- plot 4: Delta zeta trajectory -----
def plot_delta_zeta(path_b_dir: Path, path_a_dir: Path, outpath: Path, dpi: int = 400) -> None:
    b_csv = path_b_dir / "aggregated/const/phase_trajectory_aggregated.csv"
    a_csv = path_a_dir / "aggregated/const/phase_trajectory_aggregated.csv"
    b = load_aggregated_csv(b_csv)
    a = load_aggregated_csv(a_csv)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Path A first, then Path B
    for label, rows, color in [
        (PATH_A_LABEL, a, PATH_A_COLOR),
        (PATH_B_LABEL, b, PATH_B_COLOR),
    ]:
        x, y, s = extract_csv_series(rows, "delta_zeta_mean", "delta_zeta_se")
        ax.plot(x, y, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Delta\zeta = \zeta_{q_{90}} - \zeta_{q_{10}}$")
    ax.set_title(r"Realized dynamic range across training")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ----- entry point -----
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path_b", type=Path,
        default=Path("results/exp1_constgate_full/adamw"),
        help="Path B (no injection) results directory; should contain "
             "aggregated/const/phase_trajectory_aggregated.csv and "
             "drift_const(gate)/tail_saturation.json",
    )
    parser.add_argument(
        "--path_a", type=Path,
        default=Path("results/exp1_constgate_inject_full/adamw"),
        help="Path A (with injection) results directory (same expected layout as --path_b)",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("results/exp1_figures"),
        help="Output directory for the comparison PNGs (default: results/exp1_figures)",
    )
    parser.add_argument(
        "--dpi", type=int, default=400,
        help="DPI for the saved PNG files (default: 400)",
    )
    parser.add_argument(
        "--which", type=str, default="all",
        choices=["all", "forcing", "drift", "spectrum", "delta_zeta"],
        help=("Which figures to produce. The default 'all' produces the two "
              "paper-bound comparison figures (drift plateau and final time-scale "
              "spectrum); 'forcing' and 'delta_zeta' are audit options and are "
              "NOT in 'all'."),
    )
    args = parser.parse_args()

    if not args.path_b.exists():
        print(f"ERROR: --path_b directory does not exist: {args.path_b}", file=sys.stderr)
        sys.exit(2)
    if not args.path_a.exists():
        print(f"ERROR: --path_a directory does not exist: {args.path_a}", file=sys.stderr)
        sys.exit(2)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Path B: {args.path_b}")
    print(f"Path A: {args.path_a}")
    print(f"Outdir: {args.outdir}")
    print()

    # 'all' produces only the manuscript-bound figures. The forcing-
    # trajectory panel is generated only on explicit request via
    # --which forcing (kept for audit, not paper-bound).
    if args.which == "forcing":
        plot_forcing_trajectory(args.path_b, args.path_a, args.outdir / "exp1_forcing_trajectory.png", dpi=args.dpi)
    if args.which in ("all", "drift"):
        plot_drift_plateau(args.path_b, args.path_a, args.outdir / "exp1_drift_plateau.png", dpi=args.dpi)
    if args.which in ("all", "spectrum"):
        plot_time_scale_spectrum(args.path_b, args.path_a, args.outdir / "exp1_time_scale_spectrum.png", dpi=args.dpi)
    if args.which == "delta_zeta":
        plot_delta_zeta(args.path_b, args.path_a, args.outdir / "exp1_delta_zeta.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
