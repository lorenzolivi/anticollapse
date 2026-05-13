#!/usr/bin/env python3
"""Capacity-ladder phase-trajectory plots for Experiment 2.

Experiment 2 of the Anti-Collapse paper walks a four-rung capacity ladder of
gated RNN architectures (SharedGate, DiagGate, GRU, LSTM) under the *same*
training pipeline used for Experiment 1, *without* any heavy-tailed gradient
injection. The pre-registered question is: at which rung of the ladder does
spontaneous training transition from a collapsed phase verdict to a sustained
anti-collapsed one? The plot set is meant to show the transition directly in
the drift, spectrum, envelope, and phase diagnostics.

This script produces the manuscript figures that overlay the four
architectures on a common axis, each driven from the per-seed aggregates that
``main_phase_trajectory.py`` already writes under
    <EXP2_OUTDIR>/<optimizer>/aggregated/<model>/

The figures are:

  - exp2_forcing_trajectory.png : pooled ECF alpha (left) and estimator
    calibration audit (right), one trace per architecture.
  - exp2_beta_env_trajectory.png : envelope exponent beta_env across training,
    one trace per architecture. The order parameter of the dynamical
    transition.
  - exp2_drift_plateau.png : far-left drift plateau kappa_tail vs q_low sweep,
    one error-bar series per architecture. Positivity at the primary q_low
    cut is the drift-side gate of the saturating closure.
  - exp2_time_scale_spectrum.png : final log-rate density p(zeta) and
    empirical tau-CCDF, one trace per architecture. The zeta panel shows the
    stationary law modeled by the SDE and pairs naturally with the drift
    diagnostic F(zeta); the tau-CCDF panel shows the tail condition fed into
    the spectrum-to-envelope Tauberian correspondence.
  - exp2_delta_zeta.png : realized dynamic range Delta zeta = zeta_q90 -
    zeta_q10 across training, one trace per architecture. This is a scalar
    width summary, not a replacement for the spectrum plot.
  - exp2_phase_summary.png : grouped bar chart of phase-verdict counts per
    architecture (collapsed / canonical AC / robust AC), from the canonical
    aggregated <model>_final_phase.json.
  - exp2_t_cross.png : per-seed scatter of t_cross_epoch per architecture,
    with right-censored seeds marked at the horizon. Localizes when
    (if ever) each rung first realizes the AC verdict.

The macro envelope ``log_envelope_vs_ell`` plot per architecture is produced
by ``plot_all.sh`` by invoking the existing envelope plotter on each
``aggregated/<arch>/`` directory, then mirroring the PNGs into the same output
folder (so all Overleaf-bound exp2 figures live in one folder).

Usage:
    python plot_exp2_phase_ladder.py
    python plot_exp2_phase_ladder.py --which forcing
    python plot_exp2_phase_ladder.py \\
        --exp2_dir results/exp2_phase_full/adamw \\
        --outdir results/exp2_figures \\
        --architectures shared,diag,gru,lstm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo-local matplotlib cache so the script is headless-safe and does not
# litter the user's home directory with config caches. Must be set BEFORE
# matplotlib is imported.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Architecture ordering, colors, and human-readable labels.
#
# The capacity ladder is encoded by the order of CANONICAL_ARCHS:
# SharedGate (lowest capacity) -> DiagGate -> GRU -> LSTM (highest capacity).
# We use a cool->warm color sequence so the ladder reads visually as
# "expressive capacity increasing left-to-right" in legends and bar charts.
# ============================================================
CANONICAL_ARCHS: List[str] = ["shared", "diag", "gru", "lstm"]

ARCH_LABEL: Dict[str, str] = {
    "diag":   "DiagGate",
    "shared": "SharedGate",
    "gru":    "GRU",
    "lstm":   "LSTM",
}

ARCH_COLOR: Dict[str, str] = {
    "shared": "#2b6ab8",   # cool blue (lowest capacity)
    "diag":   "#3aa050",   # green
    "gru":    "#dd8800",   # orange
    "lstm":   "#c43030",   # warm red (highest capacity)
}

# Phase verdicts the canonical Table-1 rule emits, in the order we want them
# to appear in stacked / grouped bars.
PHASE_ORDER: List[str] = [
    "collapsed",
    "concentrated anti-collapse",
    "anti-collapse",
]
PHASE_LABEL: Dict[str, str] = {
    "collapsed":                   "Collapsed",
    "concentrated anti-collapse":  "Canonical AC",
    "anti-collapse":               "Robust AC",
}
PHASE_FILL: Dict[str, str] = {
    "collapsed":                   "#bcbcbc",  # neutral gray
    "concentrated anti-collapse":  "#f5b400",  # warm gold
    "anti-collapse":               "#3aa050",  # green (matches sharedgate
                                                # incidentally — fine, the
                                                # contexts never collide)
}


# ============================================================
# Small CSV / JSON loaders.
#
# We keep these intentionally tiny and dependency-free (no pandas). The
# aggregated CSVs have a flat (epoch, mean, se) layout; the JSONs have a
# small fixed schema. If a file is missing we return None so the caller can
# emit a skipped-trace message instead of crashing — useful when only some
# rungs of the ladder are on disk.
# ============================================================
def _load_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_epoch_series(
    rows: List[Dict[str, str]],
    mean_col: str,
    se_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a (epoch, mean, se) triple from an aggregated CSV.

    If ``se_col`` is None or missing from the row, the returned se is zero.
    Missing or non-numeric values are masked out (the corresponding epoch is
    dropped) so that a single bad checkpoint doesn't blow up the trace.
    """
    xs: List[int] = []
    ys: List[float] = []
    ss: List[float] = []
    for r in rows:
        try:
            x = int(r["epoch"])
            y = float(r[mean_col])
        except (KeyError, ValueError):
            continue
        if se_col is None or se_col not in r:
            s = 0.0
        else:
            try:
                s = float(r[se_col])
            except ValueError:
                s = 0.0
        if not np.isfinite(y):
            continue
        if not np.isfinite(s):
            s = 0.0
        xs.append(x)
        ys.append(y)
        ss.append(s)
    return np.asarray(xs), np.asarray(ys), np.asarray(ss)


# ============================================================
# Per-architecture path resolution.
#
# main_phase_trajectory.py writes per-architecture aggregates under
#   <exp2_dir>/aggregated/<model>/
# (and per-arch envelope plots under <exp2_dir>/plots/<model>/ via the
# downstream plot_exp1_envelopes.py pass). We expose helpers that map an
# architecture short name to those paths so the figure code stays clean.
# ============================================================
def _agg_dir(exp2_dir: Path, arch: str) -> Path:
    return exp2_dir / "aggregated" / arch


def _drift_json_path(exp2_dir: Path, arch: str) -> Path:
    """Return the tail_saturation.json path for a given architecture.

    plot_all.sh's drift block writes ``drift_<arch>/tail_saturation.json`` —
    e.g. ``drift_gru/``, ``drift_diag/``. (For exp1's ConstGate the historical
    pre-rename name was ``drift_const`` or ``drift_constgate``, but exp2 is
    new code, so we standardize on ``drift_<arch>``.)
    """
    return exp2_dir / f"drift_{arch}" / "tail_saturation.json"


def _checkpoint_epoch(p: Path) -> int:
    """Extract epoch from ckpt_XXXX_taus.csv; return -1 if malformed."""
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _final_tau_files(exp2_dir: Path, arch: str) -> List[Path]:
    """Return one final checkpoint tau CSV per seed for a given architecture."""
    files: List[Path] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        tau_dir = seed_dir / arch / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
        if candidates:
            files.append(candidates[-1])
    return files


def _load_final_taus(exp2_dir: Path, arch: str) -> Optional[np.ndarray]:
    """Load finite positive final-checkpoint tau values for one architecture."""
    vals: List[float] = []
    for p in _final_tau_files(exp2_dir, arch):
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(tau) and tau > 0:
                    vals.append(tau)
    if not vals:
        return None
    return np.asarray(vals, dtype=float)


def _ccdf_from_samples(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF for finite positive samples."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    n = x.size
    if n == 0:
        return x, x
    ccdf = (n - np.arange(1, n + 1)) / max(1, n)
    return x, ccdf


# ============================================================
# Figure 1: forcing trajectory across training (per architecture).
#
# Two-panel: pooled ECF alpha (left) and estimator-calibration audit (right).
# Experiment 2 has no injected forcing, so this view is used as a sanity check
# on spontaneous forcing development across the capacity ladder.
# ============================================================
def plot_forcing_trajectory(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    for arch in architectures:
        rows = _load_csv_rows(_agg_dir(exp2_dir, arch) / "phase_trajectory_aggregated.csv")
        if rows is None:
            print(f"  [forcing] skip {arch}: aggregated CSV missing")
            continue
        label = ARCH_LABEL.get(arch, arch)
        color = ARCH_COLOR.get(arch, None)

        x, y, s = _extract_epoch_series(rows, "alpha_ecf_mean", "alpha_ecf_se")
        axes[0].plot(x, y, color=color, label=label, linewidth=1.5)
        axes[0].fill_between(x, y - s, y + s, color=color, alpha=0.18)

        x2, y2, s2 = _extract_epoch_series(rows, "alpha_reliable_mean", "alpha_reliable_se")
        axes[1].plot(x2, y2, color=color, label=label, linewidth=1.5)
        axes[1].fill_between(x2, y2 - s2, y2 + s2, color=color, alpha=0.18)

    axes[0].axhline(2.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    axes[0].axhline(1.8, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Pooled $\hat\alpha_{\mathrm{ECF}}$")
    axes[0].set_ylim(0.5, 2.15)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    axes[0].set_title("Forcing proxy across training")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\langle\mathrm{alpha\_reliable}\rangle$ across seeds")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="lower right", fontsize=9, framealpha=0.9)
    axes[1].set_title("Estimator reliability flag")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 2: envelope exponent beta_env across training.
#
# The envelope exponent is the order parameter of the dynamical transition:
# beta_env > 0 with R^2 > threshold is the power-law-window signature of
# anti-collapse. We plot all architectures on a single panel with a
# horizontal reference line at the power-law-window minimum
# (power_window_beta_min, default 0.10).
# ============================================================
def plot_beta_env_trajectory(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
    beta_floor: float = 0.10,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for arch in architectures:
        rows = _load_csv_rows(_agg_dir(exp2_dir, arch) / "phase_trajectory_aggregated.csv")
        if rows is None:
            print(f"  [beta_env] skip {arch}: aggregated CSV missing")
            continue
        label = ARCH_LABEL.get(arch, arch)
        color = ARCH_COLOR.get(arch, None)
        x, y, s = _extract_epoch_series(rows, "beta_env_mean", "beta_env_se")
        ax.plot(x, y, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)

    ax.axhline(beta_floor, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.text(
        ax.get_xlim()[1] * 0.985, beta_floor,
        rf" power-law window floor $\beta_{{\min}}={beta_floor:.2f}$",
        ha="right", va="bottom", fontsize=8, color="gray",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Envelope exponent $\hat\beta_{\mathrm{env}}$ across seeds")
    ax.set_title("Envelope exponent across training (capacity ladder)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 3: far-left drift plateau kappa_tail vs q_low sweep.
#
# Drift positivity at the primary q_low cut is the drift-side gate of the
# saturating closure. We expect kappa_tail > 0 once the architecture reaches
# anti-collapse, and kappa_tail <= 0 below the capacity threshold.
# ============================================================
def plot_drift_plateau(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Spread the x-positions slightly between architectures so the error
    # bars don't overlap visually at the same q_low.
    arch_offsets = np.linspace(-0.0055, 0.0055, num=max(len(architectures), 1))

    for arch, off in zip(architectures, arch_offsets):
        js = _load_json(_drift_json_path(exp2_dir, arch))
        if js is None or "results" not in js:
            print(f"  [drift] skip {arch}: tail_saturation.json missing")
            continue
        label = ARCH_LABEL.get(arch, arch)
        color = ARCH_COLOR.get(arch, None)

        items = sorted(js["results"].items(), key=lambda kv: float(kv[1]["q_low"]))
        q_low = np.array([r["q_low"] for _, r in items], dtype=float)
        k = np.array([r["kappa_tail"] for _, r in items], dtype=float)
        lo = np.array([r["kappa_tail_ci_lo"] for _, r in items], dtype=float)
        hi = np.array([r["kappa_tail_ci_hi"] for _, r in items], dtype=float)
        yerr = np.array([k - lo, hi - k])

        ax.errorbar(
            q_low + off, k, yerr=yerr, fmt="o",
            color=color, label=label, capsize=4, linewidth=1.5, markersize=6,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$q_{\mathrm{low}}$ (left-tail quantile cut)")
    ax.set_ylabel(r"$\hat\kappa_{\mathrm{tail}}$ (90% block-bootstrap CI)")
    ax.set_title("Far-left drift plateau — quantile-cut sweep (capacity ladder)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 4: final time-scale spectrum.
#
# This is the spectrum-level view of the phase claim. The left panel shows
# p(zeta), the empirical stationary law of the SDE coordinate and the natural
# partner of the drift diagnostic F(zeta). The right panel shows the tau-CCDF,
# the asymptotic object entering the Laplace/Tauberian envelope theorem.
# Boundary pathologies show up differently in the two panels: a few far-left
# zeta outliers or rare large taus can contaminate the tails, but
# anti-collapse should produce both a populated far-left zeta tail and a
# resolved log-log CCDF scaling window, not just isolated outliers.
# ============================================================
def plot_time_scale_spectrum(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for arch in architectures:
        tau = _load_final_taus(exp2_dir, arch)
        if tau is None:
            print(f"  [spectrum] skip {arch}: final checkpoint taus missing")
            continue
        spectra[arch] = (tau, -np.log(tau))

    if not spectra:
        print("  [spectrum] no architecture has final tau data; skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    z_all = np.concatenate([z for _, z in spectra.values()])
    lo, hi = np.quantile(z_all, [0.005, 0.995])
    pad = 0.08 * max(hi - lo, 1e-6)
    bins = np.linspace(lo - pad, hi + pad, 38)

    for arch in architectures:
        if arch not in spectra:
            continue
        tau, z = spectra[arch]
        label = ARCH_LABEL.get(arch, arch)
        color = ARCH_COLOR.get(arch, None)
        axes[0].hist(
            z, bins=bins, density=True, histtype="step",
            linewidth=1.8, color=color, label=label,
        )
        q10, q90 = np.quantile(z, [0.10, 0.90])
        axes[0].axvspan(q10, q90, color=color, alpha=0.045)

        xs, ys = _ccdf_from_samples(tau)
        mask = ys > 0
        axes[1].plot(xs[mask], ys[mask], color=color, linewidth=1.8, label=label)
        q90, q99 = np.quantile(tau, [0.90, 0.99])
        axes[1].axvspan(q90, q99, color=color, alpha=0.035)

    axes[0].set_xlabel(r"$\zeta=-\log\tau$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Final log-rate spectrum")
    axes[0].legend(loc="best", fontsize=8.5, framealpha=0.9)
    axes[0].grid(alpha=0.25)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Time scale $\tau$")
    axes[1].set_ylabel(r"Empirical $\mathbb{P}(T\geq\tau)$")
    axes[1].set_title(r"Final $\tau$-spectrum tail")
    axes[1].legend(loc="best", fontsize=8.5, framealpha=0.9)
    axes[1].grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 5: realized dynamic range Delta zeta across training.
#
# Delta zeta = zeta_q90 - zeta_q10 measures how broadly the realized
# log-effective decay-rate distribution spreads. A widening trajectory is a
# scalar signature of the spectrum filling out under spontaneous training.
# It should be interpreted alongside exp2_time_scale_spectrum.png, which
# shows the actual zeta distribution and tau tail.
# ============================================================
def plot_delta_zeta(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for arch in architectures:
        rows = _load_csv_rows(_agg_dir(exp2_dir, arch) / "phase_trajectory_aggregated.csv")
        if rows is None:
            print(f"  [delta_zeta] skip {arch}: aggregated CSV missing")
            continue
        label = ARCH_LABEL.get(arch, arch)
        color = ARCH_COLOR.get(arch, None)
        x, y, s = _extract_epoch_series(rows, "delta_zeta_mean", "delta_zeta_se")
        ax.plot(x, y, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Delta\zeta = \zeta_{q_{90}} - \zeta_{q_{10}}$")
    ax.set_title(r"Realized dynamic range across training (capacity ladder)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 6: phase-verdict summary across the capacity ladder.
#
# Reads <model>_final_phase.json's phase_counts dict for each architecture
# and renders a grouped bar chart of fractional counts (collapsed / canonical
# AC / robust AC), normalized by n_seeds. This is the "headline plot" of
# exp2: at which rung of the ladder does the verdict flip?
# ============================================================
def plot_phase_summary(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    # Collect normalized counts per architecture for each phase label.
    counts_per_arch: Dict[str, Dict[str, float]] = {}
    seeds_per_arch: Dict[str, int] = {}
    for arch in architectures:
        js = _load_json(_agg_dir(exp2_dir, arch) / f"{arch}_final_phase.json")
        if js is None:
            print(f"  [phase_summary] skip {arch}: final_phase.json missing")
            counts_per_arch[arch] = {p: 0.0 for p in PHASE_ORDER}
            seeds_per_arch[arch] = 0
            continue
        n = int(js.get("n_seeds", 0))
        raw_counts = dict(js.get("phase_counts", {}))
        seeds_per_arch[arch] = n
        if n <= 0:
            counts_per_arch[arch] = {p: 0.0 for p in PHASE_ORDER}
            continue
        counts_per_arch[arch] = {p: float(raw_counts.get(p, 0)) / n for p in PHASE_ORDER}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    n_arch = len(architectures)
    n_phase = len(PHASE_ORDER)
    bar_width = 0.8 / max(n_phase, 1)
    x_idx = np.arange(n_arch)

    for k, phase in enumerate(PHASE_ORDER):
        heights = [counts_per_arch[arch][phase] for arch in architectures]
        ax.bar(
            x_idx + (k - (n_phase - 1) / 2.0) * bar_width,
            heights,
            width=bar_width,
            label=PHASE_LABEL[phase],
            color=PHASE_FILL[phase],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{ARCH_LABEL.get(a, a)}\n(n={seeds_per_arch[a]})" for a in architectures])
    ax.set_ylabel("Fraction of seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final phase verdict across the capacity ladder")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 7: per-seed t_cross epoch per architecture.
#
# The threshold-crossing observable t_cross is the first epoch at which the
# diagnostic flips from collapsed to anti-collapse (concentrated or robust).
# We render it as a strip plot: x = architecture (capacity ladder), y =
# epoch. Right-censored seeds (never crossed) are drawn as open markers at
# the horizon.
# ============================================================
def plot_t_cross(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rng = np.random.default_rng(20260511)

    horizon = 0  # set from the largest horizon_epoch we see
    for arch_idx, arch in enumerate(architectures):
        js = _load_json(_agg_dir(exp2_dir, arch) / f"{arch}_threshold_crossing.json")
        if js is None or "per_seed" not in js:
            print(f"  [t_cross] skip {arch}: threshold_crossing.json missing")
            continue
        traj_rows = _load_csv_rows(_agg_dir(exp2_dir, arch) / "phase_trajectory_aggregated.csv")
        epoch_per_step = None
        if traj_rows:
            try:
                final_epoch = float(traj_rows[-1]["epoch"])
                final_step = float(traj_rows[-1]["step"])
                if np.isfinite(final_epoch) and np.isfinite(final_step) and final_step > 0:
                    epoch_per_step = final_epoch / final_step
            except (KeyError, ValueError, TypeError):
                epoch_per_step = None
        color = ARCH_COLOR.get(arch, None)
        # Each per_seed entry has either an observed crossing (t_cross_epoch)
        # or a right_censored flag with a horizon epoch.  Older JSONs only
        # recorded horizon_step, so we convert via the aggregate step/epoch
        # ratio when needed.
        for sd in js["per_seed"]:
            if bool(sd.get("crossed", False)) and sd.get("t_cross_epoch") is not None:
                t = float(sd["t_cross_epoch"])
                horizon = max(horizon, t)
                ax.scatter(
                    arch_idx + rng.uniform(-0.07, 0.07),
                    t, marker="o", color=color, s=60,
                    edgecolor="black", linewidth=0.5, alpha=0.85,
                )
            else:
                # Right-censored: mark at horizon epoch with an open
                # downward-pointing triangle.
                horizon_epoch = sd.get("horizon_epoch")
                if horizon_epoch is None:
                    horizon_step = sd.get("horizon_step", np.nan)
                    if epoch_per_step is not None and np.isfinite(float(horizon_step)):
                        horizon_epoch = float(horizon_step) * epoch_per_step
                    else:
                        horizon_epoch = np.nan
                if horizon_epoch is not None and np.isfinite(float(horizon_epoch)):
                    he = float(horizon_epoch)
                    horizon = max(horizon, he)
                    ax.scatter(
                        arch_idx + rng.uniform(-0.07, 0.07),
                        he, marker="v", facecolor="none",
                        edgecolor=color, s=70, linewidth=1.2,
                    )

    ax.set_xticks(np.arange(len(architectures)))
    ax.set_xticklabels([ARCH_LABEL.get(a, a) for a in architectures])
    ax.set_ylabel(r"Threshold-crossing epoch $t_{\mathrm{cross}}$")
    ax.set_title("Per-seed threshold crossing (open = right-censored)")
    if horizon > 0:
        ax.set_ylim(0, 1.05 * horizon)
    ax.grid(alpha=0.25, axis="y")

    # Build a custom legend explaining marker semantics.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="observed crossing"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=9, label="right-censored (never crossed)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# CLI
# ============================================================
def parse_arch_list(s: str) -> List[str]:
    """Comma-separated architecture list; tolerates spaces and case."""
    items = [a.strip().lower() for a in s.split(",") if a.strip()]
    # Normalize known aliases.
    aliases = {"diaggate": "diag", "sharedgate": "shared"}
    return [aliases.get(a, a) for a in items]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp2_dir", type=Path,
        default=Path("results/exp2_phase_full/adamw"),
        help="Experiment 2 result directory (the per-optimizer subdir that "
             "contains aggregated/<arch>/ and drift_<arch>/). Default: "
             "results/exp2_phase_full/adamw",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("results/exp2_figures"),
        help="Output directory for the comparison PNGs "
             "(default: results/exp2_figures).",
    )
    parser.add_argument(
        "--architectures", type=str, default=",".join(CANONICAL_ARCHS),
        help=("Comma-separated architecture short names, in capacity-ladder "
              "order. Default: 'shared,diag,gru,lstm'. Aliases 'diaggate' "
              "and 'sharedgate' are accepted."),
    )
    parser.add_argument(
        "--dpi", type=int, default=400,
        help="DPI for the saved PNG files (default: 400).",
    )
    parser.add_argument(
        "--which", type=str, default="all",
        choices=["all", "forcing", "beta_env", "drift", "spectrum", "delta_zeta",
                 "phase_summary", "t_cross"],
        help="Which figure(s) to produce (default: all).",
    )
    args = parser.parse_args()

    if not args.exp2_dir.exists():
        print(f"ERROR: --exp2_dir does not exist: {args.exp2_dir}", file=sys.stderr)
        sys.exit(2)
    args.outdir.mkdir(parents=True, exist_ok=True)
    architectures = parse_arch_list(args.architectures)
    if not architectures:
        print("ERROR: no architectures parsed from --architectures", file=sys.stderr)
        sys.exit(2)

    print(f"exp2_dir       : {args.exp2_dir}")
    print(f"outdir         : {args.outdir}")
    print(f"architectures  : {architectures}")
    print(f"dpi            : {args.dpi}")
    print()

    if args.which in ("all", "forcing"):
        plot_forcing_trajectory(
            args.exp2_dir, architectures,
            args.outdir / "exp2_forcing_trajectory.png", dpi=args.dpi,
        )
    if args.which in ("all", "beta_env"):
        plot_beta_env_trajectory(
            args.exp2_dir, architectures,
            args.outdir / "exp2_beta_env_trajectory.png", dpi=args.dpi,
        )
    if args.which in ("all", "drift"):
        plot_drift_plateau(
            args.exp2_dir, architectures,
            args.outdir / "exp2_drift_plateau.png", dpi=args.dpi,
        )
    if args.which in ("all", "spectrum"):
        plot_time_scale_spectrum(
            args.exp2_dir, architectures,
            args.outdir / "exp2_time_scale_spectrum.png", dpi=args.dpi,
        )
    if args.which in ("all", "delta_zeta"):
        plot_delta_zeta(
            args.exp2_dir, architectures,
            args.outdir / "exp2_delta_zeta.png", dpi=args.dpi,
        )
    if args.which in ("all", "phase_summary"):
        plot_phase_summary(
            args.exp2_dir, architectures,
            args.outdir / "exp2_phase_summary.png", dpi=args.dpi,
        )
    if args.which in ("all", "t_cross"):
        plot_t_cross(
            args.exp2_dir, architectures,
            args.outdir / "exp2_t_cross.png", dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
