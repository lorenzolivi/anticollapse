#!/usr/bin/env python3
"""Path A vs Path B comparison plots for Experiment 1.

Produces the manuscript-bound comparison figures:
  - exp1_drift_plateau.png       : kappa_tail vs q_low sweep, both paths, with 90% CIs.
  - exp1_time_scale_spectrum_fitlag.png : final log-rate / time-scale spectrum,
    capped at max(tau_fit_lags), both paths.

The broader sequence-length-capped spectrum is also useful as an appendix /
audit view and is written as exp1_time_scale_spectrum.png when
--tau_cap_mode seq_len is used.

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

Experiment 2 has its own implementation in
``plot_exp2_phase_ladder.plot_forcing_trajectory``. The function
``plot_forcing_trajectory`` below is kept here in case anyone wants the
audit view of the two-path comparison.

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
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
PATH_A_LABEL = "Path A (injection)"


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


def infer_path_a_label(path_a_dir: Path) -> str:
    """Build the Path A legend label from the saved run configuration."""
    cfg_path = path_a_dir / "exp1_config.json"
    if not cfg_path.exists():
        return PATH_A_LABEL
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        alpha = float(cfg.get("inject_alpha"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return PATH_A_LABEL
    return rf"Path A ($\alpha\!=\!{alpha:g}$ injection)"


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
    label: str = "drift",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull valid (q_low, kappa_tail, ci_lo, ci_hi) entries from drift output.

    Smoke runs often skip the deepest q-cuts because the single-seed tail slice
    has too few samples. Those rows are useful audit metadata, but they should
    not be plotted as if they had plateau estimates.
    """
    all_items = sorted(d["results"].items(), key=lambda kv: float(kv[1].get("q_low", kv[0])))
    skipped = [
        str(r.get("q_low", key))
        for key, r in all_items
        if bool(r.get("skipped", False)) or "kappa_tail" not in r
    ]
    if skipped:
        print(
            f"  [drift] {label}: skipped q_low={', '.join(skipped)} "
            "(insufficient usable tail samples)"
        )

    items = [
        (key, r)
        for key, r in all_items
        if (not bool(r.get("skipped", False))) and "kappa_tail" in r
    ]
    if not items:
        raise ValueError(f"No valid drift tail-saturation cuts found for {label}.")

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


# --------------------------------------------------------------------
# Tau filter: physical-resolvability cap
# --------------------------------------------------------------------
# diagnostics.estimate_tau_spectrum fits, per neuron q,
#       log f_q(ell) = a_q - mu_bar_q * ell
# and reports tau_q = 1 / mu_bar_q.  A tau longer than the sequence
# length T is physically unresolvable; a tau longer than the largest
# fitted lag is a stricter estimator-leverage warning.  The default
# manuscript figure uses the obvious physical cap tau <= T.  Appendix
# / audit variants can use tau <= max(tau_fit_lags) to remove the
# near-singular slope regime more aggressively.
#
# We apply this only at the *analysis* layer.  The stored
# per-checkpoint CSVs are not modified, and the upstream training
# pipeline (anticollapse.sh -> run_phase_trajectory.py -> diagnostics.py)
# is untouched: re-running plot_all.sh applies the filter without
# regenerating any envelopes.
#
# The sequence length is read per-seed from that seed's cli_args.json
# (key "T"), so the cap automatically tracks whichever experimental
# configuration produced the run.  _DEFAULT_SEQ_LEN is only the
# fallback if cli_args.json is missing or omits the key; we set it to
# the Exp1 full configuration (T=1280) for safety, but in practice it
# is rarely used because every recorded seed dir carries cli_args.json.
_DEFAULT_SEQ_LEN = 1280
_SEQ_LEN_KEYS = ("T", "seq_len", "seq_length", "context_length",
                 "T_train", "train_seq_len")
_TAU_CAP_MODES = ("seq_len", "fit_lag_max")


def _seed_seq_length(seed_dir: Path, default: int = _DEFAULT_SEQ_LEN) -> int:
    """Read the training sequence length T from this seed's cli_args.json.

    The cli_args dump produced by run_phase_trajectory.py stores the
    sequence length under the key ``"T"`` (cf. ``--T`` in
    ``run_phase_trajectory.parse_args``); a few alternative names are
    tried for robustness against historical / renamed runs.  Falls back
    to ``default`` only when cli_args.json is missing or carries none
    of the recognized keys.
    """
    cli = seed_dir / "cli_args.json"
    if not cli.exists():
        return int(default)
    try:
        with open(cli) as f:
            args = json.load(f)
    except (OSError, json.JSONDecodeError):
        return int(default)
    for k in _SEQ_LEN_KEYS:
        if k in args:
            try:
                return int(args[k])
            except (TypeError, ValueError):
                continue
    return int(default)


def _read_lag_grid_max(seed_dir: Path) -> Optional[int]:
    """Read max(tau_fit_lags) from lag_grid.json when available."""
    lag_grid = seed_dir / "lag_grid.json"
    if not lag_grid.exists():
        return None
    try:
        with open(lag_grid) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("tau_fit_lags", "fit_lags"):
        vals = d.get(key)
        if isinstance(vals, list) and vals:
            try:
                return int(max(int(v) for v in vals))
            except (TypeError, ValueError):
                continue
    return None


def _read_tau_info_fit_lag_max(tau_csv: Path) -> Optional[int]:
    """Read final-checkpoint fit_lags from the tau slope-fit info JSON."""
    info = tau_csv.with_name(tau_csv.name.replace("_taus.csv", "_tau_slope_fit_info.json"))
    if not info.exists():
        return None
    try:
        with open(info) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    vals = d.get("fit_lags")
    if isinstance(vals, list) and vals:
        try:
            return int(max(int(v) for v in vals))
        except (TypeError, ValueError):
            return None
    return None


def _tau_cap(seed_dir: Path, tau_csv: Path, mode: str, factor: float) -> Tuple[float, str]:
    """Resolve the tau cap for one seed under the requested cap mode."""
    T = _seed_seq_length(seed_dir)
    if mode == "fit_lag_max":
        fit_lag_max = _read_lag_grid_max(seed_dir) or _read_tau_info_fit_lag_max(tau_csv)
        if fit_lag_max is None:
            fit_lag_max = max(1, int(T / 2))
            source = f"fallback T/2={fit_lag_max}"
        else:
            source = f"max(tau_fit_lags)={fit_lag_max}"
        return float(factor) * float(fit_lag_max), source
    return float(factor) * float(T), f"T={T}"


def final_tau_files(adamw_dir: Path, model: str = "const") -> List[Tuple[Path, Path]]:
    """Return one (seed_dir, final-checkpoint tau CSV) pair per seed directory."""
    files: List[Tuple[Path, Path]] = []
    for seed_dir in sorted(adamw_dir.glob("seed_*")):
        tau_dir = seed_dir / model / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=checkpoint_epoch)
        if candidates:
            files.append((seed_dir, candidates[-1]))
    if not files:
        raise FileNotFoundError(f"No per-seed checkpoint tau CSVs found under {adamw_dir}")
    return files


def load_final_taus(adamw_dir: Path, model: str = "const",
                    tau_cap_factor: float = 1.0,
                    tau_cap_mode: str = "seq_len") -> Optional[np.ndarray]:
    """Load and concatenate finite positive final-checkpoint tau values.

    ``tau_cap_mode="seq_len"`` drops tau values above the training
    sequence length T.  ``tau_cap_mode="fit_lag_max"`` drops tau values
    above max(tau_fit_lags), the stricter regression-leverage cap.
    """
    if tau_cap_mode not in _TAU_CAP_MODES:
        raise ValueError(f"Unknown tau_cap_mode={tau_cap_mode!r}")
    vals: List[float] = []
    n_total = 0
    n_dropped = 0
    cap_sources: List[str] = []
    for seed_dir, p in final_tau_files(adamw_dir, model=model):
        cap, source = _tau_cap(seed_dir, p, tau_cap_mode, tau_cap_factor)
        cap_sources.append(source)
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (np.isfinite(tau) and tau > 0):
                    continue
                n_total += 1
                if tau > cap:
                    n_dropped += 1
                    continue
                vals.append(tau)
    if n_total > 0:
        pct = 100.0 * n_dropped / n_total
        source_text = ", ".join(sorted(set(cap_sources)))
        print(f"    [tau filter] {adamw_dir}: dropped {n_dropped}/{n_total} "
              f"({pct:.2f}%) tau values exceeding {tau_cap_mode} cap "
              f"({source_text})")
    if not vals:
        return None
    return np.asarray(vals, dtype=float)



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


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, computed via erf (no scipy dependency)."""
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def lognormal_ccdf_in_tau(tau_grid: np.ndarray, z_mean: float, z_std: float) -> np.ndarray:
    """Analytical CCDF of tau when zeta = -log(tau) is Gaussian(z_mean, z_std^2).

    For tau > 0, P(tau > x) = P(zeta < -log x) = Phi((-log x - z_mean)/z_std).
    The result is the visual light-tail reference for the tau-CCDF panel:
    if the empirical curve tracks this reference, the apparent linearity
    on log-log axes is the well-known log-normal artifact, not a power law.
    """
    tau_grid = np.asarray(tau_grid, dtype=float)
    sigma = max(float(z_std), 1e-12)
    z_thresh = (-np.log(np.clip(tau_grid, 1e-300, None)) - float(z_mean)) / sigma
    return _phi(z_thresh)


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

    q_b, k_b, lo_b, hi_b = extract_tail_sweep(b, PATH_B_LABEL)
    q_a, k_a, lo_a, hi_a = extract_tail_sweep(a, PATH_A_LABEL)

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
def plot_time_scale_spectrum(
    path_b_dir: Path,
    path_a_dir: Path,
    outpath: Path,
    dpi: int = 400,
    tau_cap_mode: str = "seq_len",
) -> None:
    tau_b = load_final_taus(path_b_dir, tau_cap_mode=tau_cap_mode)
    tau_a = load_final_taus(path_a_dir, tau_cap_mode=tau_cap_mode)
    if tau_a is None and tau_b is None:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.axis("off")
        ax.text(
            0.5, 0.55,
            f"No resolved final $\\tau$ values under cap mode '{tau_cap_mode}'.",
            ha="center", va="center", fontsize=12,
        )
        ax.text(
            0.5, 0.38,
            "All fitted time scales exceed the regression leverage or sequence-length cap.",
            ha="center", va="center", fontsize=9, color="0.35",
        )
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {outpath} (no resolved tau under cap; dpi={dpi})")
        return
    z_b = -np.log(tau_b) if tau_b is not None else None
    z_a = -np.log(tau_a) if tau_a is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Left: p(zeta), the empirical stationary law in the SDE coordinate and
    # the natural partner of the drift diagnostic F(zeta). The Gaussian curve
    # is a same-mean/same-variance visual reference, not a goodness-of-fit
    # claim.
    z_all = np.concatenate([z for z in (z_a, z_b) if z is not None])
    lo, hi = np.quantile(z_all, [0.005, 0.995])
    pad = 0.08 * max(hi - lo, 1e-6)
    bins = np.linspace(lo - pad, hi + pad, 34)
    x_grid = np.linspace(bins[0], bins[-1], 400)

    # Store per-path (mean, std) of zeta for use as the reference in the
    # tau-CCDF panel: if zeta is approximately Gaussian then tau is
    # approximately log-normal, and the analytical light-tail reference can
    # be drawn directly from those moments.
    z_moments: Dict[str, Tuple[float, float]] = {}

    for label, z, color in [
        (PATH_A_LABEL, z_a, PATH_A_COLOR),
        (PATH_B_LABEL, z_b, PATH_B_COLOR),
    ]:
        if z is None:
            continue
        axes[0].hist(
            z, bins=bins, density=True, histtype="step",
            linewidth=2.0, color=color, label=f"{label} empirical",
        )
        mu = float(np.mean(z))
        sigma = float(np.std(z, ddof=1))
        z_moments[label] = (mu, sigma)
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
    # spectrum-to-envelope Tauberian correspondence. A power-law spectrum
    # appears as a *persistently* straight scaling window with slope -beta
    # over a wide log-log range. CAUTION: a log-normal tau population --- the
    # expected light-tailed case --- can ALSO look approximately linear on
    # log-log axes over a finite range, because the log-normal CCDF
    # curves only at the very far tail. To make that visible we overlay
    # the analytical log-normal CCDF with the same (mu_zeta, sigma_zeta) as
    # the empirical zeta on the left panel: if the empirical CCDF tracks
    # the reference, the apparent linearity is the log-normal artifact.
    for label, tau, color in [
        (PATH_A_LABEL, tau_a, PATH_A_COLOR),
        (PATH_B_LABEL, tau_b, PATH_B_COLOR),
    ]:
        if tau is None:
            continue
        xs, ys = ccdf_from_samples(tau)
        mask = ys > 0
        axes[1].plot(
            xs[mask], ys[mask], color=color, linewidth=1.8,
            label=f"{label} empirical",
        )
        q90, q99 = np.quantile(tau, [0.90, 0.99])
        axes[1].axvspan(q90, q99, color=color, alpha=0.06)

        # Analytical reference: log-normal-in-tau CCDF.
        mu, sigma = z_moments[label]
        tau_min = float(np.min(xs[mask])) if mask.any() else 1e-6
        tau_max = float(np.max(xs[mask])) if mask.any() else 1.0
        tau_ref = np.geomspace(tau_min, tau_max, 256)
        ccdf_ref = lognormal_ccdf_in_tau(tau_ref, mu, sigma)
        ref_mask = ccdf_ref > 0
        axes[1].plot(
            tau_ref[ref_mask], ccdf_ref[ref_mask],
            color=color, linestyle="--", linewidth=1.3,
            label=f"{label} log-normal reference",
        )

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Time scale $\tau$")
    axes[1].set_ylabel(r"Empirical $\mathbb{P}(T\geq\tau)$")
    axes[1].set_title(r"Final $\tau$-spectrum tail")
    axes[1].legend(loc="lower left", fontsize=8.0, framealpha=0.9)
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
    parser.add_argument(
        "--tau_cap_mode", type=str, default="seq_len",
        choices=list(_TAU_CAP_MODES),
        help=("Tau resolvability cap for spectrum plots. 'seq_len' uses "
              "tau <= T; 'fit_lag_max' uses the stricter regression-leverage "
              "cap tau <= max(tau_fit_lags)."),
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

    global PATH_A_LABEL
    PATH_A_LABEL = infer_path_a_label(args.path_a)

    # 'all' produces only the manuscript-bound figures. The forcing-
    # trajectory panel is generated only on explicit request via
    # --which forcing (kept for audit, not paper-bound).
    if args.which == "forcing":
        plot_forcing_trajectory(args.path_b, args.path_a, args.outdir / "exp1_forcing_trajectory.png", dpi=args.dpi)
    if args.which in ("all", "drift"):
        plot_drift_plateau(args.path_b, args.path_a, args.outdir / "exp1_drift_plateau.png", dpi=args.dpi)
    if args.which in ("all", "spectrum"):
        suffix = "" if args.tau_cap_mode == "seq_len" else "_fitlag"
        plot_time_scale_spectrum(
            args.path_b, args.path_a,
            args.outdir / f"exp1_time_scale_spectrum{suffix}.png",
            dpi=args.dpi,
            tau_cap_mode=args.tau_cap_mode,
        )
    if args.which == "delta_zeta":
        plot_delta_zeta(args.path_b, args.path_a, args.outdir / "exp1_delta_zeta.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
