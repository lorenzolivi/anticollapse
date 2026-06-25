#!/usr/bin/env python3
"""Access-route plots for Experiment 2.

Experiment 2 compares SharedGate and DiagGate under the same task and optimizer.
The question is not an architecture benchmark; it is whether the route to
anti-collapse is realized once the model has per-unit trainable gates.  The
paper-facing forcing plot uses the slow-mode update-increment tail estimator,
not the legacy pooled ECF/McCulloch gradient diagnostic.

This script produces the manuscript figures that overlay the selected
architectures on a common axis, each driven from the per-seed aggregates that
``main_phase_trajectory.py`` already writes under
    <EXP2_OUTDIR>/<optimizer>/aggregated/<model>/

The figures are:

  - exp2_forcing_trajectory.png : slow-mode forcing tail index (left) with the
    alpha=2 resolution band, and calibrated Gaussian-boundary p-value (right),
    one trace per architecture.
  - exp2_beta_env_trajectory.png : DiagGate time-scale-tail exponent and
    envelope power-law exponent across training, with seed-level 95% bands.
  - exp2_drift_plateau.png : far-left restoring drift kappa_tail vs q_low
    sweep, one error-bar series per architecture.
  - exp2_time_scale_spectrum.png : final log-rate density p(zeta) and
    empirical tau-CCDF, one trace per architecture. The zeta panel shows the
    stationary law modeled by the SDE and pairs naturally with the drift
    diagnostic F(zeta); the tau-CCDF panel shows the tail condition fed into
    the spectrum-to-envelope Tauberian correspondence.
  - exp2_envelope_loglog_comparison.png : corrected macroscopic envelope
    (mu0+mu1 when available) on log-log axes, with same-color dashed
    full-window power-law fits.
  - exp2_delta_zeta.png : realized dynamic range Delta zeta = zeta_q90 -
    zeta_q10 across training, one trace per architecture. This is a scalar
    width summary, not a replacement for the spectrum plot.
  - exp2_phase_summary.png : audit-only grouped bar chart of recorded
    phase-label counts per architecture. Generated only when explicitly
    requested with --which phase_summary.
  - exp2_t_cross.png : audit-only per-seed scatter of recorded crossing times.
    Generated only when explicitly requested with --which t_cross.

Per-architecture envelope audit plots are still produced by ``plot_all.sh`` via
``plot_exp1_envelopes.py`` under ``_per_arch_<arch>/``. The manuscript-facing
Exp2 envelope figure is the combined comparison generated here.

Usage:
    python plot_exp2_phase_ladder.py
    python plot_exp2_phase_ladder.py --which forcing
    python plot_exp2_phase_ladder.py \\
        --exp2_dir results/exp2_phase_full/adamw \\
        --outdir results/exp2_figures \\
        --architectures shared,diag
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
# Default manuscript comparison: SharedGate collapsed reference vs DiagGate
# route candidate. Additional architectures can still be passed manually for
# audits, but they are no longer part of the main Exp2 design.
# ============================================================
CANONICAL_ARCHS: List[str] = ["shared", "diag"]

ARCH_LABEL: Dict[str, str] = {
    "diag":   "DiagGate",
    "shared": "SharedGate",
}

ARCH_COLOR: Dict[str, str] = {
    "shared": "#2b6ab8",
    "diag":   "#3aa050",   # green
}

# Phase labels recorded by the diagnostic pipeline, in the order we want them
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


def _read_envelope_f_rows(path: Path, corrected: bool = True) -> Dict[float, float]:
    rows = _load_csv_rows(path)
    if not rows:
        return {}
    out: Dict[float, float] = {}
    for r in rows:
        try:
            ell = float(r["ell"])
            if corrected and "f_mu0_plus_mu1" in r:
                f = float(r["f_mu0_plus_mu1"])
            elif "f_mu0" in r:
                f = float(r["f_mu0"])
            elif "f_mean" in r:
                f = float(r["f_mean"])
            elif "mu_mean" in r:
                f = float(r["mu_mean"])
            elif "f" in r:
                f = float(r["f"])
            elif "mu" in r:
                f = float(r["mu"])
            else:
                f = float("nan")
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(ell) and ell > 0 and np.isfinite(f) and f > 0:
            out[ell] = f
    return out


def _load_envelope_log_curve(
    exp2_dir: Path,
    arch: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]]:
    """Load the paper-facing envelope curve for one architecture.

    Prefer seed-level first-order-corrected curves and compute mean +/- SE in
    envelope space before converting to log coordinates. If seed-level audit
    files are unavailable, fall back to the aggregate corrected curve and then
    to the canonical aggregate envelope. The return value is
    ``(ell, log_mean, log_lo, log_hi, source_label)``.
    """
    seed_maps: List[Dict[float, float]] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        audit_path = seed_dir / arch / f"{arch}_envelope_audit.csv"
        if audit_path.exists():
            vals = _read_envelope_f_rows(audit_path, corrected=True)
            if vals:
                seed_maps.append(vals)
    if seed_maps:
        common_ells = sorted(set.intersection(*(set(m.keys()) for m in seed_maps)))
        if len(common_ells) >= 6:
            arr = np.asarray([[m[e] for e in common_ells] for m in seed_maps], dtype=float)
            mean_f = np.nanmean(arr, axis=0)
            se_f = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean_f)
            lo_f = np.maximum(mean_f - se_f, np.finfo(float).tiny)
            hi_f = np.maximum(mean_f + se_f, np.finfo(float).tiny)
            mask = np.isfinite(mean_f) & (mean_f > 0) & np.isfinite(lo_f) & np.isfinite(hi_f)
            if int(np.count_nonzero(mask)) >= 6:
                ell_arr = np.asarray(common_ells, dtype=float)[mask]
                return (
                    ell_arr,
                    np.log(mean_f[mask]),
                    np.log(lo_f[mask]),
                    np.log(hi_f[mask]),
                    rf"$\mu_0+\mu_1$ mean across {arr.shape[0]} seeds",
                )

    agg = _agg_dir(exp2_dir, arch)
    audit_path = agg / f"{arch}_envelope_audit.csv"
    if audit_path.exists():
        vals = _read_envelope_f_rows(audit_path, corrected=True)
        if len(vals) >= 6:
            ell = np.asarray(sorted(vals.keys()), dtype=float)
            f = np.asarray([vals[e] for e in ell], dtype=float)
            return ell, np.log(f), None, None, r"$\mu_0+\mu_1$ aggregate"

    env_path = agg / f"{arch}_envelope.csv"
    vals = _read_envelope_f_rows(env_path, corrected=False)
    if len(vals) < 6:
        return None
    ell = np.asarray(sorted(vals.keys()), dtype=float)
    f = np.asarray([vals[e] for e in ell], dtype=float)
    return ell, np.log(f), None, None, r"$\mu_0$ aggregate"


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
    e.g. ``drift_shared/`` or ``drift_diag/``. (For exp1's ConstGate the
    historical pre-rename name was ``drift_const`` or ``drift_constgate``, but
    exp2 standardizes on ``drift_<arch>``.)
    """
    return exp2_dir / f"drift_{arch}" / "tail_saturation.json"


def _checkpoint_epoch(p: Path) -> int:
    """Extract epoch from ckpt_XXXX_taus.csv; return -1 if malformed."""
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _read_envelope_lags(seed_dir: Path) -> np.ndarray:
    """Read the lag grid used for envelope plots, with a conservative fallback."""
    lag_grid = seed_dir / "lag_grid.json"
    if lag_grid.exists():
        try:
            with open(lag_grid) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            d = {}
        for key in ("ells", "envelope_lags", "lags"):
            vals = d.get(key)
            if isinstance(vals, list) and len(vals) >= 6:
                out = np.asarray(vals, dtype=float)
                out = out[np.isfinite(out) & (out > 0)]
                if out.size >= 6:
                    return np.unique(out)
    return np.unique(np.linspace(4, 384, 128, dtype=float))


def _load_tau_csv(path: Path) -> np.ndarray:
    vals: List[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                tau = float(row["tau"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(tau) and tau > 0:
                vals.append(tau)
    return np.asarray(vals, dtype=float)


def _power_fit_beta_from_envelope(ell: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
    """Fit f(ell) ~= C ell^{-beta}; return (beta, R^2)."""
    ell = np.asarray(ell, dtype=float)
    f = np.asarray(f, dtype=float)
    mask = np.isfinite(ell) & (ell > 0) & np.isfinite(f) & (f > 0)
    if int(np.count_nonzero(mask)) < 6:
        return float("nan"), float("nan")
    x = np.log(ell[mask])
    y = np.log(f[mask])
    coef = np.polyfit(x, y, 1)
    yhat = np.polyval(coef, x)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / denom if denom > 0 else float("nan")
    return float(-coef[0]), float(r2)


def _zero_order_envelope_beta_from_taus(tau: np.ndarray, ell: np.ndarray) -> Tuple[float, float]:
    """Reconstruct the zero-order envelope from tau samples and fit a power law.

    Checkpoint-level first-order corrected envelopes are not saved, so this
    trajectory is a zero-order consistency diagnostic. The final envelope
    comparison figure still uses the corrected mu0+mu1 curve when available.
    """
    tau = np.asarray(tau, dtype=float)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    if tau.size < 20:
        return float("nan"), float("nan")
    ell = np.asarray(ell, dtype=float)
    ell = ell[np.isfinite(ell) & (ell > 0)]
    if ell.size < 6:
        return float("nan"), float("nan")
    # Chunked evaluation avoids allocating a huge ell-by-unit matrix if a
    # future run uses many more units.
    f_sum = np.zeros_like(ell, dtype=float)
    chunk = 1024
    for i in range(0, tau.size, chunk):
        t = tau[i:i + chunk]
        f_sum += np.exp(-ell[:, None] / t[None, :]).sum(axis=1)
    f = f_sum / float(tau.size)
    return _power_fit_beta_from_envelope(ell, f)


def _tcrit_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for small seed counts."""
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
    }
    if n <= 1:
        return 0.0
    return table.get(n, 1.96)


def _summarize_epoch_values(values: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs: List[int] = []
    means: List[float] = []
    los: List[float] = []
    his: List[float] = []
    ns: List[int] = []
    for ep in sorted(values):
        arr = np.asarray([v for v in values[ep] if np.isfinite(v)], dtype=float)
        n = int(arr.size)
        if n == 0:
            continue
        mean = float(np.mean(arr))
        if n > 1:
            half_width = _tcrit_95(n) * float(np.std(arr, ddof=1)) / np.sqrt(n)
        else:
            half_width = 0.0
        epochs.append(int(ep))
        means.append(mean)
        los.append(mean - half_width)
        his.append(mean + half_width)
        ns.append(n)
    return (
        np.asarray(epochs, dtype=int),
        np.asarray(means, dtype=float),
        np.asarray(los, dtype=float),
        np.asarray(his, dtype=float),
        np.asarray(ns, dtype=int),
    )


def _diag_beta_tail_by_epoch(exp2_dir: Path, arch: str = "diag") -> Dict[int, List[float]]:
    values: Dict[int, List[float]] = {}
    for traj in sorted(exp2_dir.glob(f"seed_*/{arch}/phase_trajectory.csv")):
        rows = _load_csv_rows(traj)
        if not rows:
            continue
        for row in rows:
            try:
                ep = int(row["epoch"])
                beta = float(row["beta_hat"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(beta):
                values.setdefault(ep, []).append(beta)
    return values


def _diag_envelope_beta_by_epoch(exp2_dir: Path, arch: str = "diag") -> Dict[int, List[float]]:
    values: Dict[int, List[float]] = {}
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        tau_dir = seed_dir / arch / "checkpoint_taus"
        if not tau_dir.exists():
            continue
        ell = _read_envelope_lags(seed_dir)
        for tau_csv in sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch):
            ep = _checkpoint_epoch(tau_csv)
            if ep < 0:
                continue
            tau = _load_tau_csv(tau_csv)
            beta, _r2 = _zero_order_envelope_beta_from_taus(tau, ell)
            if np.isfinite(beta):
                values.setdefault(ep, []).append(beta)
    return values


# --------------------------------------------------------------------
# Tau filter: physical-resolvability cap
# --------------------------------------------------------------------
# diagnostics.estimate_tau_spectrum fits log f_q(ell) = a_q - mu_bar_q * ell
# and reports tau_q = 1 / mu_bar_q.  The default manuscript figure uses
# the obvious physical cap tau <= T.  Appendix / audit variants can use
# the stricter regression-leverage cap tau <= max(tau_fit_lags), which
# removes near-flat slope reciprocals from the tau-CCDF panel.
#
# Sequence length is read per-seed from cli_args.json (key "T", matching
# the ``--T`` flag of run_phase_trajectory.py), so the cap adapts
# automatically to whichever Exp2 ladder rung produced the run.
# _DEFAULT_SEQ_LEN is just the fallback when cli_args.json is missing
# or omits the key; set to the Exp1 full configuration (T=1280).
_DEFAULT_SEQ_LEN = 1280
_SEQ_LEN_KEYS = ("T", "seq_len", "seq_length", "context_length",
                 "T_train", "train_seq_len")
_TAU_CAP_MODES = ("seq_len", "fit_lag_max")


def _seed_seq_length(seed_dir: Path, default: int = _DEFAULT_SEQ_LEN) -> int:
    """Read training sequence length T from seed's cli_args.json.

    Uses the cli_args dump written by run_phase_trajectory.py; the
    key is "T" (mirroring the ``--T`` argparse flag).
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


def _final_tau_files(exp2_dir: Path, arch: str) -> List[Tuple[Path, Path]]:
    """Return one (seed_dir, final-checkpoint tau CSV) pair per seed."""
    files: List[Tuple[Path, Path]] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        tau_dir = seed_dir / arch / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
        if candidates:
            files.append((seed_dir, candidates[-1]))
    return files


def _load_final_taus(exp2_dir: Path, arch: str,
                     tau_cap_factor: float = 1.0,
                     tau_cap_mode: str = "seq_len") -> Optional[np.ndarray]:
    """Load finite positive final-checkpoint tau values for one architecture.

    ``tau_cap_mode="seq_len"`` drops tau values above T.  The stricter
    ``tau_cap_mode="fit_lag_max"`` drops values above max(tau_fit_lags).
    """
    if tau_cap_mode not in _TAU_CAP_MODES:
        raise ValueError(f"Unknown tau_cap_mode={tau_cap_mode!r}")
    vals: List[float] = []
    n_total = 0
    n_dropped = 0
    cap_sources: List[str] = []
    for seed_dir, p in _final_tau_files(exp2_dir, arch):
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
    if not vals:
        return None
    if n_total > 0 and n_dropped > 0:
        pct = 100.0 * n_dropped / n_total
        source_text = ", ".join(sorted(set(cap_sources)))
        print(f"    [tau filter] {exp2_dir}/{arch}: dropped {n_dropped}/"
              f"{n_total} ({pct:.2f}%) tau values exceeding {tau_cap_mode} "
              f"cap ({source_text})")
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
# Figure 1: slow-mode forcing trajectory across training.
# Left: calibrated effective alpha on slow-mode update increments, with the
# alpha=2 reference band. Right: one-sided calibrated p-value for the Gaussian
# boundary test.
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

        x, y, s = _extract_epoch_series(rows, "forcing_alpha_hat_mean", "forcing_alpha_hat_se")
        axes[0].plot(x, y, color=color, label=label, linewidth=1.5)
        axes[0].fill_between(x, y - s, y + s, color=color, alpha=0.18)

        xb_lo, band_lo, _ = _extract_epoch_series(rows, "forcing_alpha2_band_lo_mean", "forcing_alpha2_band_lo_se")
        xb_hi, band_hi, _ = _extract_epoch_series(rows, "forcing_alpha2_band_hi_mean", "forcing_alpha2_band_hi_se")
        if len(xb_lo) and len(xb_hi):
            axes[0].fill_between(xb_lo, band_lo, band_hi, color=color, alpha=0.08)

        x2, y2, s2 = _extract_epoch_series(
            rows,
            "forcing_gaussian_p_value_mean",
            "forcing_gaussian_p_value_se",
        )
        if len(x2):
            axes[1].plot(x2, y2, color=color, label=label, linewidth=1.5)
            lo = np.maximum(y2 - s2, 1e-4)
            hi = np.minimum(y2 + s2, 1.0)
            axes[1].fill_between(x2, lo, hi, color=color, alpha=0.18)

    axes[0].axhline(2.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Calibrated $\hat\alpha_{\mathrm{eff}}$")
    axes[0].set_ylim(1.85, 2.03)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    axes[0].set_title("Slow-mode forcing: tail-index effect size")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Matched Gaussian-tail p-value")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-3, 1.05)
    axes[1].axhline(0.05, color="gray", linewidth=0.7, linestyle="--", alpha=0.7)
    axes[1].legend(loc="lower right", fontsize=9, framealpha=0.9)
    axes[1].set_title("Calibrated one-sided Gaussian test")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 2: spectral/envelope exponent consistency across training.
#
# The paper uses this as a finite-window consistency diagnostic: beta_tail is
# estimated from the tau-CCDF, while beta_env is obtained by reconstructing the
# zero-order envelope from checkpoint tau samples and fitting a power law. It is
# deliberately DiagGate-focused; SharedGate's fitted exponent is not interpreted
# as an anti-collapsed-regime parameter.
# ============================================================
def plot_beta_env_trajectory(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
    min_epoch: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    arch = "diag" if "diag" in architectures else architectures[-1]
    color = ARCH_COLOR.get(arch, "#3aa050")

    tail_values = _diag_beta_tail_by_epoch(exp2_dir, arch)
    env_values = _diag_envelope_beta_by_epoch(exp2_dir, arch)

    x_tail, y_tail, lo_tail, hi_tail, _n_tail = _summarize_epoch_values(tail_values)
    x_env, y_env, lo_env, hi_env, _n_env = _summarize_epoch_values(env_values)

    mask_tail = x_tail >= min_epoch
    mask_env = x_env >= min_epoch
    if int(np.count_nonzero(mask_tail)) == 0:
        print(f"  [beta_env] skip {arch}: beta_tail trajectory missing")
    else:
        ax.plot(
            x_tail[mask_tail], y_tail[mask_tail],
            color="#1f4e79", linewidth=1.6, linestyle="-",
            label=r"$\hat\beta_{\mathrm{tail}}$ from $\tau$-CCDF",
        )
        ax.fill_between(
            x_tail[mask_tail], lo_tail[mask_tail], hi_tail[mask_tail],
            color="#1f4e79", alpha=0.16, linewidth=0,
        )

    if int(np.count_nonzero(mask_env)) == 0:
        print(f"  [beta_env] skip {arch}: checkpoint tau envelopes missing")
    else:
        ax.plot(
            x_env[mask_env], y_env[mask_env],
            color=color, linewidth=1.8, linestyle="--",
            label=r"$\hat\beta_{\mathrm{env}}$ from envelope fit",
        )
        ax.fill_between(
            x_env[mask_env], lo_env[mask_env], hi_env[mask_env],
            color=color, alpha=0.18, linewidth=0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Estimated exponent $\beta$")
    ax.set_title("DiagGate exponent consistency across training")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 3: far-left restoring drift kappa_tail vs q_low sweep.
#
# Smoke profiles can have too few far-tail samples at the deepest q cuts.
# The drift diagnostic records those entries as skipped; the plotter keeps the
# valid cuts and reports skipped ones rather than treating them as failures.
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
        skipped = [
            f"{float(r['q_low']):.3f}"
            for _, r in items
            if bool(r.get("skipped", False)) or "kappa_tail" not in r
        ]
        items = [
            (name, r)
            for name, r in items
            if (not bool(r.get("skipped", False))) and "kappa_tail" in r
        ]
        if skipped:
            print(f"  [drift] {arch}: skipped q_low={', '.join(skipped)} (insufficient usable tail samples)")
        if not items:
            print(f"  [drift] skip {arch}: no valid q_low cuts")
            continue
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
    ax.set_ylabel(r"Far-left restoring drift $\hat\kappa_{\mathrm{tail}}$ (90% CI)")
    ax.set_title("Far-left restoring drift across tail cuts")
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
    tau_cap_mode: str = "seq_len",
) -> None:
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for arch in architectures:
        tau = _load_final_taus(exp2_dir, arch, tau_cap_mode=tau_cap_mode)
        if tau is None:
            print(f"  [spectrum] skip {arch}: no resolved final tau values under cap")
            continue
        spectra[arch] = (tau, -np.log(tau))

    if not spectra:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.axis("off")
        ax.text(
            0.5, 0.55,
            f"No resolved final $\\tau$ values under cap mode '{tau_cap_mode}'.",
            ha="center", va="center", fontsize=12,
        )
        ax.text(
            0.5, 0.38,
            "This usually means all fitted time scales exceed the regression "
            "leverage window or sequence-length cap.",
            ha="center", va="center", fontsize=9, color="0.35",
        )
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {outpath} (no resolved tau under cap; dpi={dpi})")
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
# Figure 4b: corrected macroscopic envelope comparison.
#
# Paper-facing envelope view: compare SharedGate and DiagGate directly on
# log-log axes. Solid lines use the first-order corrected envelope mu0+mu1
# when the audit CSV is available; dashed same-color lines are full-window
# power-law fits. The legend is architecture-only by design; fit diagnostics
# belong in the text rather than in figure labels.
# ============================================================
def plot_envelope_loglog_comparison(
    exp2_dir: Path,
    architectures: Sequence[str],
    outpath: Path,
    dpi: int = 400,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    any_curve = False

    for arch in architectures:
        loaded = _load_envelope_log_curve(exp2_dir, arch)
        if loaded is None:
            print(f"  [envelope] skip {arch}: no usable aggregate envelope")
            continue
        ell, log_f, log_lo, log_hi, source_label = loaded
        mask = np.isfinite(ell) & (ell > 0) & np.isfinite(log_f)
        ell = ell[mask]
        log_f = log_f[mask]
        if log_lo is not None and log_hi is not None:
            log_lo = log_lo[mask]
            log_hi = log_hi[mask]
        if ell.size < 6:
            print(f"  [envelope] skip {arch}: too few finite envelope points")
            continue

        order = np.argsort(ell)
        ell = ell[order]
        log_f = log_f[order]
        if log_lo is not None and log_hi is not None:
            log_lo = log_lo[order]
            log_hi = log_hi[order]
        log_ell = np.log(ell)

        color = ARCH_COLOR.get(arch, None)
        label = ARCH_LABEL.get(arch, arch)
        if log_lo is not None and log_hi is not None:
            band_mask = np.isfinite(log_lo) & np.isfinite(log_hi)
            if np.any(band_mask):
                ax.fill_between(
                    log_ell[band_mask],
                    log_lo[band_mask],
                    log_hi[band_mask],
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    label="_nolegend_",
                )
        ax.plot(
            log_ell,
            log_f,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            markevery=max(1, ell.size // 32),
            label=label,
        )

        slope, intercept = np.polyfit(log_ell, log_f, deg=1)
        fit = intercept + slope * log_ell
        beta = max(0.0, -float(slope))
        ss_res = float(np.sum((log_f - fit) ** 2))
        ss_tot = float(np.sum((log_f - float(np.mean(log_f))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.plot(
            log_ell,
            fit,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.80,
            label="_nolegend_",
        )
        print(
            f"  [envelope] {label}: source={source_label}, "
            f"full-window power beta={beta:.4g}, R2={r2:.4g}, n={ell.size}"
        )
        any_curve = True

    if not any_curve:
        ax.axis("off")
        ax.text(0.5, 0.5, "No usable aggregate envelope curves found.",
                ha="center", va="center", fontsize=11)
    else:
        ax.set_xlabel(r"$\log \ell$")
        ax.set_ylabel(r"$\log \hat f(\ell)$")
        ax.set_title(r"Corrected macroscopic envelope on log-log axes")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, framealpha=0.92)

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
    ax.set_title(r"Realized dynamic range across training")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


# ============================================================
# Figure 6: recorded phase-label summary.
#
# Reads <model>_final_phase.json's phase_counts dict for each architecture
# and renders a grouped bar chart of fractional counts (collapsed / canonical
# AC / robust AC), normalized by n_seeds. This is an audit plot; it is not
# used as paper-bound evidence for the phase interpretation.
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
    ax.set_title("Final recorded phase label")
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
# We render it as a strip plot: x = architecture, y =
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
        help=("Comma-separated architecture short names. Default: 'shared,diag'. "
              "Aliases 'diaggate' "
              "and 'sharedgate' are accepted."),
    )
    parser.add_argument(
        "--dpi", type=int, default=400,
        help="DPI for the saved PNG files (default: 400).",
    )
    parser.add_argument(
        "--which", type=str, default="all",
        choices=["all", "forcing", "beta_env", "drift", "spectrum", "envelope", "delta_zeta",
                 "phase_summary", "t_cross"],
        help="Which figure(s) to produce (default: all).",
    )
    parser.add_argument(
        "--tau_cap_mode", type=str, default="seq_len",
        choices=list(_TAU_CAP_MODES),
        help=("Tau resolvability cap for spectrum plots. 'seq_len' uses "
              "tau <= T; 'fit_lag_max' uses the stricter regression-leverage "
              "cap tau <= max(tau_fit_lags)."),
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
    print(f"tau_cap_mode   : {args.tau_cap_mode}")
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
        suffix = "" if args.tau_cap_mode == "seq_len" else "_fitlag"
        plot_time_scale_spectrum(
            args.exp2_dir, architectures,
            args.outdir / f"exp2_time_scale_spectrum{suffix}.png",
            dpi=args.dpi,
            tau_cap_mode=args.tau_cap_mode,
        )
    if args.which in ("all", "envelope"):
        plot_envelope_loglog_comparison(
            args.exp2_dir, architectures,
            args.outdir / "exp2_envelope_loglog_comparison.png",
            dpi=args.dpi,
        )
    if args.which in ("all", "delta_zeta"):
        plot_delta_zeta(
            args.exp2_dir, architectures,
            args.outdir / "exp2_delta_zeta.png", dpi=args.dpi,
        )
    if args.which == "phase_summary":
        plot_phase_summary(
            args.exp2_dir, architectures,
            args.outdir / "exp2_phase_summary.png", dpi=args.dpi,
        )
    if args.which == "t_cross":
        plot_t_cross(
            args.exp2_dir, architectures,
            args.outdir / "exp2_t_cross.png", dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
