#!/usr/bin/env python3
"""Write results/exp2_phase_<profile>_results_summary.md.

This is the exp2 counterpart of ``results/exp1_results_summary.md``. It
consumes the per-architecture aggregates that ``main_phase_trajectory.py``
writes for
Experiment 2 (SharedGate collapsed reference vs DiagGate route candidate) and emits a
markdown summary for audit and manuscript drafting.

Per architecture we report:
  - recorded phase-label counts (collapsed / canonical AC / robust AC),
  - majority phase label from the full-trajectory diagnostic,
  - envelope exponent β̂_env (mean ± SE across seeds) at the final epoch,
  - envelope tail β̂ and bootstrap percentile interval from <arch>_final_phase.json,
  - final time-scale spectrum quantiles in both ζ = -log τ and τ,
    with Δζ = ζ_q90 − ζ_q10 retained as a scalar width summary,
  - far-left drift plateau κ_tail at the primary q_low (0.10) with 90% CI,
  - per-seed threshold-crossing epoch t_cross (observed vs right-censored),
  - update-space slow-mode forcing tail index at the final epoch,
  - drift-subtracted ζ-residual tail diagnostics in the far-left slice,
  - first-order-corrected envelope power-fit diagnostics.

The generated overview is intentionally lightweight: it summarizes the
recorded labels and raw measurements, but the manuscript interpretation is
made after inspecting the spectra, envelopes, drift estimates, and forcing
measurements.

Usage:
    python write_exp2_summary.py
    python write_exp2_summary.py \\
        --exp2_dir results/exp2_phase_full/adamw \\
        --architectures shared,diag \\
        --output results/exp2_phase_full_results_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Architecture human-readable labels (matches plot_exp2_phase_ladder.py).
ARCH_LABEL: Dict[str, str] = {
    "diag":   "DiagGate",
    "shared": "SharedGate",
}

# Phase-label short names and ordering.
PHASE_ORDER: List[str] = [
    "collapsed",
    "concentrated anti-collapse",
    "anti-collapse",
]
PHASE_SHORT: Dict[str, str] = {
    "collapsed":                   "collapsed",
    "concentrated anti-collapse":  "canonical AC",
    "anti-collapse":               "robust AC",
}

_DEFAULT_SEQ_LEN = 1280
_SEQ_LEN_KEYS = ("T", "seq_len", "seq_length", "context_length",
                 "T_train", "train_seq_len")
_TAU_CAP_MODES = ("raw", "seq_len", "fit_lag_max")


# ----- IO helpers -----
def _load_json(p: Path) -> Optional[Dict]:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_last_row(p: Path) -> Optional[Dict[str, str]]:
    """Return the last row of a CSV (final epoch's aggregate)."""
    if not p.exists():
        return None
    with open(p) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1] if rows else None


def _fmt(v: Optional[float], digits: int = 3) -> str:
    """Format a float or ``-`` for missing/non-finite values."""
    try:
        f = float(v)
        if not (f == f) or f in (float("inf"), float("-inf")):
            return "—"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_p_value(value: Optional[float], floor: Optional[float] = None,
                 at_floor: Optional[float] = None, digits: int = 3) -> str:
    """Format p-values, displaying Monte-Carlo floor hits as inequalities."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(p):
        return "—"
    try:
        floor_v = float(floor)
    except (TypeError, ValueError):
        floor_v = float("nan")
    try:
        at_floor_v = float(at_floor)
    except (TypeError, ValueError):
        at_floor_v = 0.0
    if math.isfinite(floor_v) and (at_floor_v >= 0.5 or p <= floor_v * (1.0 + 1e-12)):
        return f"≤{floor_v:.{digits}f}"
    return f"{p:.{digits}f}"


def _fmt_int(v: Optional[float]) -> str:
    """Format a numeric count or ``-`` for missing values."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "—"


def _safe_float(v: object) -> Optional[float]:
    """Return a finite float, or None for missing/non-finite values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _quantile(xs: List[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile for finite samples."""
    vals = sorted(x for x in xs if math.isfinite(x))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return (1.0 - w) * vals[lo] + w * vals[hi]


def _checkpoint_epoch(p: Path) -> int:
    """Extract the epoch number from ckpt_<epoch>_taus.csv."""
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _seed_seq_length(seed_dir: Path, default: int = _DEFAULT_SEQ_LEN) -> int:
    """Read the sequence length T from a seed's CLI manifest."""
    cli = seed_dir / "cli_args.json"
    if not cli.exists():
        return int(default)
    try:
        with open(cli) as f:
            args = json.load(f)
    except (OSError, json.JSONDecodeError):
        return int(default)
    for key in _SEQ_LEN_KEYS:
        if key in args:
            try:
                return int(args[key])
            except (TypeError, ValueError):
                continue
    return int(default)


def _read_lag_grid_max(seed_dir: Path) -> Optional[int]:
    """Read max(tau_fit_lags) from the seed-level lag grid, if present."""
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
    """Read max(fit_lags) from the final tau-slope fit metadata."""
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


def _tau_cap(seed_dir: Path, tau_csv: Path, mode: str) -> Tuple[float, str]:
    """Return the finite-resolution tau cap used for summary quantiles."""
    if mode == "raw":
        return float("inf"), "raw"
    T = _seed_seq_length(seed_dir)
    if mode == "fit_lag_max":
        fit_lag_max = _read_lag_grid_max(seed_dir) or _read_tau_info_fit_lag_max(tau_csv)
        if fit_lag_max is None:
            fit_lag_max = max(1, int(T / 2))
            source = f"fallback T/2={fit_lag_max}"
        else:
            source = f"max(tau_fit_lags)={fit_lag_max}"
        return float(fit_lag_max), source
    return float(T), f"T={T}"


def _load_final_tau_values(exp2_dir: Path, arch: str, tau_cap_mode: str = "seq_len") -> List[float]:
    """Load finite positive τ values from each seed's final checkpoint CSV."""
    if tau_cap_mode not in _TAU_CAP_MODES:
        raise ValueError(f"Unknown tau_cap_mode={tau_cap_mode!r}")
    vals: List[float] = []
    n_total = 0
    n_dropped = 0
    cap_sources: List[str] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        tau_dir = seed_dir / arch / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
        if not candidates:
            continue
        cap, source = _tau_cap(seed_dir, candidates[-1], tau_cap_mode)
        cap_sources.append(source)
        with open(candidates[-1]) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(tau) and tau > 0.0:
                    n_total += 1
                    if tau > cap:
                        n_dropped += 1
                        continue
                    vals.append(tau)
    if n_total > 0 and n_dropped > 0:
        pct = 100.0 * n_dropped / n_total
        source_text = ", ".join(sorted(set(cap_sources)))
        print(f"    [tau filter] {exp2_dir}/{arch}: dropped {n_dropped}/"
              f"{n_total} ({pct:.2f}%) tau values exceeding {tau_cap_mode} "
              f"cap ({source_text})")
    return vals


def _read_corrected_envelope_map(path: Path) -> Dict[float, float]:
    """Read an envelope curve, preferring the first-order-corrected column."""
    if not path.exists():
        return {}
    out: Dict[float, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ell = _safe_float(row.get("ell"))
            f_val = _safe_float(row.get("f_mu0_plus_mu1"))
            if f_val is None:
                f_val = _safe_float(row.get("f_mu0"))
            if f_val is None:
                f_val = _safe_float(row.get("f_mean"))
            if ell is not None and ell > 0.0 and f_val is not None and f_val > 0.0:
                out[ell] = f_val
    return out


def _simple_linear_fit(xs: List[float], ys: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    """Return slope and R^2 for a simple linear fit."""
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 2:
        return None, None, n
    x_mean = sum(x for x, _ in pairs) / n
    y_mean = sum(y for _, y in pairs) / n
    sxx = sum((x - x_mean) ** 2 for x, _ in pairs)
    if sxx <= 0.0:
        return None, None, n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pairs)
    ss_tot = sum((y - y_mean) ** 2 for _, y in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None
    return slope, r2, n


def _load_corrected_envelope_power_summary(exp2_dir: Path, arch: str) -> Dict[str, object]:
    """Mirror the paper-facing corrected-envelope curve used by the plotter."""
    seed_maps: List[Dict[float, float]] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        vals = _read_corrected_envelope_map(seed_dir / arch / f"{arch}_envelope_audit.csv")
        if vals:
            seed_maps.append(vals)

    source = ""
    curve: Dict[float, float] = {}
    n_seed_curves = 0
    if seed_maps:
        common_ells = sorted(set.intersection(*(set(m.keys()) for m in seed_maps)))
        if len(common_ells) >= 6:
            n_seed_curves = len(seed_maps)
            curve = {
                ell: sum(m[ell] for m in seed_maps) / len(seed_maps)
                for ell in common_ells
            }
            source = f"mu0+mu1 mean across {len(seed_maps)} seeds"

    if not curve:
        agg = exp2_dir / "aggregated" / arch
        curve = _read_corrected_envelope_map(agg / f"{arch}_envelope_audit.csv")
        if curve:
            source = "mu0+mu1 aggregate"
        else:
            curve = _read_corrected_envelope_map(agg / f"{arch}_envelope.csv")
            source = "mu0 aggregate" if curve else "missing"

    ell = sorted(e for e, v in curve.items() if e > 0.0 and v > 0.0)
    log_ell = [math.log(e) for e in ell]
    log_f = [math.log(curve[e]) for e in ell]
    slope, r2, n = _simple_linear_fit(log_ell, log_f)
    beta = max(0.0, -slope) if slope is not None else None

    corr_vals: List[float] = []
    audit = exp2_dir / "aggregated" / arch / f"{arch}_envelope_audit.csv"
    if audit.exists():
        with open(audit) as f:
            for row in csv.DictReader(f):
                v = _safe_float(row.get("corr_ratio_mu1_over_mu0"))
                if v is not None:
                    corr_vals.append(v)

    return {
        "corrected_envelope_source": source,
        "corrected_envelope_n_seed_curves": n_seed_curves,
        "corrected_envelope_power_beta": beta,
        "corrected_envelope_power_r2": r2,
        "corrected_envelope_power_n": n,
        "corr_ratio_median": _quantile(corr_vals, 0.50) if corr_vals else None,
        "corr_ratio_max": max(corr_vals) if corr_vals else None,
    }


# ----- Per-architecture record -----
def collect_arch_record(
    exp2_dir: Path, arch: str, tau_cap_mode: str = "seq_len",
) -> Dict[str, object]:
    """Collect the per-architecture summary record from the aggregates."""
    agg_dir = exp2_dir / "aggregated" / arch
    rec: Dict[str, object] = {"arch": arch}

    # Phase labels and final-epoch envelope numbers from <arch>_final_phase.json
    fp = _load_json(agg_dir / f"{arch}_final_phase.json")
    rec["final_phase_present"] = fp is not None
    if fp is not None:
        rec["n_seeds"] = int(fp.get("n_seeds", 0))
        rec["phase_counts"] = dict(fp.get("phase_counts", {}))
        rec["majority_phase_label"] = str(fp.get("majority_phase_label", "—"))
        rec["majority_phase_fraction"] = float(fp.get("phase_majority_fraction", float("nan")))
        rec["tail_beta_hat"] = fp.get("tail_beta_hat")
        rec["tail_beta_r2"] = fp.get("tail_beta_r2")
        rec["boot_beta_median"] = fp.get("boot_beta_median")
        rec["boot_beta_lo"] = fp.get("boot_beta_lo")
        rec["boot_beta_hi"] = fp.get("boot_beta_hi")
        rec["delta_zeta"] = fp.get("delta_zeta")
        rec["zeta_q10"] = fp.get("zeta_q10")
        rec["zeta_q90"] = fp.get("zeta_q90")
        rec["envelope_winner"] = fp.get("envelope_winner")
        rec["crossover_mode"] = fp.get("crossover_diagnostic", {}).get("majority_mode")
    else:
        rec["n_seeds"] = 0
        rec["phase_counts"] = {}
        rec["majority_phase_label"] = "—"
        rec["majority_phase_fraction"] = float("nan")

    # Final-epoch beta_env and slow-mode forcing readout from the aggregated
    # phase trajectory. Legacy pooled alpha fields may still exist in the CSV,
    # but they are no longer the paper-facing forcing diagnostic.
    last = _load_last_row(agg_dir / "phase_trajectory_aggregated.csv")
    if last is not None:
        rec["beta_env_final_mean"] = last.get("beta_env_mean")
        rec["beta_env_final_se"] = last.get("beta_env_se")
        rec["beta_env_r2_final_mean"] = last.get("beta_env_r2_mean")
        rec["forcing_alpha_final"] = last.get("forcing_alpha_hat_mean")
        rec["forcing_alpha_final_se"] = last.get("forcing_alpha_hat_se")
        rec["forcing_alpha_reliable_final"] = last.get("forcing_alpha_reliable_mean")
        rec["forcing_gaussian_p_value_final"] = last.get("forcing_gaussian_p_value_mean")
        rec["forcing_gaussian_p_value_floor_final"] = last.get("forcing_gaussian_p_value_floor_mean")
        rec["forcing_gaussian_p_value_at_floor_final"] = last.get("forcing_gaussian_p_value_at_floor_mean")
        rec["forcing_gaussian_reject_final"] = last.get("forcing_gaussian_reject_mean")
        rec["forcing_detectably_heavy_final"] = last.get("forcing_alpha_detectably_heavy_mean")
        rec["forcing_substantively_heavy_final"] = last.get("forcing_alpha_substantively_heavy_mean")
        rec["forcing_resolvably_heavy_final"] = last.get("forcing_alpha_resolvably_heavy_mean")
        rec["forcing_alpha_substantive_threshold_final"] = last.get("forcing_alpha_substantive_threshold_mean")
        rec["forcing_alpha2_band_lo_final"] = last.get("forcing_alpha2_band_lo_mean")
        rec["forcing_alpha2_band_hi_final"] = last.get("forcing_alpha2_band_hi_mean")
        rec["forcing_heavy_fraction_final"] = last.get("forcing_heavy_fraction_mean")
        rec["final_epoch"] = last.get("epoch")
    else:
        rec["beta_env_final_mean"] = None
        rec["beta_env_final_se"] = None
        rec["beta_env_r2_final_mean"] = None
        rec["forcing_alpha_final"] = None
        rec["forcing_alpha_final_se"] = None
        rec["forcing_alpha_reliable_final"] = None
        rec["forcing_gaussian_p_value_final"] = None
        rec["forcing_gaussian_p_value_floor_final"] = None
        rec["forcing_gaussian_p_value_at_floor_final"] = None
        rec["forcing_gaussian_reject_final"] = None
        rec["forcing_detectably_heavy_final"] = None
        rec["forcing_substantively_heavy_final"] = None
        rec["forcing_resolvably_heavy_final"] = None
        rec["forcing_alpha_substantive_threshold_final"] = None
        rec["forcing_alpha2_band_lo_final"] = None
        rec["forcing_alpha2_band_hi_final"] = None
        rec["forcing_heavy_fraction_final"] = None
        rec["final_epoch"] = None

    # Drift-subtracted ζ-residual forcing diagnostic. This is the
    # paper-facing forcing view: increments in the modeled log-rate coordinate,
    # after subtracting a nonparametric conditional drift estimate.
    residual_dir = exp2_dir / f"zeta_residual_forcing_{arch}"
    residual_summary = _load_last_row(residual_dir / "zeta_residual_forcing_summary.csv")
    if residual_summary is not None:
        rec["zeta_resid_n_tail"] = residual_summary.get("n_tail_residual_samples_used")
        rec["zeta_resid_tail_zeta_cut"] = residual_summary.get("tail_zeta_cut")
        rec["zeta_resid_alpha_eff"] = residual_summary.get("alpha_eff")
        rec["zeta_resid_alpha_eff_lo"] = residual_summary.get("alpha_eff_lo")
        rec["zeta_resid_alpha_eff_hi"] = residual_summary.get("alpha_eff_hi")
        rec["zeta_resid_xi_hat"] = residual_summary.get("xi_hat")
        rec["zeta_resid_gaussian_p_value"] = residual_summary.get("gaussian_p_value")
        rec["zeta_resid_gaussian_p_value_floor"] = residual_summary.get("gaussian_p_value_floor")
        rec["zeta_resid_gaussian_p_value_at_floor"] = residual_summary.get("gaussian_p_value_at_floor")
        rec["zeta_resid_gaussian_reject"] = residual_summary.get("gaussian_reject")
        rec["zeta_resid_detectably_heavy"] = residual_summary.get("detectably_heavy")
        rec["zeta_resid_substantively_heavy"] = residual_summary.get("substantively_heavy")
        rec["zeta_resid_reliable"] = residual_summary.get("reliable")
        rec["zeta_resid_k_selected"] = residual_summary.get("k_selected")
    else:
        rec["zeta_resid_n_tail"] = None
        rec["zeta_resid_tail_zeta_cut"] = None
        rec["zeta_resid_alpha_eff"] = None
        rec["zeta_resid_alpha_eff_lo"] = None
        rec["zeta_resid_alpha_eff_hi"] = None
        rec["zeta_resid_xi_hat"] = None
        rec["zeta_resid_gaussian_p_value"] = None
        rec["zeta_resid_gaussian_p_value_floor"] = None
        rec["zeta_resid_gaussian_p_value_at_floor"] = None
        rec["zeta_resid_gaussian_reject"] = None
        rec["zeta_resid_detectably_heavy"] = None
        rec["zeta_resid_substantively_heavy"] = None
        rec["zeta_resid_reliable"] = None
        rec["zeta_resid_k_selected"] = None

    residual_metrics = _load_json(residual_dir / "zeta_residual_forcing_metrics.json")
    if residual_metrics is not None:
        rec["zeta_resid_n_runs"] = residual_metrics.get("n_runs")
        rec["zeta_resid_dt_is_constant"] = residual_metrics.get("dt_is_constant")
        rec["zeta_resid_dt_unique"] = residual_metrics.get("dt_unique")
        rec["zeta_resid_standardization"] = residual_metrics.get("standardization")
        std_info = residual_metrics.get("standardization_info", {}) or {}
        rec["zeta_resid_unit_scaled_tail_samples"] = std_info.get("n_unit_scaled_tail_samples")
        rec["zeta_resid_global_fallback_tail_samples"] = std_info.get("n_global_fallback_tail_samples")
    else:
        rec["zeta_resid_n_runs"] = None
        rec["zeta_resid_dt_is_constant"] = None
        rec["zeta_resid_dt_unique"] = None
        rec["zeta_resid_standardization"] = None
        rec["zeta_resid_unit_scaled_tail_samples"] = None
        rec["zeta_resid_global_fallback_tail_samples"] = None

    # Threshold-crossing per-seed table.
    tc = _load_json(agg_dir / f"{arch}_threshold_crossing.json")
    if tc is not None:
        per_seed = tc.get("per_seed", [])
        crossed = [s for s in per_seed if bool(s.get("crossed", False))]
        right_censored = [s for s in per_seed if bool(s.get("right_censored", False))]
        rec["t_cross_n_crossed"] = len(crossed)
        rec["t_cross_n_seeds"] = len(per_seed)
        rec["t_cross_n_right_censored"] = len(right_censored)
        rec["t_cross_epochs"] = [s.get("t_cross_epoch") for s in crossed]
    else:
        rec["t_cross_n_crossed"] = 0
        rec["t_cross_n_seeds"] = 0
        rec["t_cross_n_right_censored"] = 0
        rec["t_cross_epochs"] = []

    # Drift plateau at primary q_low (0.10) with 90% CI.
    # Looks at <exp2_dir>/drift_<arch>/tail_saturation.json (where exp2_dir
    # is the per-optimizer level, e.g. .../adamw/).
    drift = _load_json(exp2_dir / f"drift_{arch}" / "tail_saturation.json")
    if drift is not None and "results" in drift:
        # Find the entry closest to q_low=0.10.
        best_key = None
        best_dist = float("inf")
        for k, r in drift["results"].items():
            try:
                d = abs(float(r.get("q_low", float("nan"))) - 0.10)
            except (TypeError, ValueError):
                continue
            if d < best_dist:
                best_dist = d
                best_key = k
        if best_key is not None:
            r = drift["results"][best_key]
            rec["kappa_tail_q10"] = r.get("kappa_tail")
            rec["kappa_tail_q10_lo"] = r.get("kappa_tail_ci_lo")
            rec["kappa_tail_q10_hi"] = r.get("kappa_tail_ci_hi")
            rec["kappa_tail_n_tail"] = r.get("n_tail")
        else:
            rec["kappa_tail_q10"] = None
    else:
        rec["kappa_tail_q10"] = None

    # Final time-scale spectrum. The ζ quantiles are the SDE/drift coordinate;
    # the τ quantiles summarize the right tail that feeds the envelope theorem.
    # These are descriptive empirical summaries, not Gaussian/log-normal fits.
    tau_vals = _load_final_tau_values(exp2_dir, arch, tau_cap_mode=tau_cap_mode)
    rec["tau_spectrum_n"] = len(tau_vals)
    rec["tau_cap_mode"] = tau_cap_mode
    if tau_vals:
        zeta_vals = [-math.log(t) for t in tau_vals if math.isfinite(t) and t > 0.0]
        for name, q in (
            ("q01", 0.01),
            ("q10", 0.10),
            ("q50", 0.50),
            ("q90", 0.90),
            ("q99", 0.99),
        ):
            rec[f"zeta_{name}_final"] = _quantile(zeta_vals, q)
            rec[f"tau_{name}_final"] = _quantile(tau_vals, q)
        z10 = rec.get("zeta_q10_final")
        z90 = rec.get("zeta_q90_final")
        if isinstance(z10, (int, float)) and isinstance(z90, (int, float)):
            rec["delta_zeta_capped"] = float(z90) - float(z10)

    rec.update(_load_corrected_envelope_power_summary(exp2_dir, arch))

    return rec


# ----- Compact overview heuristic -----
def _phase_bucket_for_arch(rec: Dict[str, object]) -> str:
    """Three-bucket label: 'collapsed', 'canonical AC', 'robust AC', or 'no data'.

    Uses the majority_phase_label when present, falling back to 'no data'.
    """
    mp = rec.get("majority_phase_label")
    if isinstance(mp, str) and mp in PHASE_SHORT:
        return PHASE_SHORT[mp]
    return "no data"


def build_overview(records: List[Dict[str, object]]) -> str:
    """Generate a neutral route-readout overview for audit summaries."""
    names = [ARCH_LABEL.get(r["arch"], r["arch"]) for r in records]
    return (
        "**Audit overview.** This file reports the SharedGate/DiagGate route "
        "readout: spectra, corrected envelopes, far-left drift, update-space "
        "forcing audits, and drift-subtracted ζ-residual forcing measurements "
        f"for {', '.join(names)}. The recorded labels are included "
        "for audit only; manuscript interpretation should be made from the "
        "measured observables."
    )


# ----- Markdown rendering -----
def render_markdown(
    records: List[Dict[str, object]],
    exp2_dir: Path,
    profile: str,
    seeds: Optional[str],
    tau_cap_mode: str,
) -> str:
    """Render the full markdown summary."""
    n_seeds = max((int(r.get("n_seeds", 0)) for r in records), default=0)
    seeds_str = seeds if seeds else f"{n_seeds} (auto-detected)"

    lines: List[str] = []
    lines.append(f"# Experiment 2 — Access route through trainable diagonal gates")
    lines.append("")
    lines.append(f"**Source:** `{exp2_dir}`")
    lines.append(f"**Profile:** `{profile}`")
    lines.append(f"**Seeds:** {seeds_str}")
    lines.append(f"**Time-scale summary cap:** `{tau_cap_mode}`")
    lines.append(
        "**Setup:** same long-memory regression task and diagnostic pipeline as "
        "Experiment 1, with no heavy-tailed injection. SharedGate is the "
        "trainable collapsed reference and DiagGate is the per-unit trainable-gate "
        "route candidate. Per-seed aggregates were computed by "
        "`main_phase_trajectory.py` and reduced to the summary numbers below."
    )
    lines.append("")
    lines.append(build_overview(records))
    lines.append("")

    # --- Phase-label-per-architecture table ---
    lines.append("## Per-architecture recorded labels")
    lines.append("")
    lines.append(
        "| Architecture | n_seeds | Recorded label | Counts (collapsed / canonical AC / robust AC) "
        "| envelope_winner | crossover mode |"
    )
    lines.append("|------|--------:|----------------|-----------------------------------------------"
                 "|-----------------|----------------|")
    for r in records:
        pc = r.get("phase_counts", {}) or {}
        triple = "/".join(str(pc.get(p, 0)) for p in PHASE_ORDER)
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {r.get('n_seeds', 0)} | "
            f"{r.get('majority_phase_label', '—')} ({_fmt(r.get('majority_phase_fraction'), 2)}) | "
            f"{triple} | {r.get('envelope_winner', '—')} | {r.get('crossover_mode', '—')} |"
        )
    lines.append("")

    # --- Envelope / dynamic range / drift / forcing table ---
    lines.append("## Summary numbers")
    lines.append("")
    lines.append(
        "| Architecture | β̂_env(final) ± SE | β̂_tail (boot 90% CI) | Δζ (capped final) "
        "| κ_tail @ q_low=0.10 (90% CI) | ζ-resid α_eff | ζ-resid Gaussian p | t_cross (crossed/seeds) |"
    )
    lines.append(
        "|--------------|---------------------|-----------------------------|------------"
        "|--------------------------------------|---------------|----------------------|-------------------------|"
    )
    for r in records:
        boot = "—"
        if r.get("boot_beta_median") is not None:
            boot = (
                f"{_fmt(r.get('tail_beta_hat'), 3)} "
                f"[{_fmt(r.get('boot_beta_lo'), 3)}, {_fmt(r.get('boot_beta_hi'), 3)}]"
            )
        kappa = "—"
        if r.get("kappa_tail_q10") is not None:
            kappa = (
                f"{_fmt(r.get('kappa_tail_q10'), 4)} "
                f"[{_fmt(r.get('kappa_tail_q10_lo'), 4)}, {_fmt(r.get('kappa_tail_q10_hi'), 4)}]"
            )
        beta_env = f"{_fmt(r.get('beta_env_final_mean'), 3)} ± {_fmt(r.get('beta_env_final_se'), 3)}"
        zeta_resid_alpha = (
            f"{_fmt(r.get('zeta_resid_alpha_eff'), 3)} "
            f"[{_fmt(r.get('zeta_resid_alpha_eff_lo'), 3)}, "
            f"{_fmt(r.get('zeta_resid_alpha_eff_hi'), 3)}]"
        )
        zeta_resid_p = _fmt_p_value(
            r.get("zeta_resid_gaussian_p_value"),
            r.get("zeta_resid_gaussian_p_value_floor"),
            r.get("zeta_resid_gaussian_p_value_at_floor"),
            3,
        )
        t_cross = f"{r.get('t_cross_n_crossed', 0)}/{r.get('t_cross_n_seeds', 0)}"
        delta_zeta = r.get("delta_zeta_capped", r.get("delta_zeta"))
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {beta_env} | {boot} | "
            f"{_fmt(delta_zeta, 3)} | {kappa} | "
            f"{zeta_resid_alpha} | {zeta_resid_p} | {t_cross} |"
        )
    lines.append("")

    # --- Drift-subtracted ζ-residual forcing diagnostics ---
    lines.append("## Drift-subtracted ζ-residual forcing")
    lines.append("")
    lines.append(
        "| Architecture | runs | tail samples | ζ tail cut | α_eff (90% CI) | "
        "Gaussian p | reject | reliable | dt constant | unit/global scaled samples |"
    )
    lines.append(
        "|--------------|-----:|-------------:|-----------:|----------------|------------|--------|----------|-------------|----------------------------|"
    )
    for r in records:
        alpha_ci = (
            f"{_fmt(r.get('zeta_resid_alpha_eff'), 3)} "
            f"[{_fmt(r.get('zeta_resid_alpha_eff_lo'), 3)}, "
            f"{_fmt(r.get('zeta_resid_alpha_eff_hi'), 3)}]"
        )
        p = _fmt_p_value(
            r.get("zeta_resid_gaussian_p_value"),
            r.get("zeta_resid_gaussian_p_value_floor"),
            r.get("zeta_resid_gaussian_p_value_at_floor"),
            3,
        )
        unit_global = (
            f"{_fmt_int(r.get('zeta_resid_unit_scaled_tail_samples'))}/"
            f"{_fmt_int(r.get('zeta_resid_global_fallback_tail_samples'))}"
        )
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {_fmt_int(r.get('zeta_resid_n_runs'))} | "
            f"{_fmt_int(r.get('zeta_resid_n_tail'))} | "
            f"{_fmt(r.get('zeta_resid_tail_zeta_cut'), 3)} | {alpha_ci} | {p} | "
            f"{_fmt(r.get('zeta_resid_gaussian_reject'), 2)} | "
            f"{_fmt(r.get('zeta_resid_reliable'), 2)} | "
            f"{_fmt(r.get('zeta_resid_dt_is_constant'), 0)} | {unit_global} |"
        )
    lines.append("")
    lines.append(
        "This is the paper-facing forcing diagnostic: checkpoint-to-checkpoint "
        "increments in the far-left ζ slice after subtracting a nonparametric "
        "conditional-drift estimate and robustly standardizing per unit. The "
        "Gaussian p-value tests a matched-sample light-tail null; α_eff is an "
        "effect-size calibration, not a direct estimate of a generator parameter."
    )
    lines.append("")

    # --- Update-space forcing audit ---
    lines.append("## Update-space forcing audit")
    lines.append("")
    lines.append(
        "| Architecture | α_eff(final) ± SE | Gaussian p | reject-Gaussian | "
        "substantive-heavy | reliability |"
    )
    lines.append(
        "|--------------|-------------------|------------|-----------------|-------------------|-------------|"
    )
    for r in records:
        forcing = f"{_fmt(r.get('forcing_alpha_final'), 3)} ± {_fmt(r.get('forcing_alpha_final_se'), 3)}"
        gaussian_p = _fmt_p_value(
            r.get("forcing_gaussian_p_value_final"),
            r.get("forcing_gaussian_p_value_floor_final"),
            r.get("forcing_gaussian_p_value_at_floor_final"),
            3,
        )
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {forcing} | {gaussian_p} | "
            f"{_fmt(r.get('forcing_gaussian_reject_final'), 2)} | "
            f"{_fmt(r.get('forcing_substantively_heavy_final'), 2)} | "
            f"{_fmt(r.get('forcing_alpha_reliable_final'), 2)} |"
        )
    lines.append("")
    lines.append(
        "This table is retained as an upstream audit of parameter-update "
        "increments. The ζ-residual table above is the coordinate matched to "
        "the stochastic log-rate model."
    )
    lines.append("")

    # --- Corrected envelope audit ---
    lines.append("## Corrected envelope audit")
    lines.append("")
    lines.append(
        "| Architecture | source | full-window β_power | R² | n points | corr-ratio median | corr-ratio max |"
    )
    lines.append(
        "|--------------|--------|--------------------:|---:|---------:|------------------:|---------------:|"
    )
    for r in records:
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | "
            f"{r.get('corrected_envelope_source', '—')} | "
            f"{_fmt(r.get('corrected_envelope_power_beta'), 3)} | "
            f"{_fmt(r.get('corrected_envelope_power_r2'), 3)} | "
            f"{_fmt_int(r.get('corrected_envelope_power_n'))} | "
            f"{_fmt(r.get('corr_ratio_median'), 3)} | "
            f"{_fmt(r.get('corr_ratio_max'), 3)} |"
        )
    lines.append("")
    lines.append(
        "The paper-facing envelope comparison uses the first-order-corrected "
        "`μ₀+μ₁` curve when the audit CSV is available, averaged across seed "
        "curves before taking logs. The correction-ratio columns summarize "
        "the size of the first-order term relative to the canonical kernel."
    )
    lines.append("")

    # --- Final time-scale spectrum table ---
    lines.append("## Final time-scale spectrum")
    lines.append("")
    lines.append(
        "| Rung | n finite τ | ζ q01 | ζ q10 | ζ median | ζ q90 | ζ q99 "
        "| Δζ(q90-q10) | τ median | τ q90 | τ q99 |"
    )
    lines.append(
        "|------|-----------:|------:|------:|---------:|------:|------:"
        "|-------------:|---------:|------:|------:|"
    )
    for r in records:
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {_fmt_int(r.get('tau_spectrum_n'))} | "
            f"{_fmt(r.get('zeta_q01_final'), 3)} | "
            f"{_fmt(r.get('zeta_q10_final'), 3)} | "
            f"{_fmt(r.get('zeta_q50_final'), 3)} | "
            f"{_fmt(r.get('zeta_q90_final'), 3)} | "
            f"{_fmt(r.get('zeta_q99_final'), 3)} | "
            f"{_fmt(r.get('delta_zeta_capped'), 3)} | "
            f"{_fmt(r.get('tau_q50_final'), 1)} | "
            f"{_fmt(r.get('tau_q90_final'), 1)} | "
            f"{_fmt(r.get('tau_q99_final'), 1)} |"
        )
    lines.append("")
    lines.append(
        "The ζ columns show the log-rate coordinate modeled by the drift SDE; "
        "the τ columns summarize the right tail that enters the "
        "spectrum-to-envelope correspondence. The table applies the same "
        "finite-resolution τ cap as the generated spectrum summary; raw "
        "per-checkpoint τ CSVs are left unchanged for audit."
    )
    lines.append("")

    # --- Threshold-crossing detail ---
    lines.append("## Threshold-crossing epochs (per seed, observed crossings only)")
    lines.append("")
    for r in records:
        epochs = [int(e) for e in (r.get("t_cross_epochs") or []) if e is not None]
        if not epochs:
            lines.append(
                f"- **{ARCH_LABEL.get(r['arch'], r['arch'])}**: none of {r.get('t_cross_n_seeds', 0)} "
                f"seeds crossed within the horizon."
            )
        else:
            epochs_str = ", ".join(str(e) for e in sorted(epochs))
            lines.append(
                f"- **{ARCH_LABEL.get(r['arch'], r['arch'])}**: "
                f"crossed in {len(epochs)}/{r.get('t_cross_n_seeds', 0)} seeds "
                f"at epochs {{{epochs_str}}}; "
                f"{r.get('t_cross_n_right_censored', 0)} right-censored."
            )
    lines.append("")

    # --- Provenance / caveats ---
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The paper-facing envelope figure uses the corrected `μ₀+μ₁` curve "
        "when available; the canonical `μ₀` envelope and correction-ratio "
        "columns remain in `<arch>_envelope_audit.csv` for audit."
    )
    lines.append(
        "- Phase labels are recorded by the diagnostic pipeline in "
        "`<arch>_final_phase.json` at the end of training; the majority label "
        "is reported as a compact per-rung observable."
    )
    lines.append(
        "- Drift κ_tail is read at the primary `q_low=0.10` slice. The full "
        "quantile-cut sweep is in `drift_<arch>/tail_saturation.json`."
    )
    lines.append(
        "- The final spectrum table is paired with `exp2_time_scale_spectrum_fitlag.png`: "
        "`p(ζ)` is the drift-coordinate view, while the log-log τ-CCDF is the "
        "direct visual check for a regularly varying time-scale tail."
    )
    lines.append(
        "- This file is regenerated by `write_exp2_summary.py`. Add durable "
        "analysis below a `## Manual notes` heading if needed."
    )

    return "\n".join(lines) + "\n"


# ----- CLI -----
def parse_arch_list(s: str) -> List[str]:
    items = [a.strip().lower() for a in s.split(",") if a.strip()]
    aliases = {"diaggate": "diag", "sharedgate": "shared"}
    return [aliases.get(a, a) for a in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--exp2_dir", type=Path,
        default=Path("results/exp2_phase_full/adamw"),
        help="Experiment 2 per-optimizer result directory (contains "
             "aggregated/<arch>/ and drift_<arch>/). Default: "
             "results/exp2_phase_full/adamw",
    )
    ap.add_argument(
        "--architectures", type=str, default="shared,diag",
        help="Comma-separated architecture short names.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown path. Default: results/exp2_phase_<profile>_results_summary.md",
    )
    ap.add_argument(
        "--profile", type=str, default=None,
        help="Profile label to embed in the file (e.g. 'full', 'smoke'). "
             "Default: derived from --exp2_dir name.",
    )
    ap.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list to embed in the summary (optional).",
    )
    ap.add_argument(
        "--tau_cap_mode", type=str, default="seq_len", choices=_TAU_CAP_MODES,
        help="Finite-resolution cap for final τ-spectrum quantiles. `seq_len` "
             "uses τ <= T; `fit_lag_max` uses τ <= max(tau_fit_lags); `raw` "
             "keeps all finite positive τ values.",
    )
    args = ap.parse_args()

    if not args.exp2_dir.exists():
        raise SystemExit(f"ERROR: --exp2_dir does not exist: {args.exp2_dir}")
    architectures = parse_arch_list(args.architectures)
    if not architectures:
        raise SystemExit("ERROR: no architectures parsed from --architectures")

    # Derive profile from the parent dir name "exp2_phase_<profile>" when not
    # explicitly given.
    profile = args.profile
    if profile is None:
        parent = args.exp2_dir.parent.name  # e.g. "exp2_phase_full"
        profile = parent.split("_")[-1] if parent.startswith("exp2_phase_") else "unknown"

    output = args.output
    if output is None:
        results_root = args.exp2_dir.parent.parent
        output = results_root / f"exp2_phase_{profile}_results_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    records = [
        collect_arch_record(args.exp2_dir, arch, tau_cap_mode=args.tau_cap_mode)
        for arch in architectures
    ]
    md = render_markdown(records, args.exp2_dir, profile, args.seeds, args.tau_cap_mode)
    with open(output, "w") as f:
        f.write(md)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
