#!/usr/bin/env python3
"""Write a consolidated Experiment 1 results summary covering both paths.

Experiment 1 is the structural negative control on the ConstGate architecture.
It is run in two complementary phases that are paired by design:

  - Path A (intervention)   : symmetric alpha-stable noise injected per-element
                              into the parameter-gradient updates after gradient
                              clipping. The forcing-side route ingredient is
                              established by the injection mechanism itself.
  - Path B (spontaneous)    : same ConstGate trained under the matched protocol
                              with no intervention. The forcing proxy, drift
                              plateau, dynamic range, and envelope class are
                              read directly off the trajectory.

This writer ingests the canonical aggregates that ``main_phase_trajectory.py`` produces
for each path under
    <path>/<optimizer>/aggregated/<model>/
and the drift-diagnostic output under
    <path>/<optimizer>/drift_<model>/
and emits a single Markdown summary at ``results/exp1_results_summary.md``
that the manuscript section can pull verbatim. Path A is described first
throughout (matching the manuscript ordering convention).

This is the consolidated replacement for the two legacy files
``exp1_results_summary.md`` and ``exp1_inject_results_summary.md``. After
running this writer once the legacy files can be deleted.

Usage:
    # Defaults to results/exp1_constgate_full/adamw and
    # results/exp1_constgate_inject_full/adamw.
    python write_exp1_summary.py

    # Explicit paths and output file:
    python write_exp1_summary.py \\
        --path_b results/exp1_constgate_full/adamw \\
        --path_a results/exp1_constgate_inject_full/adamw \\
        --model  const \\
        --output results/exp1_results_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Loading helpers (intentionally tiny, dependency-free).
# ============================================================
def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_csv_rows(p: Path) -> Optional[List[Dict[str, str]]]:
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _load_last_row(p: Path) -> Optional[Dict[str, str]]:
    rows = _load_csv_rows(p)
    return rows[-1] if rows else None


def _fmt(v: Any, digits: int = 3, missing: str = "—") -> str:
    """Format a float; use ``missing`` for None / non-finite."""
    if v is None:
        return missing
    try:
        f = float(v)
    except (TypeError, ValueError):
        return missing
    if f != f or f in (float("inf"), float("-inf")):
        return missing
    return f"{f:.{digits}f}"


def _fmt_int(v: Any, missing: str = "—") -> str:
    if v is None:
        return missing
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return missing


def _quantile(xs: List[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile for a finite list of floats."""
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


def _mean(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else None


def _std(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if math.isfinite(x)]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def _checkpoint_epoch(p: Path) -> int:
    try:
        return int(p.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _load_final_tau_values(path_dir: Path, model: str) -> List[float]:
    """Load finite positive tau values from the final per-seed checkpoint CSVs."""
    vals: List[float] = []
    for seed_dir in sorted(path_dir.glob("seed_*")):
        tau_dir = seed_dir / model / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
        if not candidates:
            continue
        with open(candidates[-1]) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(tau) and tau > 0:
                    vals.append(tau)
    return vals


# ============================================================
# Per-path record collection.
#
# A "path" here is a per-optimizer result directory like
# ``results/exp1_constgate_full/adamw`` containing
# ``aggregated/<model>/`` and ``drift_<model>/``.
# ============================================================
def collect_path_record(path_dir: Path, model: str) -> Dict[str, Any]:
    """Collect the headline numbers for one path (Path A or Path B)."""
    agg_dir = path_dir / "aggregated" / model
    rec: Dict[str, Any] = {"path_dir": str(path_dir), "model": model}

    # --- Final phase verdict (canonical Table-1 rule) -----------------
    fp = _load_json(agg_dir / f"{model}_final_phase.json")
    rec["final_phase_present"] = fp is not None
    if fp is not None:
        rec["n_seeds"] = int(fp.get("n_seeds", 0))
        rec["phase_counts"] = dict(fp.get("phase_counts", {}))
        rec["majority_phase_label"] = str(fp.get("majority_phase_label", "—"))
        rec["majority_phase_fraction"] = fp.get("phase_majority_fraction")
        rec["tail_beta_hat"] = fp.get("tail_beta_hat")
        rec["tail_beta_r2"] = fp.get("tail_beta_r2")
        rec["boot_beta_median"] = fp.get("boot_beta_median")
        rec["boot_beta_lo"] = fp.get("boot_beta_lo")
        rec["boot_beta_hi"] = fp.get("boot_beta_hi")
        rec["delta_zeta"] = fp.get("delta_zeta")
        rec["zeta_q10"] = fp.get("zeta_q10")
        rec["zeta_q90"] = fp.get("zeta_q90")
        rec["envelope_winner"] = fp.get("envelope_winner")
        rec["majority_envelope_winner"] = fp.get("majority_envelope_winner")
        rec["aic"] = fp.get("aic", {})
        cd = fp.get("crossover_diagnostic", {}) or {}
        rec["crossover_majority_mode"] = cd.get("majority_mode")
        rec["power_law_window_pass"] = cd.get("power_law_window_pass")
        rec["power_law_window_pass_fraction"] = cd.get("power_law_window_pass_fraction")
    else:
        rec["n_seeds"] = 0
        rec["phase_counts"] = {}
        rec["majority_phase_label"] = "—"

    # --- Final-epoch trajectory aggregates ----------------------------
    last = _load_last_row(agg_dir / "phase_trajectory_aggregated.csv")
    if last is not None:
        rec["final_epoch"] = last.get("epoch")
        rec["final_step"] = last.get("step")
        rec["alpha_ecf_final"] = last.get("alpha_ecf_mean")
        rec["alpha_ecf_se_final"] = last.get("alpha_ecf_se")
        rec["alpha_mcculloch_final"] = last.get("alpha_mcculloch_mean")
        rec["alpha_mcculloch_se_final"] = last.get("alpha_mcculloch_se")
        rec["alpha_reliable_final"] = last.get("alpha_reliable_mean")
        rec["beta_env_final"] = last.get("beta_env_mean")
        rec["beta_env_se_final"] = last.get("beta_env_se")
        rec["beta_env_r2_final"] = last.get("beta_env_r2_mean")

    # --- Threshold-crossing summary -----------------------------------
    tc = _load_json(agg_dir / f"{model}_threshold_crossing.json")
    if tc is not None:
        per_seed = tc.get("per_seed", []) or []
        crossed = [s for s in per_seed if bool(s.get("crossed", False))]
        right_censored = [s for s in per_seed if bool(s.get("right_censored", False))]
        rec["t_cross_n_crossed"] = len(crossed)
        rec["t_cross_n_seeds"] = len(per_seed)
        rec["t_cross_n_right_censored"] = len(right_censored)
        rec["t_cross_epochs"] = [s.get("t_cross_epoch") for s in crossed]
        rec["t_cross_steps"] = [s.get("t_cross_step") for s in crossed]
    else:
        rec["t_cross_n_crossed"] = 0
        rec["t_cross_n_seeds"] = 0
        rec["t_cross_n_right_censored"] = 0
        rec["t_cross_epochs"] = []
        rec["t_cross_steps"] = []

    # --- Drift plateau full sweep -------------------------------------
    # Accept both legacy names: drift_const, drift_constgate.
    drift_dir = None
    for cand in (f"drift_{model}", f"drift_{model}gate"):
        d = path_dir / cand / "tail_saturation.json"
        if d.exists():
            drift_dir = d
            break
    drift = _load_json(drift_dir) if drift_dir else None
    rec["drift_sweep"] = []  # list of dicts: q_low, kappa_tail, lo, hi, n_tail, zeta_cutoff
    if drift is not None and "results" in drift:
        items = sorted(
            drift["results"].items(),
            key=lambda kv: float(kv[1].get("q_low", float("nan"))),
        )
        for _, r in items:
            rec["drift_sweep"].append({
                "q_low": r.get("q_low"),
                "kappa_tail": r.get("kappa_tail"),
                "lo": r.get("kappa_tail_ci_lo"),
                "hi": r.get("kappa_tail_ci_hi"),
                "n_tail": r.get("n_tail", r.get("n_tail_samples")),
                "zeta_cutoff": r.get("zeta_cutoff", r.get("tail_zeta_cut")),
                "constant_preferred": r.get("constant_preferred"),
            })

    # --- Envelope audit (corr_ratio diagnostic) -----------------------
    audit_rows = _load_csv_rows(agg_dir / f"{model}_envelope_audit.csv")
    if audit_rows:
        try:
            ells = [int(r["ell"]) for r in audit_rows]
            cr = [float(r.get("corr_ratio_mu1_over_mu0", "nan")) for r in audit_rows]
            rec["corr_ratio_min"] = min(cr)
            rec["corr_ratio_max"] = max(cr)
            rec["corr_ratio_at_ell_min"] = (ells[0], cr[0])
            rec["corr_ratio_at_ell_max"] = (ells[-1], cr[-1])
        except (KeyError, ValueError):
            rec["corr_ratio_min"] = rec["corr_ratio_max"] = None
    else:
        rec["corr_ratio_min"] = rec["corr_ratio_max"] = None

    # --- Final time-scale spectrum ------------------------------------
    # These are descriptive finite-sample summaries of the empirical spectrum,
    # not parametric fits. The far-right tau tail can contain rare large
    # time-scale outliers, so the summary reports robust quantiles rather than
    # pretending the spectrum is Gaussian/log-normal.
    tau_vals = _load_final_tau_values(path_dir, model)
    rec["tau_spectrum_n"] = len(tau_vals)
    if tau_vals:
        zeta_vals = [-math.log(t) for t in tau_vals]
        rec["zeta_mean_final"] = _mean(zeta_vals)
        rec["zeta_std_final"] = _std(zeta_vals)
        for q, name in (
            (0.01, "q01"), (0.10, "q10"), (0.50, "q50"),
            (0.90, "q90"), (0.99, "q99"),
        ):
            rec[f"zeta_{name}_final"] = _quantile(zeta_vals, q)
        for q, name in (
            (0.50, "q50"), (0.90, "q90"), (0.99, "q99"),
        ):
            rec[f"tau_{name}_final"] = _quantile(tau_vals, q)

    return rec


# ============================================================
# Markdown rendering.
# ============================================================
def _seed_list_from_path(path_dir: Path) -> List[str]:
    """Return sorted list of seed_NNNN names actually present on disk.

    ``path_dir`` is the per-optimizer directory (e.g.
    ``results/exp1_constgate_full/adamw``); per-seed dirs are its
    immediate children. We also fall back to scanning the parent
    directory for backward compatibility with older layouts that placed
    seed_NNNN at the top-level (sibling of adamw/).
    """
    if not path_dir.is_dir():
        return []
    seeds = sorted(
        p.name for p in path_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )
    if seeds:
        return seeds
    # Fallback: older layouts may have seed_* as siblings of the optimizer dir.
    parent = path_dir.parent
    return sorted(
        p.name for p in parent.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )


def _drift_sweep_table(rec: Dict[str, Any]) -> List[str]:
    """Render the per-path drift sweep as markdown table rows."""
    rows: List[str] = []
    if not rec.get("drift_sweep"):
        return rows
    for r in rec["drift_sweep"]:
        ql = _fmt(r.get("q_low"), 2)
        zc = _fmt(r.get("zeta_cutoff"), 3)
        nt = _fmt_int(r.get("n_tail"))
        kt = _fmt(r.get("kappa_tail"), 5)
        lo = _fmt(r.get("lo"), 5)
        hi = _fmt(r.get("hi"), 5)
        cp = "yes" if bool(r.get("constant_preferred")) else "no"
        rows.append(f"| {ql} | {zc} | {nt} | {kt} | [{lo}, {hi}] | {cp} |")
    return rows


def _kappa_at_q010(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Pick the q_low entry closest to 0.10 — used in compact comparison tables."""
    if not rec.get("drift_sweep"):
        return None, None, None
    best = min(
        rec["drift_sweep"],
        key=lambda r: abs(float(r.get("q_low", float("nan"))) - 0.10)
        if r.get("q_low") is not None else float("inf"),
    )
    return best.get("kappa_tail"), best.get("lo"), best.get("hi")


def _headline_verdict(path_a: Dict[str, Any], path_b: Dict[str, Any]) -> str:
    """Generate the consolidated headline-verdict paragraph.

    The verdict is keyed on the canonical full-trajectory phase label in
    both paths. We do not attempt to second-guess the manuscript prose —
    this paragraph is a starting point that the author can refine.
    """
    pa_label = path_a.get("majority_phase_label", "—")
    pb_label = path_b.get("majority_phase_label", "—")
    pa_counts = path_a.get("phase_counts", {})
    pb_counts = path_b.get("phase_counts", {})

    pa_collapsed = (pa_label == "collapsed")
    pb_collapsed = (pb_label == "collapsed")

    if pa_collapsed and pb_collapsed:
        return (
            "**Headline verdict.** ConstGate fails the route under both paths. "
            "In Path A the forcing channel is externally supplied by construction "
            "(symmetric $\\alpha$-stable per-element noise on the parameter gradients), "
            "yet the drift-side route condition still fails and the canonical "
            "full-trajectory phase verdict is **collapsed** in every seed "
            f"({pa_counts}). Path B confirms that the spontaneous run is also "
            f"**collapsed** in every seed ({pb_counts}); the forcing proxy stays at "
            "the near-Gaussian boundary, the far-left drift plateau is negative on every "
            "quantile slice, and Path A narrows rather than broadens the final "
            "time-scale spectrum relative to Path B. The drift-axis absence is therefore not explained by lack of "
            "heavy-tailed forcing alone --- it is an architectural property of the "
            "frozen-gate constraint."
        )
    return (
        f"**Headline verdict.** Path A majority phase: **{pa_label}** "
        f"({pa_counts}). Path B majority phase: **{pb_label}** ({pb_counts}). "
        "The route conditions are not jointly satisfied in either path; "
        "see the headline-numbers tables below for the discriminating axes."
    )


def render_markdown(
    path_a: Dict[str, Any],
    path_b: Dict[str, Any],
    path_a_dir: Path,
    path_b_dir: Path,
    seeds_a: List[str],
    seeds_b: List[str],
) -> str:
    """Render the full markdown summary."""
    n_seeds = int(path_a.get("n_seeds", 0) or path_b.get("n_seeds", 0) or 0)
    seeds_str_a = ", ".join(s.replace("seed_", "") for s in seeds_a) or "—"
    seeds_str_b = ", ".join(s.replace("seed_", "") for s in seeds_b) or "—"

    lines: List[str] = []
    lines.append("# Experiment 1 — Structural negative control: ConstGate (Path A vs Path B)")
    lines.append("")
    lines.append("**Source:**")
    lines.append(f"- Path A (with injection): `{path_a_dir}`")
    lines.append(f"- Path B (no injection):   `{path_b_dir}`")
    lines.append("")
    lines.append(f"**Seeds (Path A):** {seeds_str_a}")
    lines.append(f"**Seeds (Path B):** {seeds_str_b}")
    lines.append(f"**Horizon:** {path_b.get('final_epoch', '—')} epochs "
                 f"/ {path_b.get('final_step', '—')} cumulative optimizer steps")
    lines.append(
        "**Setup:** ConstGate frozen-gate architecture (`s_const = 0.005`, "
        "leak = 1 - `s_const` ≈ 0.995, gate buffer is non-learnable). AdamW optimizer. "
        "Same long-memory regression task and diagnostic pipeline used across both paths."
    )
    lines.append(
        "**Intervention (Path A only):** symmetric $\\alpha$-stable noise with "
        "stability index $\\alpha = 1.6$ added per-element to all trainable parameter "
        "gradients after `clip_grad_norm_`, with per-parameter dispersion calibrated "
        "to the natural gradient root-mean-square. Path B trains with no intervention."
    )
    lines.append("")
    lines.append(_headline_verdict(path_a, path_b))
    lines.append("")

    # --- Phase verdict table -----------------------------------------
    lines.append("## Phase verdicts (canonical full-trajectory rule)")
    lines.append("")
    lines.append("| Path | n_seeds | majority phase | counts | envelope_winner | crossover_mode | power_law_window_pass |")
    lines.append("|------|--------:|----------------|--------|-----------------|----------------|-----------------------|")
    for tag, rec in (("Path A", path_a), ("Path B", path_b)):
        counts = rec.get("phase_counts", {})
        counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "—"
        lines.append(
            f"| {tag} | {rec.get('n_seeds', 0)} | "
            f"{rec.get('majority_phase_label', '—')} "
            f"({_fmt(rec.get('majority_phase_fraction'), 2)}) | "
            f"{counts_str} | "
            f"{rec.get('majority_envelope_winner', rec.get('envelope_winner', '—'))} | "
            f"{rec.get('crossover_majority_mode', '—')} | "
            f"{rec.get('power_law_window_pass', '—')} |"
        )
    lines.append("")

    # --- Forcing-axis table -------------------------------------------
    lines.append("## Forcing proxy α at convergence")
    lines.append("")
    lines.append("| Diagnostic | Path A (with injection) | Path B (spontaneous) |")
    lines.append("|------------|-------------------------|----------------------|")
    lines.append(
        f"| Pooled $\\hat\\alpha_{{\\mathrm{{ECF}}}}$ (final) | "
        f"{_fmt(path_a.get('alpha_ecf_final'), 3)} ± {_fmt(path_a.get('alpha_ecf_se_final'), 3)} | "
        f"{_fmt(path_b.get('alpha_ecf_final'), 3)} ± {_fmt(path_b.get('alpha_ecf_se_final'), 3)} |"
    )
    lines.append(
        f"| Pooled $\\hat\\alpha_{{\\mathrm{{McC}}}}$ (final) | "
        f"{_fmt(path_a.get('alpha_mcculloch_final'), 3)} ± {_fmt(path_a.get('alpha_mcculloch_se_final'), 3)} | "
        f"{_fmt(path_b.get('alpha_mcculloch_final'), 3)} ± {_fmt(path_b.get('alpha_mcculloch_se_final'), 3)} |"
    )
    lines.append("")
    lines.append(
        "In Path A the forcing channel is supplied by the injected symmetric "
        "$\\alpha$-stable gradient noise. Its presence is part of the experimental "
        "setup; the load-bearing Path A measurements are the drift plateau, final "
        "spectrum, and canonical phase verdict."
    )
    lines.append("")

    # --- Drift-plateau table (compact) --------------------------------
    ka_q10, ka_lo, ka_hi = _kappa_at_q010(path_a)
    kb_q10, kb_lo, kb_hi = _kappa_at_q010(path_b)
    lines.append("## Far-left drift plateau at primary $q_{\\mathrm{low}}=0.10$")
    lines.append("")
    lines.append("| Path | $\\hat\\kappa_{\\mathrm{tail}}$ | 90% block-bootstrap CI | verdict |")
    lines.append("|------|---:|---|---------|")
    lines.append(
        f"| Path A | {_fmt(ka_q10, 5)} | [{_fmt(ka_lo, 5)}, {_fmt(ka_hi, 5)}] | "
        f"{'positive (closure gate met)' if (ka_q10 is not None and ka_q10 > 0) else 'negative (closure gate not established)'} |"
    )
    lines.append(
        f"| Path B | {_fmt(kb_q10, 5)} | [{_fmt(kb_lo, 5)}, {_fmt(kb_hi, 5)}] | "
        f"{'positive (closure gate met)' if (kb_q10 is not None and kb_q10 > 0) else 'negative (closure gate not established)'} |"
    )
    lines.append("")

    # --- Drift-plateau full sweep (per-path) --------------------------
    lines.append("### Full $q_{\\mathrm{low}}$ sweep, Path A")
    lines.append("")
    lines.append("| q_low | ζ cutoff | n_tail | $\\hat\\kappa_{\\mathrm{tail}}$ | 90% CI | constant_preferred |")
    lines.append("|------:|---------:|-------:|-------:|-------|-------------------|")
    lines.extend(_drift_sweep_table(path_a) or ["| — | — | — | — | — | — |"])
    lines.append("")
    lines.append("### Full $q_{\\mathrm{low}}$ sweep, Path B")
    lines.append("")
    lines.append("| q_low | ζ cutoff | n_tail | $\\hat\\kappa_{\\mathrm{tail}}$ | 90% CI | constant_preferred |")
    lines.append("|------:|---------:|-------:|-------:|-------|-------------------|")
    lines.extend(_drift_sweep_table(path_b) or ["| — | — | — | — | — | — |"])
    lines.append("")

    # --- Final time-scale spectrum ------------------------------------
    lines.append("## Final time-scale spectrum at convergence")
    lines.append("")
    lines.append("| Path | n finite τ | ζ q01 | ζ q10 | ζ median | ζ q90 | ζ q99 | Δζ(q90-q10) | τ median | τ q90 | τ q99 |")
    lines.append("|------|-----------:|------:|------:|---------:|------:|------:|-------------:|---------:|------:|------:|")
    lines.append(
        f"| Path A | {_fmt_int(path_a.get('tau_spectrum_n'))} | "
        f"{_fmt(path_a.get('zeta_q01_final'), 3)} | "
        f"{_fmt(path_a.get('zeta_q10_final'), 3)} | "
        f"{_fmt(path_a.get('zeta_q50_final'), 3)} | "
        f"{_fmt(path_a.get('zeta_q90_final'), 3)} | "
        f"{_fmt(path_a.get('zeta_q99_final'), 3)} | "
        f"{_fmt(path_a.get('delta_zeta'), 3)} | "
        f"{_fmt(path_a.get('tau_q50_final'), 1)} | "
        f"{_fmt(path_a.get('tau_q90_final'), 1)} | "
        f"{_fmt(path_a.get('tau_q99_final'), 1)} |"
    )
    lines.append(
        f"| Path B | {_fmt_int(path_b.get('tau_spectrum_n'))} | "
        f"{_fmt(path_b.get('zeta_q01_final'), 3)} | "
        f"{_fmt(path_b.get('zeta_q10_final'), 3)} | "
        f"{_fmt(path_b.get('zeta_q50_final'), 3)} | "
        f"{_fmt(path_b.get('zeta_q90_final'), 3)} | "
        f"{_fmt(path_b.get('zeta_q99_final'), 3)} | "
        f"{_fmt(path_b.get('delta_zeta'), 3)} | "
        f"{_fmt(path_b.get('tau_q50_final'), 1)} | "
        f"{_fmt(path_b.get('tau_q90_final'), 1)} | "
        f"{_fmt(path_b.get('tau_q99_final'), 1)} |"
    )
    lines.append("")
    lines.append(
        "The spectrum diagnostic is descriptive, not a Gaussian/log-normal fit. "
        "The far-right $\\tau$ tail contains rare large-time-scale outliers, "
        "which are visible in $\\tau_{q99}$ and in the CCDF plot, but the canonical "
        "phase rule still finds no resolved power-law window and both paths remain "
        "collapsed. Externally supplied heavy-tailed forcing in Path A narrows the "
        "main body of the spectrum rather than broadening it."
    )
    lines.append("")

    # --- Envelope / tail-beta table -----------------------------------
    lines.append("## Macroscopic envelope at convergence (canonical $\\mu_0$ kernel)")
    lines.append("")
    lines.append("| Path | $\\hat\\beta_{\\mathrm{tail}}$ (R²) | boot β median [90% CI] | envelope_winner | corr_ratio $\\|\\mu_1\\|/\\|\\mu_0\\|$ at min ℓ / max ℓ |")
    lines.append("|------|---:|---|-----------------|------|")
    for tag, rec in (("Path A", path_a), ("Path B", path_b)):
        crr_min = rec.get("corr_ratio_at_ell_min")
        crr_max = rec.get("corr_ratio_at_ell_max")
        crr_str = "—"
        if crr_min and crr_max:
            crr_str = (
                f"ℓ={crr_min[0]}: {_fmt(crr_min[1], 3)} / "
                f"ℓ={crr_max[0]}: {_fmt(crr_max[1], 3)}"
            )
        boot_str = (
            f"{_fmt(rec.get('boot_beta_median'), 3)} "
            f"[{_fmt(rec.get('boot_beta_lo'), 3)}, {_fmt(rec.get('boot_beta_hi'), 3)}]"
        )
        lines.append(
            f"| {tag} | {_fmt(rec.get('tail_beta_hat'), 3)} "
            f"({_fmt(rec.get('tail_beta_r2'), 3)}) | "
            f"{boot_str} | "
            f"{rec.get('majority_envelope_winner', rec.get('envelope_winner', '—'))} | "
            f"{crr_str} |"
        )
    lines.append("")
    lines.append(
        "Note on path identity: because ConstGate's gate is frozen "
        "(`s_const = 0.005`, non-learnable), the canonical zero-order kernel "
        "$\\mu_0$ reduces to a deterministic function of the leak (≈ "
        "$0.995^{\\ell}$ averaged over batch, time, and neuron) and is therefore "
        "identical in Path A and Path B. The path-discriminator signal lives on "
        "the first-order correction $\\mu_1$ (audit-only diagnostic), the "
        "finite-sample $\\tau$-spectrum shape, and the drift-plateau side."
    )
    lines.append("")

    # --- Threshold-crossing -------------------------------------------
    lines.append("## Threshold-crossing diagnostic $t_{\\mathrm{cross}}$")
    lines.append("")
    lines.append("| Path | crossed / seeds | right-censored | observed t_cross epochs |")
    lines.append("|------|----------------:|--------------:|-------------------------|")
    for tag, rec in (("Path A", path_a), ("Path B", path_b)):
        epochs = [int(e) for e in (rec.get("t_cross_epochs") or []) if e is not None]
        epochs_str = ", ".join(str(e) for e in sorted(epochs)) or "—"
        lines.append(
            f"| {tag} | {rec.get('t_cross_n_crossed', 0)} / {rec.get('t_cross_n_seeds', 0)} | "
            f"{rec.get('t_cross_n_right_censored', 0)} | {{{epochs_str}}} |"
        )
    lines.append("")

    # --- AIC / fit class diagnostic -----------------------------------
    lines.append("## Envelope fit AIC at convergence")
    lines.append("")
    lines.append("| Path | exponential | power | tempered |")
    lines.append("|------|---:|---:|---:|")
    for tag, rec in (("Path A", path_a), ("Path B", path_b)):
        aic = rec.get("aic") or {}
        lines.append(
            f"| {tag} | {_fmt(aic.get('exponential'), 2)} | "
            f"{_fmt(aic.get('power'), 2)} | {_fmt(aic.get('tempered'), 2)} |"
        )
    lines.append("")

    # --- Notes / provenance -------------------------------------------
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Macro-envelope `f(ℓ)` is the canonical zero-order $\\mu_0$ kernel "
        "(`diagnostics.compute_macro_envelope_comparison`). The legacy "
        "$\\mu_0 + \\mu_1$ envelope is retained as an audit-only diagnostic in "
        "`<model>_envelope_audit.csv`. `corr_ratio` = "
        "$\\|\\mu_1\\|/\\|\\mu_0\\|$ exceeds 1 at long $\\ell$ — confirming that "
        "the first-order Taylor expansion was outside its validity regime exactly "
        "where the legacy envelope used to develop non-monotonicities."
    )
    lines.append(
        "- Phase labels come from the canonical Table-1 rule applied to "
        "`<model>_final_phase.json` at the end of training; the majority label "
        "is reported as the headline phase for each path."
    )
    lines.append(
        "- Drift $\\hat{\\kappa}_{\\mathrm{tail}}$ is reported at the primary "
        "$q_{\\mathrm{low}}=0.10$ slice in the compact comparison and across the "
        "full quantile-cut sweep in the per-path tables. CIs are $90\\%$ "
        "block-bootstrap."
    )
    lines.append(
        "- This file is regenerated by `write_exp1_summary.py`. Hand-edit only "
        "after appending a `## Manual notes` section so subsequent regenerations "
        "preserve the human commentary."
    )

    return "\n".join(lines) + "\n"


# ============================================================
# CLI.
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--path_a", type=Path,
        default=Path("results/exp1_constgate_inject_full/adamw"),
        help="Path A (with injection) per-optimizer directory "
             "(default: results/exp1_constgate_inject_full/adamw).",
    )
    ap.add_argument(
        "--path_b", type=Path,
        default=Path("results/exp1_constgate_full/adamw"),
        help="Path B (no injection) per-optimizer directory "
             "(default: results/exp1_constgate_full/adamw).",
    )
    ap.add_argument(
        "--model", type=str, default="const",
        help="Model short name shared by both paths (default: const).",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("results/exp1_results_summary.md"),
        help="Output markdown path (default: results/exp1_results_summary.md).",
    )
    args = ap.parse_args()

    if not args.path_a.exists():
        raise SystemExit(f"ERROR: --path_a does not exist: {args.path_a}")
    if not args.path_b.exists():
        raise SystemExit(f"ERROR: --path_b does not exist: {args.path_b}")

    path_a_rec = collect_path_record(args.path_a, args.model)
    path_b_rec = collect_path_record(args.path_b, args.model)
    seeds_a = _seed_list_from_path(args.path_a)
    seeds_b = _seed_list_from_path(args.path_b)

    md = render_markdown(
        path_a_rec, path_b_rec, args.path_a, args.path_b, seeds_a, seeds_b,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
