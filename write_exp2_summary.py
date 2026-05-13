#!/usr/bin/env python3
"""Write results/exp2_phase_<profile>_results_summary.md.

This is the exp2 counterpart of the ``results/exp1_results_summary.md`` and
``results/exp1_inject_results_summary.md`` artifacts. It consumes the per-
architecture aggregates that ``main_phase_trajectory.py`` writes for Experiment 2 (the
capacity-ladder phase-trajectory experiment) and emits a markdown summary
that the LaTeX section can pull verbatim.

Per architecture we report:
  - phase verdict counts (collapsed / canonical AC / robust AC),
  - majority phase label and the canonical full-trajectory verdict,
  - envelope exponent β̂_env (mean ± SE across seeds) at the final epoch,
  - envelope tail β̂ and bootstrap percentile interval from <arch>_final_phase.json,
  - final time-scale spectrum quantiles in both ζ = -log τ and τ,
    with Δζ = ζ_q90 − ζ_q10 retained as a scalar width summary,
  - far-left drift plateau κ_tail at the primary q_low (0.10) with 90% CI,
  - per-seed threshold-crossing epoch t_cross (observed vs right-censored),
  - alpha_ECF at the final epoch as the spontaneous forcing proxy.

The headline-verdict paragraph is generated heuristically from the per-
architecture results: we walk the capacity ladder in order and identify the
first rung at which a sustained anti-collapsed verdict is reached. The
generated paragraph is meant as a starting point for hand-editing into the
manuscript; the script writes the raw numbers underneath so the prose can be
cross-checked.

Usage:
    python write_exp2_summary.py
    python write_exp2_summary.py \\
        --exp2_dir results/exp2_phase_full/adamw \\
        --architectures shared,diag,gru,lstm \\
        --output results/exp2_phase_full_results_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Architecture human-readable labels (matches plot_exp2_phase_ladder.py).
ARCH_LABEL: Dict[str, str] = {
    "diag":   "DiagGate",
    "shared": "SharedGate",
    "gru":    "GRU",
    "lstm":   "LSTM",
}

# Phase-verdict short labels and ordering.
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


def _fmt_int(v: Optional[float]) -> str:
    """Format a numeric count or ``-`` for missing values."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "—"


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


def _load_final_tau_values(exp2_dir: Path, arch: str) -> List[float]:
    """Load finite positive τ values from each seed's final checkpoint CSV."""
    vals: List[float] = []
    for seed_dir in sorted(exp2_dir.glob("seed_*")):
        tau_dir = seed_dir / arch / "checkpoint_taus"
        candidates = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
        if not candidates:
            continue
        with open(candidates[-1]) as f:
            for row in csv.DictReader(f):
                try:
                    tau = float(row["tau"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(tau) and tau > 0.0:
                    vals.append(tau)
    return vals


# ----- Per-architecture record -----
def collect_arch_record(
    exp2_dir: Path, arch: str,
) -> Dict[str, object]:
    """Collect the per-architecture summary record from the aggregates."""
    agg_dir = exp2_dir / "aggregated" / arch
    rec: Dict[str, object] = {"arch": arch}

    # Phase verdict and final-epoch envelope numbers from <arch>_final_phase.json
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

    # Final-epoch beta_env and spontaneous forcing proxy from the aggregated
    # phase trajectory. The reliability flag is retained internally as audit,
    # but the summary foregrounds alpha_ECF rather than log-like flags.
    last = _load_last_row(agg_dir / "phase_trajectory_aggregated.csv")
    if last is not None:
        rec["beta_env_final_mean"] = last.get("beta_env_mean")
        rec["beta_env_final_se"] = last.get("beta_env_se")
        rec["beta_env_r2_final_mean"] = last.get("beta_env_r2_mean")
        rec["alpha_reliable_final"] = last.get("alpha_reliable_mean")
        rec["alpha_ecf_final"] = last.get("alpha_ecf_mean")
        rec["final_epoch"] = last.get("epoch")
    else:
        rec["beta_env_final_mean"] = None
        rec["beta_env_final_se"] = None
        rec["beta_env_r2_final_mean"] = None
        rec["alpha_reliable_final"] = None
        rec["alpha_ecf_final"] = None
        rec["final_epoch"] = None

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
    tau_vals = _load_final_tau_values(exp2_dir, arch)
    rec["tau_spectrum_n"] = len(tau_vals)
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

    return rec


# ----- Headline-verdict heuristic -----
def _verdict_for_arch(rec: Dict[str, object]) -> str:
    """Three-bucket label: 'collapsed', 'canonical AC', 'robust AC', or 'no data'.

    Uses the majority_phase_label when present, falling back to 'no data'.
    """
    mp = rec.get("majority_phase_label")
    if isinstance(mp, str) and mp in PHASE_SHORT:
        return PHASE_SHORT[mp]
    return "no data"


def build_headline(records: List[Dict[str, object]]) -> str:
    """Generate the headline paragraph identifying the transition rung.

    The capacity ladder is the order of ``records`` (caller passes them in
    ladder order). The headline points to the first rung that reaches a
    sustained anti-collapsed verdict (canonical AC or better) and, if any
    rung downstream of that one collapses again, flags the non-monotonic
    case explicitly.
    """
    ladder = [(_verdict_for_arch(r), r) for r in records]
    first_ac_idx = next(
        (i for i, (v, _) in enumerate(ladder) if v in ("canonical AC", "robust AC")),
        None,
    )
    if first_ac_idx is None:
        return (
            "**Headline verdict.** No rung of the capacity ladder reaches a "
            "sustained anti-collapsed phase verdict in the canonical "
            "full-trajectory diagnostic. Within the explored ladder, the "
            "architectural transition to a stable anti-collapsed spectrum is not "
            "observed."
        )

    first_arch = ARCH_LABEL.get(ladder[first_ac_idx][1]["arch"], ladder[first_ac_idx][1]["arch"])
    below = ladder[:first_ac_idx]
    above = ladder[first_ac_idx:]

    below_summary = (
        f"the {len(below)} lower rung(s) "
        f"({', '.join(ARCH_LABEL.get(r['arch'], r['arch']) for _, r in below)}) "
        f"remain collapsed"
        if below else "no rung below it is collapsed (the lowest rung itself transitions)"
    )

    above_verdicts = [v for v, _ in above]
    if all(v in ("canonical AC", "robust AC") for v in above_verdicts):
        upward = (
            f"all rungs at and above {first_arch} reach anti-collapse "
            f"(verdict counts in the table)"
        )
    else:
        upward = (
            "anti-collapse is non-monotone along the ladder: at least one rung "
            f"above {first_arch} falls back to collapsed (see the per-architecture "
            "row in the table)"
        )

    return (
        f"**Headline verdict.** The capacity-ladder transition first occurs at "
        f"**{first_arch}**: {below_summary}, while {upward}. This pins the "
        f"minimum architectural rung at which spontaneous training realizes "
        f"stable anti-collapse on this task to {first_arch}."
    )


# ----- Markdown rendering -----
def render_markdown(
    records: List[Dict[str, object]],
    exp2_dir: Path,
    profile: str,
    seeds: Optional[str],
) -> str:
    """Render the full markdown summary."""
    n_seeds = max((int(r.get("n_seeds", 0)) for r in records), default=0)
    seeds_str = seeds if seeds else f"{n_seeds} (auto-detected)"

    lines: List[str] = []
    lines.append(f"# Experiment 2 — Dynamical phase trajectory across the capacity ladder")
    lines.append("")
    lines.append(f"**Source:** `{exp2_dir}`")
    lines.append(f"**Profile:** `{profile}`")
    lines.append(f"**Seeds:** {seeds_str}")
    lines.append(
        "**Setup:** same long-memory regression task and diagnostic pipeline as "
        "Experiment 1, with no heavy-tailed gradient injection. Architectures "
        "are walked in capacity-ladder order; each architecture's per-seed "
        "aggregates were computed by `main_phase_trajectory.py` and reduced to the "
        "headline numbers below."
    )
    lines.append("")
    lines.append(build_headline(records))
    lines.append("")

    # --- Phase-verdict-per-rung table ---
    lines.append("## Per-architecture verdicts")
    lines.append("")
    lines.append(
        "| Rung | n_seeds | Majority phase | Counts (collapsed / canonical AC / robust AC) "
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

    # --- Envelope / dynamic range / drift table ---
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        "| Rung | β̂_env(final) ± SE | β̂_tail (boot 90% CI)              | Δζ (final) "
        "| κ_tail @ q_low=0.10 (90% CI)        | α_ECF(final) | t_cross (crossed/seeds) |"
    )
    lines.append(
        "|------|---------------------|-------------------------------------|------------"
        "|--------------------------------------|--------------|-------------------------|"
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
        t_cross = f"{r.get('t_cross_n_crossed', 0)}/{r.get('t_cross_n_seeds', 0)}"
        lines.append(
            f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {beta_env} | {boot} | "
            f"{_fmt(r.get('delta_zeta'), 3)} | {kappa} | "
            f"{_fmt(r.get('alpha_ecf_final'), 3)} | {t_cross} |"
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
            f"{_fmt(r.get('delta_zeta'), 3)} | "
            f"{_fmt(r.get('tau_q50_final'), 1)} | "
            f"{_fmt(r.get('tau_q90_final'), 1)} | "
            f"{_fmt(r.get('tau_q99_final'), 1)} |"
        )
    lines.append("")
    lines.append(
        "The ζ columns show the log-rate coordinate modeled by the drift SDE; "
        "the τ columns summarize the right tail that enters the "
        "spectrum-to-envelope correspondence. Large isolated values of τ are "
        "reported here, but they should be interpreted together with the "
        "τ-CCDF figure: anti-collapse requires a resolved scaling window, not "
        "a few far-tail outliers."
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
        "- Macro-envelope `f(ℓ)` here is the canonical zero-order `μ₀` kernel "
        "(see `diagnostics.compute_macro_envelope_comparison`). The legacy "
        "`μ₀+μ₁` envelope and the per-lag correction ratio are still written "
        "to `<arch>_envelope_audit.csv` for audit."
    )
    lines.append(
        "- Phase labels come from the canonical Table-1 rule applied to "
        "`<arch>_final_phase.json` at the end of training; the majority label "
        "is reported as the headline phase for each rung."
    )
    lines.append(
        "- Drift κ_tail is read at the primary `q_low=0.10` slice. The full "
        "quantile-cut sweep is in `drift_<arch>/tail_saturation.json`."
    )
    lines.append(
        "- The final spectrum table is paired with `exp2_time_scale_spectrum.png`: "
        "`p(ζ)` is the drift-coordinate view, while the log-log τ-CCDF is the "
        "direct visual check for a regularly varying time-scale tail."
    )
    lines.append(
        "- This file is regenerated by `write_exp2_summary.py`; hand-edit only "
        "after appending a `## Manual notes` section so subsequent regenerations "
        "preserve the human commentary."
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
        "--architectures", type=str, default="shared,diag,gru,lstm",
        help="Comma-separated architecture short names in capacity-ladder order.",
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

    records = [collect_arch_record(args.exp2_dir, arch) for arch in architectures]
    md = render_markdown(records, args.exp2_dir, profile, args.seeds)
    with open(output, "w") as f:
        f.write(md)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
