#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the slow-mode forcing tail estimator on synthetic data.

The experiment-facing diagnostic estimates the tail heaviness of slow-mode
forcing increments with a *calibration-anchored* extreme-value-index estimator
(``estimate_tail_index_calibrated`` in the repo-root ``diagnostics.py``): the
moment (DEdH) extreme-value index gamma is the primary statistic, bias-corrected
by inverting it through a matched-n synthetic-stable calibration curve to an
in-range effective alpha. This script audits that production estimator on
synthetic samples with known tail behaviour.

Outputs:
  diagnostics/forcing_tail_estimator_validation.md
  diagnostics/forcing_tail_estimator_validation.csv
  diagnostics/forcing_tail_estimator_validation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
ROOT_DIAGNOSTICS = os.path.join(PROJECT_ROOT, "diagnostics.py")


def _load_root_diagnostics():
    """Load the repo-root diagnostics.py without colliding with this package."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    spec = importlib.util.spec_from_file_location("_anticollapse_root_diagnostics", ROOT_DIAGNOSTICS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import diagnostics.py from {ROOT_DIAGNOSTICS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diag = _load_root_diagnostics()


@dataclass(frozen=True)
class TrialResult:
    case: str
    n: int
    true_alpha: Optional[float]
    alpha_eff: float
    xi_hat: float
    gaussian_p_value: float
    gaussian_reject: int
    substantively_heavy: int
    reliable: int
    k_selected: int


def _parse_float_list(text: str) -> List[float]:
    return [float(s.strip()) for s in str(text).split(",") if s.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(s.strip()) for s in str(text).split(",") if s.strip()]


def _finite(a: Sequence[float]) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    return x[np.isfinite(x)]


def _percentile(a: Sequence[float], q: float) -> float:
    x = _finite(a)
    return float(np.percentile(x, q)) if x.size else float("nan")


def _mean(a: Sequence[float]) -> float:
    x = _finite(a)
    return float(np.mean(x)) if x.size else float("nan")


def _rmse(a: Sequence[float], target: float) -> float:
    x = _finite(a)
    return float(np.sqrt(np.mean((x - float(target)) ** 2))) if x.size else float("nan")


def _wilson_interval(successes: int, total: int, z: float = 1.6448536269514722) -> Tuple[float, float]:
    """Wilson interval for a binomial rate; z is 90 percent by default."""
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _symmetric_stable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(diag._cms_symmetric_stable_np(float(alpha), int(n), rng), dtype=np.float64)


def _gaussian_stable_mixture(alpha, n, fraction, stable_scale, rng) -> np.ndarray:
    x = rng.normal(loc=0.0, scale=1.0, size=int(n)).astype(np.float64)
    mask = rng.random(int(n)) < float(fraction)
    m = int(mask.sum())
    if m:
        x[mask] += float(stable_scale) * _symmetric_stable(alpha, m, rng)
    return x


def _estimate(signed_samples: np.ndarray, args, seed: int) -> Tuple[float, float, float, int, int, int, int]:
    """Call the production calibrated estimator and return the key fields."""
    est = diag.estimate_tail_index_calibrated(
        signed_samples,
        k_frac=float(args.k_frac),
        k_min=int(args.k_min),
        calib_B=int(args.calib_reps),
        ci_B=int(args.ci_B),
        substantive_alpha_threshold=float(args.substantive_alpha),
        gaussian_test_alpha=float(args.gaussian_test_alpha),
        seed=int(seed),
    )
    return (
        float(est["alpha_eff"]),
        float(est["xi_hat"]),
        float(est["gaussian_p_value"]),
        int(est["gaussian_reject"]),
        int(est["substantively_heavy"]),
        int(est["reliable"]),
        int(est["k_selected"]),
    )


def _run_pure_trials(args, rng: np.random.Generator) -> List[TrialResult]:
    rows: List[TrialResult] = []
    t = 0
    for n in args.n_grid:
        for alpha in args.alphas:
            for _ in range(int(args.reps)):
                x = _symmetric_stable(alpha, int(n), rng)
                a_eff, xi, pval, reject, heavy, rel, k = _estimate(x, args, seed=1000 + t)
                t += 1
                rows.append(TrialResult("pure_sas", int(n), float(alpha), a_eff, xi,
                                        pval, reject, heavy, rel, k))
    return rows


def _run_mixture_trials(args, rng: np.random.Generator) -> List[TrialResult]:
    rows: List[TrialResult] = []
    t = 0
    for n in args.n_grid:
        for fraction in args.mixture_fracs:
            case = f"gaussian_plus_{fraction:.3g}_stable"
            for _ in range(int(args.reps)):
                x = _gaussian_stable_mixture(args.mixture_alpha, int(n), float(fraction),
                                             args.mixture_stable_scale, rng)
                a_eff, xi, pval, reject, heavy, rel, k = _estimate(x, args, seed=5000 + t)
                t += 1
                rows.append(TrialResult(case, int(n), float(args.mixture_alpha), a_eff, xi,
                                        pval, reject, heavy, rel, k))
    return rows


def _group(rows: Iterable[TrialResult]) -> Dict[Tuple[str, int, Optional[float]], List[TrialResult]]:
    g: Dict[Tuple[str, int, Optional[float]], List[TrialResult]] = {}
    for r in rows:
        g.setdefault((r.case, r.n, r.true_alpha), []).append(r)
    return g


def _summarize(rows: List[TrialResult]) -> Dict[str, object]:
    a_eff = np.asarray([r.alpha_eff for r in rows], dtype=np.float64)
    xi = np.asarray([r.xi_hat for r in rows], dtype=np.float64)
    pvals = np.asarray([r.gaussian_p_value for r in rows], dtype=np.float64)
    reject = np.asarray([r.gaussian_reject for r in rows], dtype=np.float64)
    ks = np.asarray([r.k_selected for r in rows], dtype=np.float64)
    heavy = np.asarray([r.substantively_heavy for r in rows], dtype=np.float64)
    reliable = np.asarray([r.reliable for r in rows], dtype=np.float64)
    n_tr = len(rows)
    true_alpha = rows[0].true_alpha
    rej_lo, rej_hi = _wilson_interval(int(np.nansum(reject)), n_tr)
    h_lo, h_hi = _wilson_interval(int(np.nansum(heavy)), n_tr)
    r_lo, r_hi = _wilson_interval(int(np.nansum(reliable)), n_tr)
    out: Dict[str, object] = {
        "n_trials": int(n_tr),
        "alpha_eff_p05": _percentile(a_eff, 5),
        "alpha_eff_p50": _percentile(a_eff, 50),
        "alpha_eff_p95": _percentile(a_eff, 95),
        "xi_p50": _percentile(xi, 50),
        "gaussian_p_value_p50": _percentile(pvals, 50),
        "gaussian_reject_rate": float(np.nanmean(reject)) if n_tr else float("nan"),
        "gaussian_reject_rate_lo90": rej_lo,
        "gaussian_reject_rate_hi90": rej_hi,
        "k_p50": _percentile(ks, 50),
        "substantively_heavy_rate": float(np.nanmean(heavy)) if n_tr else float("nan"),
        "substantively_heavy_rate_lo90": h_lo,
        "substantively_heavy_rate_hi90": h_hi,
        "reliable_rate": float(np.nanmean(reliable)) if n_tr else float("nan"),
        "reliable_rate_lo90": r_lo,
        "reliable_rate_hi90": r_hi,
    }
    # alpha=2 is Gaussian (light): effective alpha should sit near 2 and the
    # heavy rate is the false-positive rate; bias against 2.0 is meaningful here
    # because the calibrated estimator maps the light end to ~2 by construction.
    if true_alpha is not None:
        out["alpha_eff_bias_median"] = (out["alpha_eff_p50"] - float(true_alpha)
                                        if np.isfinite(out["alpha_eff_p50"]) else float("nan"))
        out["alpha_eff_rmse"] = _rmse(a_eff, float(true_alpha))
    return out


def _build_summary_rows(rows: List[TrialResult]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for (case, n, ta), grp in sorted(_group(rows).items(),
                                     key=lambda x: (x[0][0], x[0][1], -1 if x[0][2] is None else x[0][2])):
        row: Dict[str, object] = {"case": case, "n": int(n),
                                  "true_alpha": float(ta) if ta is not None else ""}
        row.update(_summarize(grp))
        out.append(row)
    return out


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "nan"
    if abs(xf) >= 1000 or (0 < abs(xf) < 0.001):
        return f"{xf:.{digits}e}"
    return f"{xf:.{digits}f}"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    # keep a stable, human-friendly leading order
    lead = ["case", "n", "true_alpha", "n_trials", "alpha_eff_p50", "alpha_eff_p05",
            "alpha_eff_p95", "alpha_eff_bias_median", "alpha_eff_rmse", "xi_p50",
            "gaussian_p_value_p50", "gaussian_reject_rate",
            "substantively_heavy_rate", "reliable_rate", "k_p50"]
    fieldnames = lead + [c for c in fieldnames if c not in lead]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _table(rows: List[Dict[str, object]], cols: Sequence[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown(path: str, args, summary_rows: List[Dict[str, object]]) -> None:
    pure = [r for r in summary_rows if r["case"] == "pure_sas"]
    mixtures = [r for r in summary_rows if str(r["case"]).startswith("gaussian_plus_")]
    alpha16 = [r for r in pure if abs(float(r["true_alpha"]) - 1.6) < 1e-12]
    alpha2 = [r for r in pure if abs(float(r["true_alpha"]) - 2.0) < 1e-12]

    L: List[str] = []
    L.append("# Slow-mode forcing tail estimator validation")
    L.append("")
    L.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    L.append(f"Host: `{platform.platform()}`")
    L.append("")
    L.append("## Purpose")
    L.append("")
    L.append(
        "This report audits the production estimator "
        "`estimate_tail_index_calibrated`: the moment (DEdH) extreme-value index "
        "gamma, bias-corrected by inversion through a matched-n synthetic-stable "
        "calibration curve to an in-range effective alpha. The key checks are that "
        "the effective alpha recovers the true alpha, that the Gaussian/light case "
        "(alpha=2) maps to effective alpha ~2 (not a spurious >2 value) with a low "
        "Gaussian-boundary false-rejection rate, and that genuinely heavy tails "
        "both reject the Gaussian boundary and cross the configured substantive "
        "effect-size cutoff."
    )
    L.append("")
    L.append("## Configuration")
    L.append("")
    L.append(f"- Monte Carlo test trials per condition: `{args.reps}`")
    L.append(f"- Synthetic-stable draws per alpha for the calibration curve (calib_B): `{args.calib_reps}`")
    L.append(f"- Per-trial effective-alpha CI bootstraps (ci_B): `{args.ci_B}`")
    L.append(f"- Gaussian-boundary test level: `{args.gaussian_test_alpha}`")
    L.append(f"- Substantive heavy-tail cutoff: `alpha_eff_hi <= {args.substantive_alpha}`")
    L.append(f"- Sample sizes: `{','.join(str(x) for x in args.n_grid)}`")
    L.append(f"- Pure SaS alpha grid: `{','.join(str(x) for x in args.alphas)}`")
    L.append(f"- k_frac: `{args.k_frac}`  k_min: `{args.k_min}`")
    L.append(f"- Mixture alpha / scale / fractions: `{args.mixture_alpha}` / `{args.mixture_stable_scale}` / "
             f"`{','.join(str(x) for x in args.mixture_fracs)}`")
    L.append(f"- Random seed: `{args.seed}`")
    L.append(f"- Command: `{' '.join(sys.argv)}`")
    L.append("")
    L.append("## Main checks")
    L.append("")
    if alpha2:
        L.append("Gaussian boundary (alpha=2): effective alpha, false rejection, and false substantive-heavy rate")
        for r in alpha2:
            L.append(
                f"- n={r['n']}: effective alpha p50 = {float(r['alpha_eff_p50']):.3f}; "
                f"false Gaussian rejection {100.0 * float(r['gaussian_reject_rate']):.1f}% "
                f"(90% Wilson {100.0 * float(r['gaussian_reject_rate_lo90']):.1f}-"
                f"{100.0 * float(r['gaussian_reject_rate_hi90']):.1f}%); "
                f"false substantive-heavy {100.0 * float(r['substantively_heavy_rate']):.1f}% "
                f"(90% Wilson {100.0 * float(r['substantively_heavy_rate_lo90']):.1f}-"
                f"{100.0 * float(r['substantively_heavy_rate_hi90']):.1f}%)."
            )
    if alpha16:
        L.append("")
        L.append("Detection at alpha=1.6 (Gaussian rejection and effective-alpha recovery)")
        for r in alpha16:
            L.append(
                f"- n={r['n']}: Gaussian rejection {100.0 * float(r['gaussian_reject_rate']):.1f}%; "
                f"substantive-heavy {100.0 * float(r['substantively_heavy_rate']):.1f}%; "
                f"effective alpha p50 = {float(r['alpha_eff_p50']):.3f} "
                f"(bias {float(r['alpha_eff_bias_median']):+.3f})."
            )
    L.append("")
    L.append("Interpretation guide:")
    L.append("- Effective-alpha bias near zero across alpha<2 shows the calibration inversion removes the finite-sample tail bias and stays in range (0,2].")
    L.append("- At alpha=2 the effective alpha should sit near 2 and the false substantive-heavy rate should be low; this is the Gaussian-boundary (light-tail) control and is out-of-sample (test draws are independent of the calibration draws).")
    L.append("- A high Gaussian-rejection rate at alpha in [1.5,1.7] indicates power against the light-tail boundary; the substantive-heavy rate adds the effect-size gate used for internal calibration.")
    L.append("- Mixture rows quantify how much rare heavy-tailed contamination is needed before the estimator resolves it.")
    L.append("- The reliability rate flags agreement between the moment and Hill effective alphas; it is a numerical sanity flag, not a phase classifier.")
    L.append("")
    L.append("## Pure symmetric stable laws")
    L.append("")
    L.append(_table(pure, ["n", "true_alpha", "alpha_eff_p50", "alpha_eff_p05", "alpha_eff_p95",
                           "alpha_eff_bias_median", "alpha_eff_rmse", "xi_p50",
                           "gaussian_p_value_p50", "gaussian_reject_rate",
                           "substantively_heavy_rate", "reliable_rate", "k_p50"]))
    L.append("")
    if mixtures:
        L.append("## Gaussian core plus rare stable contamination")
        L.append("")
        L.append(_table(mixtures, ["case", "n", "true_alpha", "alpha_eff_p50", "alpha_eff_p05",
                                   "alpha_eff_p95", "xi_p50", "gaussian_p_value_p50",
                                   "gaussian_reject_rate", "substantively_heavy_rate",
                                   "reliable_rate", "k_p50"]))
        L.append("")
    L.append("## Notes")
    L.append("")
    L.append(
        "The validation is estimator-level: it does not assert that update-space "
        "forcing increments in a trained network are alpha-stable; it checks that the "
        "production calibrated estimator behaves correctly when the tail law is "
        "controlled synthetically. Set calib_B equal to the runtime "
        "`forcing_tail_bootstrap_B` to audit the operational calibration."
    )
    L.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate the calibrated slow-mode forcing tail estimator on synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default=os.path.join(THIS_DIR, "forcing_tail_estimator_validation.md"))
    p.add_argument("--csv", default=os.path.join(THIS_DIR, "forcing_tail_estimator_validation.csv"))
    p.add_argument("--json", default=os.path.join(THIS_DIR, "forcing_tail_estimator_validation.json"))
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--reps", type=int, default=200, help="Monte Carlo test trials per condition.")
    p.add_argument("--calib-reps", type=int, default=250,
                   help="Synthetic-stable draws per alpha for the estimator's internal calibration "
                        "(maps to calib_B). Set to the runtime forcing_tail_bootstrap_B to mirror it.")
    p.add_argument("--ci-B", type=int, default=0,
                   help="Per-trial effective-alpha CI bootstraps; 0 skips CIs for speed.")
    p.add_argument("--n-grid", type=str, default="2000,8000,20000")
    p.add_argument("--alphas", type=str, default="1.5,1.6,1.7,1.8,1.9,1.95,2.0")
    p.add_argument("--k-frac", type=float, default=0.08)
    p.add_argument("--k-min", type=int, default=50)
    p.add_argument("--substantive-alpha", type=float, default=1.8,
                   help="Substantive-heaviness cutoff applied to alpha_eff_hi.")
    p.add_argument("--gaussian-test-alpha", type=float, default=0.05,
                   help="One-sided calibrated Gaussian-boundary test level.")
    p.add_argument("--mixture-alpha", type=float, default=1.6)
    p.add_argument("--mixture-fracs", type=str, default="0.01,0.03,0.10")
    p.add_argument("--mixture-stable-scale", type=float, default=8.0)
    p.add_argument("--skip-mixtures", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.n_grid = _parse_int_list(args.n_grid)
    args.alphas = _parse_float_list(args.alphas)
    args.mixture_fracs = _parse_float_list(args.mixture_fracs)

    if not any(abs(a - 2.0) < 1e-9 for a in args.alphas):
        sys.exit("ERROR: --alphas must include 2.0 (the Gaussian-boundary / light control).")
    if not any(1.5 <= a <= 1.7 for a in args.alphas):
        print("WARNING: no alpha in [1.5,1.7] in --alphas; the alpha=1.6 power readout will be empty.",
              file=sys.stderr)

    rng = np.random.default_rng(int(args.seed))
    pure_rows = _run_pure_trials(args, rng)
    mix_rows: List[TrialResult] = [] if args.skip_mixtures else _run_mixture_trials(args, rng)
    summary_rows = _build_summary_rows(pure_rows + mix_rows)

    _write_csv(args.csv, summary_rows)
    with open(args.json, "w") as f:
        json.dump(_json_safe({
            "config": {
                "seed": int(args.seed), "reps": int(args.reps), "calib_reps": int(args.calib_reps),
                "ci_B": int(args.ci_B), "n_grid": args.n_grid, "alphas": args.alphas,
                "k_frac": float(args.k_frac), "k_min": int(args.k_min),
                "substantive_alpha": float(args.substantive_alpha),
                "gaussian_test_alpha": float(args.gaussian_test_alpha),
                "mixture_alpha": float(args.mixture_alpha), "mixture_fracs": args.mixture_fracs,
                "mixture_stable_scale": float(args.mixture_stable_scale),
                "skip_mixtures": bool(args.skip_mixtures),
            },
            "summary": summary_rows,
        }), f, indent=2)
    _write_markdown(args.out, args, summary_rows)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
