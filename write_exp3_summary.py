#!/usr/bin/env python3
"""Write the Experiment 3 forcing-ablation markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


MODEL_LABEL = {
    "diag": "DiagGate",
    "shared": "SharedGate",
    "gru": "GRU",
    "lstm": "LSTM",
    "const": "ConstGate",
}

ABLATION_ORDER = ["baseline", "batch_ablation", "clip_ablation", "winsorize_ablation"]
ABLATION_LABEL = {
    "baseline": "baseline",
    "batch_ablation": "batch",
    "clip_ablation": "clip",
    "winsorize_ablation": "winsor",
}


@dataclass(frozen=True)
class Condition:
    key: str
    dirname: str
    ablation: str
    value: Optional[float]

    @property
    def label(self) -> str:
        if self.ablation == "baseline":
            return "baseline"
        return f"{ABLATION_LABEL.get(self.ablation, self.ablation)} {self.value:g}"


def _condition_from_dir(path: Path) -> Optional[Condition]:
    name = path.name
    if name == "condition_baseline":
        return Condition("baseline", name, "baseline", None)
    patterns = [
        ("batch_ablation", r"condition_batch_ablation_([0-9.]+)"),
        ("clip_ablation", r"condition_clip_ablation_([0-9.]+)"),
        ("winsorize_ablation", r"condition_winsorize_ablation_([0-9.]+)"),
    ]
    for ablation, pat in patterns:
        m = re.fullmatch(pat, name)
        if m:
            value = float(m.group(1))
            return Condition(f"{ablation}_{m.group(1)}", name, ablation, value)
    return None


def _condition_sort_key(cond: Condition) -> Tuple[int, float]:
    idx = ABLATION_ORDER.index(cond.ablation) if cond.ablation in ABLATION_ORDER else 99
    if cond.ablation == "baseline":
        return (idx, 0.0)
    value = float(cond.value if cond.value is not None else 0.0)
    if cond.ablation == "batch_ablation":
        strength = value
    else:
        strength = -value
    return (idx, strength)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _fmt(x, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _float(row: Dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            try:
                return float(row[name])
            except Exception:
                pass
    return float("nan")


def _discover_conditions(agg_dir: Path, model: str) -> List[Condition]:
    root = agg_dir / model
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir():
            cond = _condition_from_dir(child)
            if cond is not None:
                out.append(cond)
    return sorted(out, key=_condition_sort_key)


def _last_row(agg_dir: Path, model: str, cond: Condition) -> Dict[str, str]:
    cdir = agg_dir / model / cond.dirname
    for fname in ("phase_trajectory_aggregated.csv", "phase_trajectory.csv"):
        rows = _read_csv(cdir / fname)
        if rows:
            return rows[-1]
    return {}


def _final_phase(agg_dir: Path, model: str, cond: Condition) -> Dict:
    return _read_json(agg_dir / model / cond.dirname / f"{model}_final_phase.json") or {}


def _phase_label(phase: Dict) -> str:
    label = str(phase.get("phase_label", "")).strip()
    majority = str(phase.get("majority_phase_label", "")).strip()
    if label == "mixed" and majority:
        return f"mixed / maj: {majority}"
    return label or majority or "—"


def _checkpoint_epoch(path: Path) -> int:
    m = re.search(r"ckpt_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _load_final_taus(agg_dir: Path, model: str, cond: Condition) -> Optional[np.ndarray]:
    tau_dir = agg_dir / model / cond.dirname / "checkpoint_taus"
    if not tau_dir.exists():
        return None
    npy_files = sorted(tau_dir.glob("ckpt_*_taus.npy"), key=_checkpoint_epoch)
    csv_files = sorted(tau_dir.glob("ckpt_*_taus.csv"), key=_checkpoint_epoch)
    if npy_files:
        arr = np.load(npy_files[-1])
    elif csv_files:
        vals = []
        with csv_files[-1].open() as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    try:
                        vals.append(float(cell))
                    except Exception:
                        pass
        arr = np.asarray(vals, dtype=float)
    else:
        return None
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return arr if arr.size else None


def _spectrum_record(agg_dir: Path, model: str, cond: Condition) -> Optional[Dict[str, float]]:
    tau = _load_final_taus(agg_dir, model, cond)
    if tau is None:
        return None
    z = -np.log(tau)
    return {
        "n": float(tau.size),
        "zeta_q10": float(np.quantile(z, 0.10)),
        "zeta_med": float(np.quantile(z, 0.50)),
        "zeta_q90": float(np.quantile(z, 0.90)),
        "delta_zeta": float(np.quantile(z, 0.90) - np.quantile(z, 0.10)),
        "tau_med": float(np.quantile(tau, 0.50)),
        "tau_q90": float(np.quantile(tau, 0.90)),
        "tau_q99": float(np.quantile(tau, 0.99)),
    }


def _infer_output(exp3_dir: Path) -> Path:
    # exp3_dir is usually results/exp3_forcing_<profile>/adamw/aggregated.
    parts = exp3_dir.resolve().parts
    for idx, part in enumerate(parts):
        if part.startswith("exp3_forcing_"):
            profile = part.removeprefix("exp3_forcing_")
            return Path(*parts[:idx]) / f"exp3_forcing_{profile}_results_summary.md"
    return Path("results/exp3_forcing_results_summary.md")


def _parse_models(raw: str) -> List[str]:
    return [m.strip().lower() for m in raw.split(",") if m.strip()]


def build_summary(agg_dir: Path, models: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# Experiment 3 — Stochastic-forcing ablation")
    lines.append("")
    lines.append(f"**Source:** `{agg_dir}`")
    lines.append(f"**Models:** {', '.join(MODEL_LABEL.get(m, m) for m in models)}")
    lines.append("")
    lines.append(
        "**Headline placeholder.** This summary is generated from the Exp3 "
        "ablation aggregates. The manuscript verdict should be written after "
        "checking whether forcing suppression contracts the spectrum, moves the "
        "phase labels toward collapse, and localizes a threshold bracket in the "
        "intervention ladders."
    )
    lines.append("")

    lines.append("## Final diagnostics by condition")
    lines.append("")
    lines.append("| Model | Condition | phase | alpha(final) | beta_env(final) | beta_med(final) | Delta zeta(final) | envelope | tail beta [90% CI] |")
    lines.append("|------|-----------|-------|-------------:|----------------:|----------------:|-------------------:|----------|--------------------|")
    for model in models:
        for cond in _discover_conditions(agg_dir, model):
            row = _last_row(agg_dir, model, cond)
            phase = _final_phase(agg_dir, model, cond)
            tail = "—"
            if phase:
                tail = f"{_fmt(phase.get('tail_beta_hat'), 3)} [{_fmt(phase.get('boot_beta_lo'), 3)}, {_fmt(phase.get('boot_beta_hi'), 3)}]"
            lines.append(
                f"| {MODEL_LABEL.get(model, model)} | {cond.label} | {_phase_label(phase)} | "
                f"{_fmt(_float(row, 'alpha_hat', 'alpha_hat_mean'), 3)} | "
                f"{_fmt(_float(row, 'beta_env', 'beta_env_mean'), 3)} | "
                f"{_fmt(_float(row, 'beta_median', 'beta_median_mean'), 3)} | "
                f"{_fmt(_float(row, 'delta_zeta', 'delta_zeta_mean'), 3)} | "
                f"{phase.get('majority_envelope_winner', phase.get('envelope_winner', '—')) if phase else '—'} | "
                f"{tail} |"
            )
    lines.append("")

    lines.append("## Final time-scale spectrum")
    lines.append("")
    lines.append("| Model | Condition | n finite tau | zeta q10 | zeta median | zeta q90 | Delta zeta | tau median | tau q90 | tau q99 |")
    lines.append("|------|-----------|-------------:|---------:|------------:|---------:|-----------:|-----------:|--------:|--------:|")
    for model in models:
        for cond in _discover_conditions(agg_dir, model):
            rec = _spectrum_record(agg_dir, model, cond)
            if rec is None:
                continue
            lines.append(
                f"| {MODEL_LABEL.get(model, model)} | {cond.label} | {int(rec['n'])} | "
                f"{_fmt(rec['zeta_q10'], 3)} | {_fmt(rec['zeta_med'], 3)} | {_fmt(rec['zeta_q90'], 3)} | "
                f"{_fmt(rec['delta_zeta'], 3)} | {_fmt(rec['tau_med'], 1)} | {_fmt(rec['tau_q90'], 1)} | {_fmt(rec['tau_q99'], 1)} |"
            )
    lines.append("")

    brackets = _read_json(agg_dir / "threshold_brackets.json")
    lines.append("## Threshold brackets")
    lines.append("")
    if isinstance(brackets, dict):
        lines.append("| Model | Ablation | status | last surviving AC | first collapsed | width |")
        lines.append("|------|----------|--------|-------------------:|----------------:|------:|")
        for model in models:
            for ablation, bracket in (brackets.get(model, {}) or {}).items():
                lines.append(
                    f"| {MODEL_LABEL.get(model, model)} | {ABLATION_LABEL.get(ablation, ablation)} | "
                    f"{bracket.get('status', '—')} | {_fmt(bracket.get('last_anti_collapsed'), 4)} | "
                    f"{_fmt(bracket.get('first_collapsed'), 4)} | {_fmt(bracket.get('bracket_width'), 4)} |"
                )
    else:
        lines.append("No `threshold_brackets.json` found yet.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Conditions are ordered from baseline to progressively stronger intervention within each ablation family.")
    lines.append("- Batch-size suppression is stronger at larger batch size; clipping and winsorization are stronger at smaller thresholds/percentiles.")
    lines.append("- This file is regenerated by `write_exp3_summary.py`; add durable hand-written interpretation below a `## Manual notes` heading.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Experiment 3 forcing-ablation summary markdown.")
    parser.add_argument("--agg_dir", type=Path, required=True, help="Path to exp3 aggregated directory.")
    parser.add_argument("--models", type=str, default="diag", help="Comma-separated model short names.")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path.")
    args = parser.parse_args()

    if not args.agg_dir.exists():
        raise SystemExit(f"ERROR: --agg_dir does not exist: {args.agg_dir}")
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("ERROR: no models parsed from --models")
    output = args.output or _infer_output(args.agg_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_summary(args.agg_dir, models))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
