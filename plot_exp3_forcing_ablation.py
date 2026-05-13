#!/usr/bin/env python3
"""Manuscript-bound figures for Experiment 3 forcing ablations.

The simulation/aggregation pipeline for Experiment 3 is handled by
``main_forcing_ablation.py``.  This script is the lightweight presentation layer used by
``plot_all.sh``: it reads the aggregated condition directories and writes the
Overleaf-bound Exp3 figures into a single folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABEL = {
    "diag": "DiagGate",
    "shared": "SharedGate",
    "gru": "GRU",
    "lstm": "LSTM",
    "const": "ConstGate",
}

PHASE_COLOR = {
    "collapsed": "#b8b8b8",
    "concentrated anti-collapse": "#f2b84b",
    "anti-collapse": "#3aa050",
    "mixed": "#8b78c7",
    "": "#dddddd",
}

ABLATION_ORDER = ["batch_ablation", "clip_ablation", "winsorize_ablation"]
ABLATION_LABEL = {
    "batch_ablation": "Batch size",
    "clip_ablation": "Gradient clip",
    "winsorize_ablation": "Winsorization",
}
ABLATION_SHORT = {
    "batch_ablation": "batch",
    "clip_ablation": "clip",
    "winsorize_ablation": "winsor",
}
ABLATION_COLOR = {
    "baseline": "#1f77b4",
    "batch_ablation": "#d95f02",
    "clip_ablation": "#1b9e77",
    "winsorize_ablation": "#7570b3",
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
        if self.ablation == "batch_ablation":
            return f"batch {self.value:g}"
        if self.ablation == "clip_ablation":
            return f"clip {self.value:g}"
        if self.ablation == "winsorize_ablation":
            return f"winsor {self.value:g}"
        return self.key


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
    if cond.ablation == "baseline":
        return (-1, 0.0)
    idx = ABLATION_ORDER.index(cond.ablation) if cond.ablation in ABLATION_ORDER else 99
    value = float(cond.value if cond.value is not None else 0.0)
    if cond.ablation == "batch_ablation":
        strength = value
    else:
        # Smaller clip thresholds / winsorization percentiles are stronger.
        strength = -value
    return (idx, strength)


def _read_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _float(row: Dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            try:
                return float(row[name])
            except Exception:
                pass
    return float("nan")


def _series(rows: List[Dict[str, str]], col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([_float(r, "step", "epoch") for r in rows], dtype=float)
    y = np.asarray([_float(r, col, f"{col}_mean") for r in rows], dtype=float)
    se = np.asarray([_float(r, f"{col}_se") for r in rows], dtype=float)
    se = np.where(np.isfinite(se), se, 0.0)
    return x, y, se


def _discover_conditions(agg_dir: Path, model: str) -> List[Condition]:
    root = agg_dir / model
    if not root.exists():
        return []
    conds = []
    for child in root.iterdir():
        if child.is_dir():
            cond = _condition_from_dir(child)
            if cond is not None:
                conds.append(cond)
    return sorted(conds, key=_condition_sort_key)


def _load_condition_rows(agg_dir: Path, model: str, cond: Condition) -> Optional[List[Dict[str, str]]]:
    cdir = agg_dir / model / cond.dirname
    for fname in ("phase_trajectory_aggregated.csv", "phase_trajectory.csv"):
        rows = _read_csv_rows(cdir / fname)
        if rows:
            return rows
    return None


def _load_final_phase(agg_dir: Path, model: str, cond: Condition) -> Dict:
    cdir = agg_dir / model / cond.dirname
    return _read_json(cdir / f"{model}_final_phase.json") or {}


def _phase_label(phase: Dict, rows: Optional[List[Dict[str, str]]] = None) -> str:
    label = str(phase.get("phase_label", "")).strip()
    majority = str(phase.get("majority_phase_label", "")).strip()
    if label == "mixed" and majority:
        return majority
    if label:
        return label
    if rows:
        for row in reversed(rows):
            val = str(row.get("phase_label", "")).strip()
            if val:
                return val
    return ""


def _phase_short(label: str) -> str:
    label = str(label).strip()
    if label == "concentrated anti-collapse":
        return "canonical AC"
    if label == "anti-collapse":
        return "robust AC"
    return label or "missing"


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


def _ccdf(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        return x, x
    y = (n - np.arange(1, n + 1)) / max(1, n)
    return x, y


def _strongest_conditions(conditions: Sequence[Condition]) -> List[Condition]:
    selected = [c for c in conditions if c.ablation == "baseline"]
    for ablation in ABLATION_ORDER:
        group = [c for c in conditions if c.ablation == ablation and c.value is not None]
        if not group:
            continue
        if ablation == "batch_ablation":
            selected.append(max(group, key=lambda c: float(c.value)))
        else:
            selected.append(min(group, key=lambda c: float(c.value)))
    return selected


def plot_forcing_beta_trajectory(agg_dir: Path, models: Sequence[str], outpath: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), sharex=False)

    for col, ablation in enumerate(ABLATION_ORDER):
        ax_a = axes[0, col]
        ax_b = axes[1, col]
        for model_idx, model in enumerate(models):
            conds = _discover_conditions(agg_dir, model)
            selected = [c for c in conds if c.ablation in ("baseline", ablation)]
            for cond in selected:
                rows = _load_condition_rows(agg_dir, model, cond)
                if not rows:
                    continue
                color = ABLATION_COLOR.get(cond.ablation, f"C{model_idx}")
                linestyle = "-" if cond.ablation == "baseline" else "--"
                alpha = 0.95 if cond.ablation == "baseline" else 0.72
                label = f"{MODEL_LABEL.get(model, model)} {cond.label}" if len(models) > 1 else cond.label

                x, y, se = _series(rows, "alpha_hat")
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax_a.plot(x[mask], y[mask], color=color, linestyle=linestyle, alpha=alpha, label=label)
                    ax_a.fill_between(x[mask], y[mask] - se[mask], y[mask] + se[mask], color=color, alpha=0.09)

                x, y, se = _series(rows, "beta_env")
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax_b.plot(x[mask], y[mask], color=color, linestyle=linestyle, alpha=alpha, label=label)
                    ax_b.fill_between(x[mask], y[mask] - se[mask], y[mask] + se[mask], color=color, alpha=0.09)

        ax_a.axhline(2.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_b.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_a.set_title(ABLATION_LABEL[ablation])
        ax_a.set_ylabel(r"Forcing proxy $\hat\alpha$")
        ax_b.set_ylabel(r"Envelope exponent $\hat\beta_{\rm env}$")
        ax_b.set_xlabel("Optimizer step")
        ax_a.grid(alpha=0.25)
        ax_b.grid(alpha=0.25)
        ax_a.legend(fontsize=7.5, framealpha=0.88)

    fig.suptitle("Forcing suppression and spectral response", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


def plot_time_scale_spectrum(agg_dir: Path, models: Sequence[str], outpath: Path, dpi: int) -> None:
    model = models[0] if models else "diag"
    conds = _strongest_conditions(_discover_conditions(agg_dir, model))
    spectra: List[Tuple[Condition, np.ndarray, np.ndarray]] = []
    for cond in conds:
        tau = _load_final_taus(agg_dir, model, cond)
        if tau is not None:
            spectra.append((cond, tau, -np.log(tau)))
    if not spectra:
        print("  [spectrum] no final tau data found; skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    z_all = np.concatenate([z for _, _, z in spectra])
    lo, hi = np.quantile(z_all, [0.005, 0.995])
    pad = 0.08 * max(hi - lo, 1e-6)
    bins = np.linspace(lo - pad, hi + pad, 38)

    for cond, tau, z in spectra:
        color = ABLATION_COLOR.get(cond.ablation, None)
        axes[0].hist(z, bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=cond.label)
        x, y = _ccdf(tau)
        mask = y > 0
        axes[1].plot(x[mask], y[mask], color=color, linewidth=1.8, label=cond.label)

    axes[0].set_xlabel(r"$\zeta=-\log\tau$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Final log-rate spectrum ({MODEL_LABEL.get(model, model)})")
    axes[0].legend(fontsize=8, framealpha=0.9)
    axes[0].grid(alpha=0.25)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Time scale $\tau$")
    axes[1].set_ylabel(r"Empirical $\mathbb{P}(T\geq\tau)$")
    axes[1].set_title(r"Final $\tau$-spectrum tail")
    axes[1].legend(fontsize=8, framealpha=0.9)
    axes[1].grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


def plot_phase_summary(agg_dir: Path, models: Sequence[str], outpath: Path, dpi: int) -> None:
    records = []
    for model in models:
        for cond in _discover_conditions(agg_dir, model):
            rows = _load_condition_rows(agg_dir, model, cond)
            phase = _phase_label(_load_final_phase(agg_dir, model, cond), rows)
            records.append((model, cond, phase))
    if not records:
        print("  [phase] no final phase records found; skipping")
        return

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(records)), 4.2))
    x = np.arange(len(records))
    colors = [PHASE_COLOR.get(phase, "#dddddd") for _, _, phase in records]
    ax.bar(x, np.ones_like(x), color=colors, edgecolor="white", linewidth=0.8)
    labels = [
        (f"{MODEL_LABEL.get(model, model)}\n{cond.label}" if len(models) > 1 else cond.label)
        for model, cond, _ in records
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title("Final phase verdict by ablation condition")
    for xi, (_, _, phase) in zip(x, records):
        ax.text(xi, 0.5, _phase_short(phase), rotation=90, ha="center", va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=_phase_short(label))
        for label, color in PHASE_COLOR.items()
        if label
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


def plot_threshold_brackets(agg_dir: Path, models: Sequence[str], outpath: Path, dpi: int) -> None:
    data = _read_json(agg_dir / "threshold_brackets.json")
    if not isinstance(data, dict):
        print("  [threshold] no threshold_brackets.json found; skipping")
        return

    rows = []
    for model in models:
        model_data = data.get(model, {})
        for ablation in ABLATION_ORDER:
            bracket = model_data.get(ablation, {})
            if bracket:
                rows.append((model, ablation, bracket))
    if not rows:
        print("  [threshold] no bracket records found; skipping")
        return

    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.48 * len(rows) + 1.0)))
    y = np.arange(len(rows))
    for yi, (model, ablation, bracket) in zip(y, rows):
        color = ABLATION_COLOR.get(ablation, "gray")
        label = f"{MODEL_LABEL.get(model, model)} / {ABLATION_LABEL.get(ablation, ablation)}"
        last_ac = bracket.get("last_anti_collapsed")
        first_c = bracket.get("first_collapsed")
        found = bool(bracket.get("bracket_found"))
        if found and last_ac is not None and first_c is not None:
            x0 = float(last_ac)
            x1 = float(first_c)
            ax.plot([x0, x1], [yi, yi], color=color, linewidth=4, solid_capstyle="round")
            ax.scatter([x0, x1], [yi, yi], color=[color, "black"], s=38, zorder=3)
            ax.text(max(x0, x1), yi + 0.12, "bracket", fontsize=8, color=color)
        else:
            xs = []
            for key in ("last_anti_collapsed", "first_collapsed"):
                try:
                    xs.append(float(bracket.get(key)))
                except Exception:
                    pass
            if xs:
                ax.scatter(xs, [yi] * len(xs), color=color, s=38)
            ax.text(0.02, yi + 0.12, str(bracket.get("status", "no full bracket")), fontsize=8, transform=ax.get_yaxis_transform())
        ax.text(-0.02, yi, label, ha="right", va="center", transform=ax.get_yaxis_transform(), fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Intervention value")
    ax.set_title("Threshold localization brackets in intervention space")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath} (dpi={dpi})")


def _parse_models(raw: str) -> List[str]:
    return [m.strip().lower() for m in raw.split(",") if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript-bound Exp3 ablation figures.")
    parser.add_argument("--agg_dir", type=Path, required=True, help="Path to exp3 aggregated directory.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for figures.")
    parser.add_argument("--models", type=str, default="diag", help="Comma-separated model short names.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--which",
        choices=["all", "trajectory", "spectrum", "phase", "threshold"],
        default="all",
    )
    args = parser.parse_args()

    if not args.agg_dir.exists():
        print(f"ERROR: --agg_dir does not exist: {args.agg_dir}", file=sys.stderr)
        raise SystemExit(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    models = _parse_models(args.models)
    if not models:
        print("ERROR: no models parsed from --models", file=sys.stderr)
        raise SystemExit(1)

    print(f"agg_dir : {args.agg_dir}")
    print(f"outdir  : {args.outdir}")
    print(f"models  : {models}")

    if args.which in ("all", "trajectory"):
        plot_forcing_beta_trajectory(args.agg_dir, models, args.outdir / "exp3_forcing_beta_trajectory.png", args.dpi)
    if args.which in ("all", "spectrum"):
        plot_time_scale_spectrum(args.agg_dir, models, args.outdir / "exp3_time_scale_spectrum.png", args.dpi)
    if args.which in ("all", "phase"):
        plot_phase_summary(args.agg_dir, models, args.outdir / "exp3_phase_summary.png", args.dpi)
    if args.which in ("all", "threshold"):
        plot_threshold_brackets(args.agg_dir, models, args.outdir / "exp3_threshold_brackets.png", args.dpi)


if __name__ == "__main__":
    main()
