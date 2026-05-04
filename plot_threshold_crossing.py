#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold-crossing and bracket visualization
=============================================

Reads threshold_crossing.json files from the phase-trajectory experiment
(main-text Experiment 2) and threshold_brackets.json from the forcing-ablation
experiment (main-text Experiment 3), then produces summary plots.

Can be run standalone:

    python plot_threshold_crossing.py \
        --phase_agg results/exp2_phase_full/adamw/aggregated \
        --ablation_agg results/exp3_forcing_full/adamw/aggregated \
        --outdir   results/plots/threshold \
        --dpi 300
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Helpers
# ============================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


MODEL_ORDER = ["const", "shared", "diag", "gru", "lstm"]
MODEL_LABELS = {
    "const": "ConstGate",
    "shared": "SharedGate",
    "diag": "DiagGate",
    "gru": "GRU",
    "lstm": "LSTM",
}
MODEL_COLORS = {
    "const": "#888888",
    "shared": "#1f77b4",
    "diag": "#ff7f0e",
    "gru": "#2ca02c",
    "lstm": "#d62728",
}


# ============================================================
# Main-text Exp 2: Threshold-crossing summary (bar chart of t_cross)
# ============================================================

def plot_threshold_crossing_exp1(agg_dir: str, outdir: str, dpi: int = 300):
    """
    Strip plot of observed and censored threshold times per architecture.
    """
    model_records = []
    max_step = 0.0

    for model_name in MODEL_ORDER:
        tc_path = os.path.join(agg_dir, model_name,
                               f"{model_name}_threshold_crossing.json")
        tc = _load_json(tc_path)
        if tc is None:
            continue

        per_seed = tc.get("per_seed", [])
        for ps in per_seed:
            if ps.get("crossed"):
                y = float(ps.get("t_cross_step", 0))
            elif ps.get("left_censored"):
                y = float(ps.get("first_observed_step", 0) or 0)
            else:
                y = float(ps.get("horizon_step", 0) or 0)
            max_step = max(max_step, y)

        model_records.append({
            "model_name": model_name,
            "n_seeds": int(tc.get("n_seeds", 0)),
            "n_observed": int(tc.get("n_observed_crossings", tc.get("n_crossed", 0))),
            "n_left_censored": int(tc.get("n_left_censored", 0)),
            "n_right_censored": int(tc.get("n_right_censored", 0)),
            "observed_mean": tc.get("observed_t_cross_step_mean", tc.get("t_cross_step_mean", float("nan"))),
            "observed_se": tc.get("observed_t_cross_step_se", tc.get("t_cross_step_se", 0.0)),
            "per_seed": per_seed,
        })

    if not model_records:
        log("  No threshold-crossing data found for phase-trajectory experiment, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(model_records))
    max_step = max(max_step, 1.0)
    text_pad = 0.04 * max_step

    for i, rec in enumerate(model_records):
        model_name = rec["model_name"]
        color = MODEL_COLORS.get(model_name, "#666666")
        per_seed = rec["per_seed"]
        if not per_seed:
            continue

        offsets = np.linspace(-0.16, 0.16, len(per_seed)) if len(per_seed) > 1 else np.array([0.0])
        y_top = 0.0
        for offset, ps in zip(offsets, per_seed):
            x_pos = x[i] + float(offset)
            if ps.get("crossed"):
                y_val = float(ps.get("t_cross_step", 0))
                ax.scatter(
                    x_pos, y_val, s=70, marker="o", zorder=4,
                    facecolor=color, edgecolor="black", linewidth=0.6,
                )
            elif ps.get("left_censored"):
                y_val = float(ps.get("first_observed_step", 0) or 0)
                ax.scatter(
                    x_pos, y_val, s=70, marker="s", zorder=4,
                    facecolor="white", edgecolor=color, linewidth=1.4,
                )
            else:
                y_val = float(ps.get("horizon_step", 0) or 0)
                ax.scatter(
                    x_pos, y_val, s=80, marker="^", zorder=4,
                    facecolor="white", edgecolor=color, linewidth=1.4,
                )
            y_top = max(y_top, y_val)

        observed_mean = rec["observed_mean"]
        observed_se = rec["observed_se"]
        if np.isfinite(observed_mean):
            ax.errorbar(
                x[i], observed_mean, yerr=observed_se,
                fmt="_", color="black", markersize=18,
                capsize=4, linewidth=1.2, zorder=5,
            )

        summary = (
            f"obs {rec['n_observed']}/{rec['n_seeds']}\n"
            f"L {rec['n_left_censored']}  R {rec['n_right_censored']}"
        )
        ax.text(
            x[i], y_top + text_pad, summary,
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS.get(rec["model_name"], rec["model_name"]) for rec in model_records],
        fontsize=9,
    )
    ax.set_ylabel("Threshold-crossing step  $t_{\\mathrm{cross}}$", fontsize=10)
    ax.set_title("Observed and censored threshold times (Exp 2)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max_step + 3.0 * text_pad)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="#666666", markeredgecolor="black",
               label="Observed crossing"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=8,
               markerfacecolor="white", markeredgecolor="#666666",
               label="Left-censored"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=8,
               markerfacecolor="white", markeredgecolor="#666666",
               label="Right-censored"),
        Line2D([0], [0], marker="_", linestyle="None", markersize=14,
               color="black", label="Observed mean ± SE"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "threshold_crossing_exp1.png"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "threshold_crossing_exp1.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved threshold_crossing_exp1.{{png,pdf}}")


# ============================================================
# Main-text Exp 2: Alpha and beta_env at crossing
# ============================================================

def plot_observables_at_crossing_exp1(agg_dir: str, outdir: str, dpi: int = 300):
    """
    Scatter of (alpha_hat, beta_env) at threshold-crossing per architecture.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name in MODEL_ORDER:
        tc_path = os.path.join(agg_dir, model_name,
                               f"{model_name}_threshold_crossing.json")
        tc = _load_json(tc_path)
        if tc is None:
            continue

        per_seed = tc.get("per_seed", [])
        alphas = []
        betas = []
        for ps in per_seed:
            if ps.get("crossed") and ps.get("alpha_at_cross") is not None and ps.get("beta_env_at_cross") is not None:
                alphas.append(ps["alpha_at_cross"])
                betas.append(ps["beta_env_at_cross"])

        if not alphas:
            continue

        color = MODEL_COLORS.get(model_name, "#666666")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.scatter(alphas, betas, color=color, label=label,
                   s=60, alpha=0.8, edgecolors="black", linewidths=0.5, zorder=3)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="$\\beta=1$")
    ax.axvline(2.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
               label="$\\alpha=2$")
    ax.set_xlabel(r"$\hat{\alpha}(t_{\mathrm{cross}})$", fontsize=10)
    ax.set_ylabel(r"$\hat{\beta}_{\mathrm{env}}(t_{\mathrm{cross}})$", fontsize=10)
    ax.set_title("Observables at threshold crossing (Exp 2)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "observables_at_crossing_exp1.png"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "observables_at_crossing_exp1.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved observables_at_crossing_exp1.{{png,pdf}}")


# ============================================================
# Main-text Exp 3: Threshold brackets
# ============================================================

def plot_threshold_brackets_exp2(agg_dir: str, outdir: str, dpi: int = 300):
    """
    Bracket plot showing last-AC and first-collapsed intervention values
    per architecture and ablation type.
    """
    brackets_path = os.path.join(agg_dir, "threshold_brackets.json")
    brackets = _load_json(brackets_path)
    if brackets is None:
        log("  No threshold_brackets.json found for forcing-ablation experiment, skipping plot.")
        return

    abl_types = set()
    for model_name in brackets:
        abl_types.update(brackets[model_name].keys())
    abl_types = sorted(abl_types)

    if not abl_types:
        return

    n_abl = len(abl_types)
    fig, axes = plt.subplots(1, n_abl, figsize=(5 * n_abl, 4.5), squeeze=False)

    abl_labels = {
        "batch_ablation": "Batch size",
        "clip_ablation": "Clipping norm",
        "winsorize_ablation": "Winsorization percentile",
    }

    for ai, abl_type in enumerate(abl_types):
        ax = axes[0, ai]
        models_with_data = []
        axis_values = []

        for model_name in MODEL_ORDER:
            if model_name not in brackets or abl_type not in brackets[model_name]:
                continue
            b = brackets[model_name][abl_type]
            ordered_conditions = b.get("ordered_conditions", [])
            axis_values.extend(
                float(r["condition_value"])
                for r in ordered_conditions
                if "condition_value" in r
            )
            if (
                b.get("last_anti_collapsed") is None
                and b.get("first_collapsed") is None
                and not b.get("status")
            ):
                continue
            models_with_data.append(model_name)

        if not models_with_data:
            ax.set_title(abl_labels.get(abl_type, abl_type))
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        y_pos = np.arange(len(models_with_data))
        if axis_values:
            x_min = min(axis_values)
            x_max = max(axis_values)
        else:
            x_min, x_max = 0.0, 1.0
        if "batch" in abl_type and x_min > 0 and x_max > 0:
            x_text = float(np.exp(0.5 * (np.log(x_min) + np.log(x_max))))
        else:
            x_text = 0.5 * (x_min + x_max)

        for yi, model_name in enumerate(models_with_data):
            b = brackets[model_name][abl_type]
            color = MODEL_COLORS.get(model_name, "#666666")

            ac_val = b.get("last_anti_collapsed")
            c_val = b.get("first_collapsed")
            status = str(b.get("status", ""))
            status_note = ""
            if b.get("used_majority_vote"):
                status_note = " (maj.)"

            if ac_val is not None:
                ac_val = float(ac_val)
                ax.plot(ac_val, yi, "o", color=color, markersize=10,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=3)
                ax.annotate("AC", (ac_val, yi), textcoords="offset points",
                            xytext=(-12, 8), fontsize=7, color=color)

            if c_val is not None:
                c_val = float(c_val)
                ax.plot(c_val, yi, "x", color=color, markersize=10,
                        markeredgewidth=2.0, zorder=3)
                ax.annotate("C", (c_val, yi), textcoords="offset points",
                            xytext=(6, 8), fontsize=7, color=color)

            if ac_val is not None and c_val is not None:
                lo = min(ac_val, c_val)
                hi = max(ac_val, c_val)
                ax.fill_betweenx([yi - 0.2, yi + 0.2], lo, hi,
                                 color=color, alpha=0.15)
                ax.plot([lo, hi], [yi, yi], "-", color=color,
                        linewidth=1.5, alpha=0.5)
            elif status:
                label_map = {
                    "all_anti_collapsed": "all AC",
                    "all_collapsed": "all C",
                    "non_monotone": "non-monotone",
                    "no_resolved_conditions": "unresolved",
                    "no_conditions": "no data",
                }
                ax.text(
                    x_text, yi, label_map.get(status, status.replace("_", " ")) + status_note,
                    ha="center", va="center", fontsize=7, style="italic",
                    color=color,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models_with_data],
                           fontsize=9)
        ax.set_xlabel(abl_labels.get(abl_type, abl_type), fontsize=10)
        ax.set_title(f"Threshold bracket: {abl_labels.get(abl_type, abl_type)}",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.25)

        # Use log scale for batch size
        if "batch" in abl_type:
            ax.set_xscale("log")

    fig.suptitle("Threshold localization in intervention space (Exp 3)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "threshold_brackets_exp2.png"),
                dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "threshold_brackets_exp2.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved threshold_brackets_exp2.{{png,pdf}}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Plot threshold-crossing (Exp 2) and bracket (Exp 3) diagnostics"
    )
    p.add_argument("--phase_agg", "--exp1_agg", dest="phase_agg", type=str, default=None,
                   help="Path to phase-trajectory aggregated directory (main-text Exp 2)")
    p.add_argument("--ablation_agg", "--exp2_agg", dest="ablation_agg", type=str, default=None,
                   help="Path to forcing-ablation aggregated directory (main-text Exp 3)")
    p.add_argument("--outdir", type=str, required=True,
                   help="Output directory for plots")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.phase_agg:
        log("Plotting Exp 2 threshold-crossing diagnostics ...")
        plot_threshold_crossing_exp1(args.phase_agg, args.outdir, args.dpi)
        plot_observables_at_crossing_exp1(args.phase_agg, args.outdir, args.dpi)

    if args.ablation_agg:
        log("Plotting Exp 3 threshold brackets ...")
        plot_threshold_brackets_exp2(args.ablation_agg, args.outdir, args.dpi)

    log("Done.")


if __name__ == "__main__":
    main()
