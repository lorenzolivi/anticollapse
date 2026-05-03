#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def safe_read_csv_named(path: str):
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if data.size == 0:
            return None
        return data
    except Exception:
        return None

def write_csv(path, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

# ------------------------------------------------------------
# Recursive discovery
# ------------------------------------------------------------

def find_model_dirs(indir: str):
    """
    Find directories that look like a model output folder.

    Accept a directory if it contains at least one of:
      - phase_trajectory.csv (new dynamic plan)
      - <name>_phase_summary.json (old)
      - <name>_alpha_grad.json (old)
      - <name>_tau_tail_fit.json (old)
      - <name>_envelope_fit.json (old)

    Skips seed_XXXX directories to avoid duplicate discovery when called
    from the optimizer-level directory instead of aggregated/.

    Returns list of dicts:
      {"dir": <abs dir>, "name": <folder name>, "label": <rel path label>}
    """
    out = []
    indir = os.path.abspath(indir)
    import re
    _SEED_RE = re.compile(r"^seed_\d+$")

    for root, dirs, files in os.walk(indir):
        folder = os.path.basename(root)

        # Skip seed directories to avoid duplicate model discovery
        dirs[:] = [d for d in dirs if not _SEED_RE.match(d)]

        has_dynamic = "phase_trajectory.csv" in files
        has_old = (
            f"{folder}_phase_summary.json" in files or
            f"{folder}_alpha_grad.json" in files or
            f"{folder}_tau_tail_fit.json" in files or
            f"{folder}_envelope_fit.json" in files
        )

        if has_dynamic or has_old:
            label = folder  # canonical model name (e.g. "diag", "gru")
            out.append({"dir": root, "name": folder, "label": label})

    # de-dup by dir
    uniq = {d["dir"]: d for d in out}
    out = list(uniq.values())
    out.sort(key=lambda d: d["label"])
    return out

# ------------------------------------------------------------
# Dynamic trajectory loaders
# ------------------------------------------------------------

def load_phase_trajectory(traj_csv: str):
    """
    phase_trajectory.csv columns (new plan):
      epoch, alpha_hat, sigma_alpha_hat, beta_hat, beta_r2,
      beta_median, beta_lo, beta_hi, p_beta_lt1, phase_label, ...
    Returns dict with arrays sorted by epoch, or None.
    """
    data = safe_read_csv_named(traj_csv)
    if data is None:
        return None

    # genfromtxt returns scalar structured array if 1 row
    rows = np.array([data]) if getattr(data, "shape", ()) == () else data
    names = rows.dtype.names or ()
    if ("epoch" not in names) or ("alpha_hat" not in names) or ("beta_hat" not in names):
        return None

    ep = np.array(rows["epoch"], dtype=float)
    a  = np.array(rows["alpha_hat"], dtype=float)
    b  = np.array(rows["beta_hat"], dtype=float)

    r2 = np.array(rows["beta_r2"], dtype=float) if ("beta_r2" in names) else np.full_like(b, np.nan)
    sig_a = np.array(rows["sigma_alpha_hat"], dtype=float) if ("sigma_alpha_hat" in names) else np.full_like(a, np.nan)
    std_a = np.array(rows["alpha_hat_std"], dtype=float) if ("alpha_hat_std" in names) else np.full_like(a, np.nan)
    se_a = np.array(rows["alpha_hat_se"], dtype=float) if ("alpha_hat_se" in names) else np.full_like(a, np.nan)

    # Bootstrap columns (new)
    beta_median = np.array(rows["beta_median"], dtype=float) if ("beta_median" in names) else np.full_like(b, np.nan)
    beta_lo = np.array(rows["beta_lo"], dtype=float) if ("beta_lo" in names) else np.full_like(b, np.nan)
    beta_hi = np.array(rows["beta_hi"], dtype=float) if ("beta_hi" in names) else np.full_like(b, np.nan)
    p_beta_lt1 = np.array(rows["p_beta_lt1"], dtype=float) if ("p_beta_lt1" in names) else np.full_like(b, np.nan)

    # Phase label (string column)
    if "phase_label" in names:
        phase_label = np.array(rows["phase_label"], dtype=str)
    else:
        phase_label = np.full(ep.shape, "", dtype=object)

    m = np.isfinite(ep) & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return None
    ep, a, b, r2, sig_a, std_a, se_a = ep[m], a[m], b[m], r2[m], sig_a[m], std_a[m], se_a[m]
    beta_median, beta_lo, beta_hi, p_beta_lt1 = beta_median[m], beta_lo[m], beta_hi[m], p_beta_lt1[m]
    phase_label = phase_label[m]

    order = np.argsort(ep)
    ep, a, b, r2, sig_a, std_a, se_a = ep[order], a[order], b[order], r2[order], sig_a[order], std_a[order], se_a[order]
    beta_median, beta_lo, beta_hi, p_beta_lt1 = beta_median[order], beta_lo[order], beta_hi[order], p_beta_lt1[order]
    phase_label = phase_label[order]

    return {
        "epoch": ep.astype(int),
        "alpha_hat": a,
        "beta_hat": b,
        "beta_r2": r2,
        "sigma_alpha_hat": sig_a,
        "alpha_hat_std": std_a,
        "alpha_hat_se": se_a,
        "beta_median": beta_median,
        "beta_lo": beta_lo,
        "beta_hi": beta_hi,
        "p_beta_lt1": p_beta_lt1,
        "phase_label": phase_label,
    }

def load_final_from_phase_trajectory(traj_csv: str):
    tr = load_phase_trajectory(traj_csv)
    if tr is None:
        return None
    i = int(np.argmax(tr["epoch"]))
    result = {
        "epoch": int(tr["epoch"][i]),
        "alpha_hat": float(tr["alpha_hat"][i]),
        "sigma_alpha_hat": float(tr["sigma_alpha_hat"][i]) if np.isfinite(tr["sigma_alpha_hat"][i]) else np.nan,
        "beta_hat": float(tr["beta_hat"][i]),
        "beta_r2": float(tr["beta_r2"][i]) if np.isfinite(tr["beta_r2"][i]) else np.nan,
        "beta_median": float(tr["beta_median"][i]) if np.isfinite(tr["beta_median"][i]) else np.nan,
        "beta_lo": float(tr["beta_lo"][i]) if np.isfinite(tr["beta_lo"][i]) else np.nan,
        "beta_hi": float(tr["beta_hi"][i]) if np.isfinite(tr["beta_hi"][i]) else np.nan,
        "p_beta_lt1": float(tr["p_beta_lt1"][i]) if np.isfinite(tr["p_beta_lt1"][i]) else np.nan,
        "phase_label": str(tr["phase_label"][i]) if tr["phase_label"][i] else "",
    }
    return result


def beta_center_array(tr: dict):
    """Use bootstrap beta median when available, otherwise fall back to beta_hat."""
    b_med = np.asarray(tr.get("beta_median", np.array([])), dtype=float)
    b_hat = np.asarray(tr.get("beta_hat", np.array([])), dtype=float)
    if b_med.shape == b_hat.shape and np.any(np.isfinite(b_med)):
        return np.where(np.isfinite(b_med), b_med, b_hat)
    return b_hat


def beta_center_scalar(record: dict):
    b_med = record.get("beta_median", np.nan)
    if b_med is not None and np.isfinite(b_med):
        return float(b_med)
    b_hat = record.get("beta_hat", np.nan)
    return float(b_hat) if b_hat is not None else np.nan


def format_phase_label(phase_data: dict):
    """Render mixed cross-seed labels without hiding the majority class."""
    if not isinstance(phase_data, dict):
        return ""
    label = str(phase_data.get("phase_label", "")).strip()
    majority = str(phase_data.get("majority_phase_label", "")).strip()
    if label == "mixed" and majority:
        return f"mixed / maj: {majority}"
    return label


def abbreviate_phase_label(label: str):
    return (
        str(label)
        .replace("anti-collapse", "AC")
        .replace("(soft classification)", "(soft)")
    )


def load_final_phase_json(model_dir: str, model_name: str):
    for candidate in [
        os.path.join(model_dir, f"{model_name}_final_phase.json"),
        os.path.join(model_dir, "final_phase.json"),
    ]:
        data = safe_load_json(candidate)
        if isinstance(data, dict):
            return data
    return None

# ------------------------------------------------------------
# Extract "final phase point" for a model folder (dynamic first)
# ------------------------------------------------------------

def merge_phase_point(model_dir: str, model_name: str):
    """
    Priority:
      1) phase_trajectory.csv (dynamic)
      2) <model>_phase_summary.json (old combined)
      3) <model>_alpha_grad.json + <model>_tau_tail_fit.json (old split)
    Envelope fit R2:
      - <model>_envelope_fit.json if present
    """
    # (1) dynamic
    traj_p = os.path.join(model_dir, "phase_trajectory.csv")
    dyn = load_final_from_phase_trajectory(traj_p) if os.path.exists(traj_p) else None

    # (2) old combined
    p_phase = os.path.join(model_dir, f"{model_name}_phase_summary.json")
    phase = safe_load_json(p_phase) or {}

    # (3) old split
    p_alpha = os.path.join(model_dir, f"{model_name}_alpha_grad.json")
    p_tail  = os.path.join(model_dir, f"{model_name}_tau_tail_fit.json")
    alpha = safe_load_json(p_alpha) or {}
    tail  = safe_load_json(p_tail)  or {}

    # envelope fit (final-only)
    p_env = os.path.join(model_dir, f"{model_name}_envelope_fit.json")
    env = safe_load_json(p_env) or {}
    final_phase = load_final_phase_json(model_dir, model_name) or {}

    # choose alpha/beta
    if dyn is not None:
        alpha_hat = dyn.get("alpha_hat", np.nan)
        beta_hat  = dyn.get("beta_hat", np.nan)
        beta_r2   = dyn.get("beta_r2", np.nan)
        beta_median = dyn.get("beta_median", np.nan)
        beta_lo = dyn.get("beta_lo", np.nan)
        beta_hi = dyn.get("beta_hi", np.nan)
        p_beta_lt1 = dyn.get("p_beta_lt1", np.nan)
        phase_label = dyn.get("phase_label", "")
        epoch_fin = dyn.get("epoch", None)
    else:
        alpha_hat = phase.get("alpha_hat", alpha.get("alpha_hat", np.nan))
        beta_hat  = phase.get("beta_hat", tail.get("beta_hat", np.nan))
        beta_r2   = phase.get("beta_r2", tail.get("beta_r2", np.nan))
        beta_median = np.nan
        beta_lo = np.nan
        beta_hi = np.nan
        p_beta_lt1 = np.nan
        phase_label = ""
        epoch_fin = phase.get("epoch", None)

    phase_label = format_phase_label(final_phase) or str(phase_label).strip()

    # envelope fit R2 (may be absent; keep NaN)
    env_r2_exp = phase.get("env_r2_exp", (env.get("exp", {}) or {}).get("r2", np.nan))
    env_r2_pow = phase.get("env_r2_power", (env.get("power", {}) or {}).get("r2", np.nan))
    env_r2_temp = phase.get("env_r2_tempered", (env.get("tempered", {}) or {}).get("r2", np.nan))

    return {
        "epoch": epoch_fin,
        "alpha_hat": float(alpha_hat) if alpha_hat is not None else np.nan,
        "beta_hat": float(beta_hat) if beta_hat is not None else np.nan,
        "beta_r2": float(beta_r2) if beta_r2 is not None else np.nan,
        "beta_median": float(beta_median) if beta_median is not None else np.nan,
        "beta_lo": float(beta_lo) if beta_lo is not None else np.nan,
        "beta_hi": float(beta_hi) if beta_hi is not None else np.nan,
        "p_beta_lt1": float(p_beta_lt1) if p_beta_lt1 is not None else np.nan,
        "phase_label": phase_label,
        "env_r2_exp": float(env_r2_exp) if env_r2_exp is not None else np.nan,
        "env_r2_pow": float(env_r2_pow) if env_r2_pow is not None else np.nan,
        "env_r2_temp": float(env_r2_temp) if env_r2_temp is not None else np.nan,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Experiment root folder (can be nested)")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <indir>/plots_exp1)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--min_r2", type=float, default=-np.inf,
                    help="Optional: only plot trajectory points with beta_r2 >= min_r2 (default: no filter).")
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_exp1")
    os.makedirs(outdir, exist_ok=True)

    entries = find_model_dirs(indir)
    if not entries:
        raise RuntimeError("No model folders found (no phase_trajectory.csv or old summary artifacts).")

    # ---- collect dynamics (if available) and final summaries
    merged = []
    traj_by_label = {}  # label -> trajectory dict
    model_dir_by_label = {}
    for e in entries:
        model_dir = e["dir"]
        model_name = e["name"]
        label = e["label"]
        model_dir_by_label[label] = model_dir

        # final point (always try)
        info = merge_phase_point(model_dir, model_name)

        merged.append({
            "label": label,
            "model": model_name,
            "epoch": info["epoch"],
            "alpha_hat": info["alpha_hat"],
            "beta_hat": info["beta_hat"],
            "beta_r2": info["beta_r2"],
            "beta_median": info.get("beta_median", np.nan),
            "beta_lo": info.get("beta_lo", np.nan),
            "beta_hi": info.get("beta_hi", np.nan),
            "p_beta_lt1": info.get("p_beta_lt1", np.nan),
            "phase_label": info.get("phase_label", ""),
            "env_r2_exp": info["env_r2_exp"],
            "env_r2_pow": info["env_r2_pow"],
            "env_r2_temp": info["env_r2_temp"],
        })

        # trajectory (new plan)
        traj_p = os.path.join(model_dir, "phase_trajectory.csv")
        if os.path.exists(traj_p):
            tr = load_phase_trajectory(traj_p)
            if tr is not None and tr["epoch"].size > 0:
                traj_by_label[label] = tr

    # ---- save merged CSV (final checkpoint summary)
    write_csv(
        os.path.join(outdir, "phase_diagram_merged_summary.csv"),
        ["label", "model", "epoch", "alpha_hat", "beta_hat", "beta_r2",
         "beta_median", "beta_lo", "beta_hi", "p_beta_lt1", "phase_label",
         "env_r2_exp", "env_r2_pow", "env_r2_temp"],
        [[r["label"], r["model"], r["epoch"], r["alpha_hat"], r["beta_hat"], r["beta_r2"],
          r["beta_median"], r["beta_lo"], r["beta_hi"], r["p_beta_lt1"], r["phase_label"],
          r["env_r2_exp"], r["env_r2_pow"], r["env_r2_temp"]]
         for r in merged]
    )

    # ============================================================
    # DYNAMIC PLOTS (new plan)
    # ============================================================

    # 1) alpha(t) vs epoch overlay
    if traj_by_label:
        plt.figure(figsize=(7.6, 4.8))
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            plt.plot(tr["epoch"], tr["alpha_hat"], linewidth=2.0, label=label)
            # R4: +/- 1 SE band
            if "alpha_hat_se" in tr and np.any(np.isfinite(tr["alpha_hat_se"])):
                se = tr["alpha_hat_se"]
                plt.fill_between(tr["epoch"], tr["alpha_hat"] - se,
                                 tr["alpha_hat"] + se, alpha=0.15)
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\alpha}(t)$")
        plt.title(r"Phase dynamics: gradient tail index")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "phase_dynamics_alpha_vs_epoch.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # 2) beta(t) vs epoch overlay — with bootstrap stability interval
        plt.figure(figsize=(7.6, 4.8))
        any_beta = False
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            b = beta_center_array(tr)
            ep = tr["epoch"]

            if np.isfinite(args.min_r2):
                r2 = tr["beta_r2"]
                mask = np.isfinite(b) & np.isfinite(ep) & np.isfinite(r2) & (r2 >= args.min_r2)
            else:
                mask = np.isfinite(b) & np.isfinite(ep)

            if np.any(mask):
                any_beta = True
                line, = plt.plot(ep[mask], b[mask], linewidth=2.0, label=label)
                # Bootstrap 90% stability interval (beta_lo, beta_hi)
                blo = tr["beta_lo"]
                bhi = tr["beta_hi"]
                boot_mask = mask & np.isfinite(blo) & np.isfinite(bhi)
                if np.any(boot_mask):
                    plt.fill_between(ep[boot_mask], blo[boot_mask], bhi[boot_mask],
                                     alpha=0.15, color=line.get_color())

        if any_beta:
            plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label=r"$\beta=1$")
            plt.xlabel("epoch")
            plt.ylabel(r"$\hat{\beta}_{\mathrm{med}}(t)$")
            plt.title(r"Phase dynamics: bootstrap spectral exponent (90% stability band)")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "phase_dynamics_beta_vs_epoch.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # 3) phase trajectory in (alpha, beta) — annotate with final phase label
        plt.figure(figsize=(6.8, 5.2))
        any_traj = False

        # Phase label -> marker style mapping
        _phase_markers = {
            "collapsed": "X",
            "concentrated anti-collapse": "^",
            "broad anti-collapse": "v",
            "boundary (soft classification)": "D",
        }

        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            a = tr["alpha_hat"]
            b = beta_center_array(tr)
            ep = tr["epoch"]

            if np.isfinite(args.min_r2):
                r2 = tr["beta_r2"]
                mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(r2) & (r2 >= args.min_r2)
            else:
                mask = np.isfinite(a) & np.isfinite(b)

            if not np.any(mask):
                continue

            any_traj = True
            # line + markers to show time direction (epoch increasing)
            plt.plot(a[mask], b[mask], marker="o", markersize=3.0, linewidth=1.6, label=label)

            # annotate final point with phase label (from trajectory or final_phase.json)
            a_last = a[mask][-1]
            b_last = b[mask][-1]

            # Try to load final phase from JSON (definitive classification)
            final_phase_label = ""
            fp_data = load_final_phase_json(model_dir_by_label.get(label, os.path.join(indir, label)), label)
            if fp_data:
                final_phase_label = format_phase_label(fp_data)

            # Fall back to last checkpoint phase_label from trajectory
            if not final_phase_label:
                pl = tr["phase_label"]
                if pl is not None and len(pl) > 0:
                    last_pl = str(pl[-1]).strip()
                    if last_pl:
                        final_phase_label = last_pl

            # Annotate final point
            ann_text = f" {label}"
            if final_phase_label:
                short_label = abbreviate_phase_label(final_phase_label)
                ann_text += f"\n [{short_label}]"
            plt.text(a_last, b_last, ann_text, fontsize=7, va="center")

            # R5: direction arrows at ~1/3 and ~2/3 of trajectory
            a_m, b_m = a[mask], b[mask]
            n_pts = a_m.size
            for frac in [0.33, 0.66]:
                idx_arr = min(int(frac * n_pts), n_pts - 2)
                if idx_arr >= 0 and idx_arr + 1 < n_pts:
                    plt.annotate("", xy=(a_m[idx_arr+1], b_m[idx_arr+1]),
                                 xytext=(a_m[idx_arr], b_m[idx_arr]),
                                 arrowprops=dict(arrowstyle="->",
                                                 color=plt.gca().lines[-1].get_color(),
                                                 lw=1.5))

        if any_traj:
            plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.4, label=r"$\beta=1$")
            plt.axvline(x=2.0, color="gray", linestyle=":", linewidth=1, alpha=0.4, label=r"$\alpha=2$")
            plt.xlabel(r"gradient tail index $\hat{\alpha}(t)$")
            plt.ylabel(r"bootstrap spectral exponent $\hat{\beta}_{\mathrm{med}}(t)$")
            ttl = r"Dynamical phase trajectories in $(\hat{\alpha}, \hat{\beta}_{\mathrm{med}})$"
            if np.isfinite(args.min_r2) and args.min_r2 > -np.inf:
                ttl += rf" (filtered: $R^2\geq {args.min_r2:g}$)"
            plt.title(ttl)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "phase_trajectory_alpha_vs_beta.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # 4) save trajectory long-form CSV (useful for external plotting)
        rows = []
        for label in sorted(traj_by_label.keys()):
            tr = traj_by_label[label]
            for i in range(tr["epoch"].size):
                rows.append([
                    label,
                    int(tr["epoch"][i]),
                    float(tr["alpha_hat"][i]),
                    float(tr["beta_hat"][i]),
                    float(tr["beta_r2"][i]) if np.isfinite(tr["beta_r2"][i]) else np.nan,
                    float(tr["beta_median"][i]) if np.isfinite(tr["beta_median"][i]) else np.nan,
                    float(tr["beta_lo"][i]) if np.isfinite(tr["beta_lo"][i]) else np.nan,
                    float(tr["beta_hi"][i]) if np.isfinite(tr["beta_hi"][i]) else np.nan,
                    float(tr["p_beta_lt1"][i]) if np.isfinite(tr["p_beta_lt1"][i]) else np.nan,
                    str(tr["phase_label"][i]) if tr["phase_label"][i] else "",
                ])
        write_csv(
            os.path.join(outdir, "phase_trajectories_long.csv"),
            ["label", "epoch", "alpha_hat", "beta_hat", "beta_r2",
             "beta_median", "beta_lo", "beta_hi", "p_beta_lt1", "phase_label"],
            rows
        )

    # ============================================================
    # FINAL SNAPSHOT PLOTS (still useful)
    # ============================================================

    # Scatter alpha vs beta (final)
    plt.figure(figsize=(6.8, 4.8))
    for r in merged:
        a = r["alpha_hat"]
        b = beta_center_scalar(r)
        if np.isfinite(a) and np.isfinite(b):
            plt.scatter([a], [b])
            plt.text(a, b, f" {r['label']}", va="center", fontsize=8)
    plt.xlabel(r"gradient tail index $\hat{\alpha}$")
    plt.ylabel(r"bootstrap spectral exponent $\hat{\beta}_{\mathrm{med}}$")
    plt.title(r"Empirical phase summary (final checkpoint)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_diagram_alpha_vs_beta_final.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # beta_r2 bar (final) — NaN-safe: replace NaN with 0 for display, mark with hatching
    labels = [r["label"] for r in merged]
    vals_raw = [r["beta_r2"] for r in merged]
    vals = [v if np.isfinite(v) else 0.0 for v in vals_raw]
    nan_mask = [not np.isfinite(v) for v in vals_raw]
    if any(np.isfinite(v) for v in vals_raw):
        x = np.arange(len(labels))
        plt.figure(figsize=(9.6, 4.2))
        colors = ["salmon" if nm else "steelblue" for nm in nan_mask]
        plt.bar(x, vals, color=colors)
        for i, nm in enumerate(nan_mask):
            if nm:
                plt.text(i, 0.02, "NaN", ha="center", fontsize=7, color="red")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylim(0.0, 1.01)
        plt.ylabel(r"$R^2$")
        plt.title(r"CCDF tail-fit quality (final checkpoint)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "phase_diagram_fit_quality_final.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    # envelope fit quality bars (exp vs power vs tempered) (final)
    exp_r2 = [r["env_r2_exp"] for r in merged]
    pow_r2 = [r["env_r2_pow"] for r in merged]
    temp_r2 = [r["env_r2_temp"] for r in merged]
    if any(np.isfinite(v) for v in exp_r2 + pow_r2 + temp_r2):
        x = np.arange(len(labels))
        width = 0.26
        plt.figure(figsize=(9.6, 4.2))
        plt.bar(x - width, exp_r2, width=width, label="exp fit R$^2$")
        plt.bar(x, pow_r2, width=width, label="power fit R$^2$")
        plt.bar(x + width, temp_r2, width=width, label="tempered fit R$^2$")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylim(0.0, 1.01)
        plt.ylabel(r"$R^2$")
        plt.title("Envelope fit quality (final checkpoint, if saved)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "phase_diagram_envelope_fit_quality_final.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    print(f"[OK] Saved phase-dynamics + phase-summary plots to: {outdir}")

if __name__ == "__main__":
    main()
