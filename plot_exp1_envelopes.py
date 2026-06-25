#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import json
import glob
import numpy as np

# Headless-safe matplotlib cache, matching the other manuscript plotters.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))
try:
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
except OSError:
    pass

import matplotlib.pyplot as plt

# Canonical model keys we want in the legend (and order)
CANON = ["const", "shared", "diag", "gru", "lstm"]

# -----------------------------------------
# JSON helper
# -----------------------------------------

def safe_load_json(path: str):
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

# -----------------------------------------
# CSV parsing helper
# -----------------------------------------

def _as_rows(data):
    # genfromtxt can return a scalar structured array for 1-row CSVs
    if data is None:
        return None
    if getattr(data, "size", 0) == 0:
        return None
    if getattr(data, "shape", None) == ():
        return np.array([data])
    return data

def _first_present(colset, candidates):
    for c in candidates:
        if c in colset:
            return c
    return None

def read_envelope_csv_any(path: str):
    """
    Robustly parse envelope CSVs.

    Supported "envelope" schemas:
      - ell, mu_mean, log_mu_mean       (baselines)
      - ell, f_mean,  log_f_mean        (gru/lstm in your pipeline)
      - ell, mu, log_mu                 (other variants)
      - ell, f,  log_f

    Supported "fit curve" schemas:
      - ell, log_mu_data, ...
      - ell, log_f_data, ...
      - ell, log_mu, ...
      - ell, log_f, ...

    Returns dict with keys: ell, mu, log_mu, src_kind
    or None if unrecognized/unreadable.
    """
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        return None

    data = _as_rows(data)
    if data is None:
        return None

    names = list(data.dtype.names or [])
    colset = set(names)

    # lag column (you use ell, but accept common aliases)
    c_ell = _first_present(colset, ["ell", "lag", "l"])
    if c_ell is None:
        return None
    ell = np.array(data[c_ell], dtype=int)

    # envelope columns (mu/f naming)
    c_mu  = _first_present(colset, ["mu_mean", "f_mean", "mu", "f"])
    c_lmu = _first_present(colset, ["log_mu_mean", "log_f_mean", "log_mu", "log_f"])

    if (c_mu is not None) and (c_lmu is not None):
        mu = np.array(data[c_mu], dtype=float)
        log_mu = np.array(data[c_lmu], dtype=float)
        return {"ell": ell, "mu": mu, "log_mu": log_mu, "src_kind": "envelope"}

    # fit curves: take log_*_data if present; else accept log_mu/log_f
    c_fit = _first_present(colset, ["log_mu_data", "log_f_data", "log_mu", "log_f"])
    if c_fit is not None:
        log_mu = np.array(data[c_fit], dtype=float)
        mu = np.exp(log_mu)
        return {"ell": ell, "mu": mu, "log_mu": log_mu, "src_kind": "fit_curves"}

    # last resort: if only mu exists, derive log
    if c_mu is not None:
        mu = np.array(data[c_mu], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mu = np.log(mu)
        return {"ell": ell, "mu": mu, "log_mu": log_mu, "src_kind": "mu_only"}

    return None

# -----------------------------------------
# Discovery: find envelopes, infer model key
# -----------------------------------------

ALIASES = {"constgate": "const", "sharedgate": "shared", "diaggate": "diag",
           "const_gate": "const", "shared_gate": "shared", "diag_gate": "diag"}

def infer_model_key(root: str, path_or_dir: str):
    """
    Infer canonical model key from any path by checking components.
    """
    rel = os.path.relpath(path_or_dir, root).replace("\\", "/").lower()
    parts = rel.split("/")
    for p in reversed(parts):
        if p in CANON:
            return p
        if p in ALIASES:
            return ALIASES[p]
    base = os.path.basename(path_or_dir).lower()
    if base in CANON:
        return base
    if base in ALIASES:
        return ALIASES[base]
    for key in CANON:
        if base.startswith(f"{key}_"):
            return key
    for alias, key in ALIASES.items():
        if base.startswith(f"{alias}_"):
            return key
    return None

def list_all_envelope_csvs(root: str):
    """
    Find all envelope CSVs under root.
    Skips seed_XXXX directories to avoid duplicate discovery.
    """
    root = os.path.abspath(root)
    _SEED_RE = re.compile(r"[/\\]seed_\d+[/\\]")
    patterns = [
        os.path.join(root, "**", "*_envelope.csv"),
        os.path.join(root, "**", "*_envelope_fit_curves.csv"),
    ]
    out = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            # skip files inside seed directories
            if _SEED_RE.search(p):
                continue
            out.append(p)
    # unique + stable
    out = sorted(set(out))
    return out

def choose_fit_json_near(env_csv_path: str):
    """
    Try to locate a nearby *_envelope_fit.json:
      1) same directory, <folder>_envelope_fit.json
      2) any *_envelope_fit.json in same directory
    """
    d = os.path.dirname(env_csv_path)
    base = os.path.basename(d)
    p0 = os.path.join(d, f"{base}_envelope_fit.json")
    if os.path.exists(p0):
        return p0
    anyfits = sorted(glob.glob(os.path.join(d, "*_envelope_fit.json")))
    return anyfits[0] if anyfits else None

def choose_audit_csv_near(env_csv_path: str):
    """Locate the audit CSV containing mu0+mu1 and corr_ratio, if present."""
    d = os.path.dirname(env_csv_path)
    audits = sorted(glob.glob(os.path.join(d, "*_envelope_audit.csv")))
    return audits[0] if audits else None

def read_envelope_audit_csv(path: str):
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        return None
    data = _as_rows(data)
    if data is None:
        return None
    names = set(data.dtype.names or [])
    required = {"ell", "log_f_mu0", "log_f_mu0_plus_mu1", "corr_ratio_mu1_over_mu0"}
    if not required.issubset(names):
        return None
    return {
        "ell": np.asarray(data["ell"], dtype=float),
        "log_mu0": np.asarray(data["log_f_mu0"], dtype=float),
        "log_mu0_plus_mu1": np.asarray(data["log_f_mu0_plus_mu1"], dtype=float),
        "corr_ratio": np.asarray(data["corr_ratio_mu1_over_mu0"], dtype=float),
    }

def mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return -1.0

def ordered_keys(keys):
    keys = list(keys)
    known = [k for k in CANON if k in keys]
    unknown = sorted([k for k in keys if k not in CANON])
    return known + unknown

# -----------------------------------------
# Plotting
# -----------------------------------------

def _mask_mu(ell, mu):
    return np.isfinite(ell) & (ell > 0) & np.isfinite(mu) & (mu > 0)

def _mask_log(ell, log_mu):
    return np.isfinite(ell) & (ell > 0) & np.isfinite(log_mu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Root folder to scan recursively (e.g., exp1/adamw)")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <indir>/plots_exp1)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--debug", type=int, default=1)
    ap.add_argument("--corr_ratio_threshold", type=float, default=0.25,
                    help="First-order validity guide: shade / fit only where |mu1|/|mu0| is below this value.")
    args = ap.parse_args()

    root = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir or os.path.join(root, "plots_exp1"))
    os.makedirs(outdir, exist_ok=True)

    csvs = list_all_envelope_csvs(root)
    if not csvs:
        raise RuntimeError(f"No *_envelope.csv or *_envelope_fit_curves.csv found under: {root}")

    # Collapse to one CSV per model key by selecting the newest CSV whose path implies that model
    picked = {}   # model_key -> {"csv":..., "mtime":...}
    ignored = []  # files that do not map to a canonical model key (still can be debugged)

    def _extract_epoch(path):
        """R12: extract checkpoint epoch from filename."""
        m = re.search(r'ckpt_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1

    for p in csvs:
        key = infer_model_key(root, p)
        if key is None:
            ignored.append(p)
            continue
        ep = _extract_epoch(p)
        mt = mtime(p)
        if key not in picked:
            picked[key] = {"csv": p, "mtime": mt, "epoch": ep}
        else:
            prev = picked[key]
            if ep > prev["epoch"] or (ep == prev["epoch"] and mt > prev["mtime"]):
                picked[key] = {"csv": p, "mtime": mt, "epoch": ep}

    keys = ordered_keys(picked.keys())

    # Read data
    env_data = {}
    env_src = {}
    env_fit = {}
    env_audit = {}

    dropped = []
    for k in keys:
        p = picked[k]["csv"]
        d = read_envelope_csv_any(p)
        if d is None:
            dropped.append((k, p))
            continue
        env_data[k] = d
        env_src[k] = p
        fitp = choose_fit_json_near(p)
        env_fit[k] = safe_load_json(fitp)
        auditp = choose_audit_csv_near(p)
        env_audit[k] = read_envelope_audit_csv(auditp) if auditp else None

    keys = [k for k in keys if k in env_data]

    if args.debug == 1:
        print(f"[DEBUG] root={root}")
        print(f"[DEBUG] found envelope CSVs total: {len(csvs)}")
        print(f"[DEBUG] picked (newest per model): {keys}")
        for k in keys:
            print(f"  - {k:>6s} [{env_data[k]['src_kind']}] <- {env_src[k]}")
        if dropped:
            print("[DEBUG] dropped (could not parse CSV):")
            for k, p in dropped:
                print(f"  - {k} <- {p}")
        if ignored:
            print(f"[DEBUG] ignored (no model key inferred): {len(ignored)}")
            # Print a few to help you see if naming is unexpected
            for p in ignored[:10]:
                print(f"  - {p}")

    if not keys:
        raise RuntimeError(
            "No model curves could be parsed. "
            "Run with --debug 1 and inspect 'dropped' / 'ignored' files."
        )

    # ---- Plot 1: mu vs ell
    plt.figure()
    any_data = False
    for k in keys:
        ell = env_data[k]["ell"]
        mu = env_data[k]["mu"]
        mask = _mask_mu(ell, mu)
        if np.any(mask):
            any_data = True
            plt.plot(ell[mask], mu[mask], marker="o", markersize=2.5, linewidth=1.5, label=k)
    if any_data:
        plt.xlabel(r"lag $\ell$")
        plt.ylabel(r"$\hat f(\ell)$")
        plt.title("Envelope scaling (final only)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "envelope_mu_vs_ell.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: log envelope vs ell
    plt.figure()
    any_data = False
    for k in keys:
        ell = env_data[k]["ell"]
        log_mu = env_data[k]["log_mu"]
        mask = _mask_log(ell, log_mu)
        if np.any(mask):
            any_data = True
            plt.plot(ell[mask], log_mu[mask], marker="o", markersize=2.5, linewidth=1.5, label=k)
    if any_data:
        plt.xlabel(r"lag $\ell$")
        plt.ylabel(r"$\log \hat f(\ell)$")
        plt.title(r"Envelope scaling (semi-log diagnostic, final only)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "log_envelope_vs_ell.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # ---- Plot 2b: first-order correction audit
    for k in keys:
        audit = env_audit.get(k)
        if audit is None:
            continue
        ell = audit["ell"]
        mask0 = _mask_log(ell, audit["log_mu0"])
        mask1 = _mask_log(ell, audit["log_mu0_plus_mu1"])
        cmask = np.isfinite(ell) & (ell > 0) & np.isfinite(audit["corr_ratio"])
        if not (np.any(mask0) and np.any(cmask)):
            continue
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})
        axes[0].plot(ell[mask0], audit["log_mu0"][mask0], color="#2b6ab8",
                     linewidth=1.5, label=r"canonical $\mu_0$")
        if np.any(mask1):
            axes[0].plot(ell[mask1], audit["log_mu0_plus_mu1"][mask1],
                         color="#dd8800", linestyle="--", linewidth=1.2,
                         label=r"audit $\mu_0+\mu_1$")
        axes[0].set_ylabel(r"$\log \hat f(\ell)$")
        axes[0].set_title(f"{k} envelope: first-order transport audit")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)

        ratio = audit["corr_ratio"]
        axes[1].plot(ell[cmask], ratio[cmask], color="#444444", linewidth=1.3)
        axes[1].axhline(float(args.corr_ratio_threshold), color="#aa3333",
                        linestyle="--", linewidth=1.0,
                        label=rf"threshold={float(args.corr_ratio_threshold):.2f}")
        axes[1].fill_between(
            ell[cmask],
            0.0,
            ratio[cmask],
            where=ratio[cmask] <= float(args.corr_ratio_threshold),
            color="#3aa050",
            alpha=0.12,
            label="controlled window",
        )
        axes[1].set_xlabel(r"lag $\ell$")
        axes[1].set_ylabel(r"$|\mu_1|/|\mu_0|$")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "envelope_first_order_audit.png"),
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    # ---- Plot 3: log envelope vs log ell
    plt.figure()
    any_data = False
    for k in keys:
        ell = env_data[k]["ell"].astype(float)
        log_mu = env_data[k]["log_mu"]
        mask = _mask_log(ell, log_mu)
        if np.any(mask):
            any_data = True
            plt.plot(np.log(ell[mask] + 1e-12), log_mu[mask],
                     marker="o", markersize=2.5, linewidth=1.5, label=k)
    if any_data:
        plt.xlabel(r"$\log \ell$")
        plt.ylabel(r"$\log \hat f(\ell)$")
        plt.title(r"Envelope scaling (log-log diagnostic, final only)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "log_envelope_vs_log_ell.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # ---- Plot 3b: finite-window power-law diagnostic on log-log axes
    # This is the paper-facing check for whether the envelope has a substantial
    # finite scaling window. A tempered curve is still shown as the finite-cutoff
    # continuation, but AIC/model-winner labels are deliberately omitted here:
    # the visual claim is the quality and length of the power-law window.
    for k in keys:
        fit = env_fit.get(k)
        if not isinstance(fit, dict):
            continue
        ell = env_data[k]["ell"].astype(float)
        log_mu = env_data[k]["log_mu"].astype(float)
        mask = _mask_log(ell, log_mu)
        if int(np.sum(mask)) < 6:
            continue
        ell_m = ell[mask]
        log_mu_m = log_mu[mask]
        log_ell_m = np.log(ell_m + 1e-12)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(log_ell_m, log_mu_m, marker="o", markersize=3.0,
                linewidth=1.4, label="data", color="#2b6ab8")

        exp_fit = fit.get("exp", {}) or {}
        if all(x in exp_fit for x in ("a", "b")):
            pred = float(exp_fit["a"]) + float(exp_fit["b"]) * ell_m
            ax.plot(log_ell_m, pred, "--", linewidth=1.1,
                    label=rf"exponential ($R^2={float(exp_fit.get('r2', np.nan)):.3f}$)",
                    color="#777777")

        pow_fit = fit.get("power", {}) or {}
        if all(x in pow_fit for x in ("c", "d")):
            pred = float(pow_fit["c"]) + float(pow_fit["d"]) * log_ell_m
            beta_pow = max(0.0, -float(pow_fit["d"]))
            ax.plot(log_ell_m, pred, "-", linewidth=0.9, alpha=0.50,
                    label=rf"full-window power ($\beta={beta_pow:.3f}$, $R^2={float(pow_fit.get('r2', np.nan)):.3f}$)",
                    color="#3aa050")

        temp_fit = fit.get("tempered", {}) or {}
        if all(x in temp_fit for x in ("a", "d_log", "b_ell")):
            pred = (
                float(temp_fit["a"])
                + float(temp_fit["d_log"]) * log_ell_m
                + float(temp_fit["b_ell"]) * ell_m
            )
            ax.plot(log_ell_m, pred, "-.", linewidth=1.1,
                    label=rf"tempered finite-cutoff ($R^2={float(temp_fit.get('r2', np.nan)):.3f}$)",
                    color="#dd8800")

        cd = fit.get("crossover_diagnostic", {}) or {}
        finite_window_text = "finite window: n/a"
        if cd.get("valid"):
            ell_star = cd.get("ell_star")
            if ell_star is not None and np.isfinite(float(ell_star)) and float(ell_star) > 0:
                ell_star_f = float(ell_star)
                x_star = np.log(ell_star_f)
                wmask = ell_m <= ell_star_f
                if int(np.count_nonzero(wmask)) >= 6:
                    xw = log_ell_m[wmask]
                    yw = log_mu_m[wmask]
                    slope_w, intercept_w = np.polyfit(xw, yw, deg=1)
                    pred_w = intercept_w + slope_w * xw
                    ss_res_w = float(np.sum((yw - pred_w) ** 2))
                    ss_tot_w = float(np.sum((yw - float(np.mean(yw))) ** 2))
                    r2_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float("nan")
                    beta_w = max(0.0, -float(slope_w))
                    ax.axvspan(float(np.min(xw)), float(np.max(xw)),
                               color="#aa3377", alpha=0.08, lw=0,
                               label="finite-window fit range")
                    ax.plot(xw, pred_w, color="#aa3377", linewidth=2.0,
                            label=rf"finite-window power ($\beta={beta_w:.3f}$, $R^2={r2_w:.3f}$)")
                    ax.axvline(x_star, color="#555555", linestyle=":", linewidth=1.0,
                               label=rf"$\ell^*={ell_star_f:.1f}$")
                    finite_window_text = (
                        f"finite-window beta={beta_w:.3g}\n"
                        f"finite-window R2={r2_w:.3g}\n"
                        f"finite-window n={int(np.count_nonzero(wmask))}\n"
                        f"ell*={ell_star_f:.3g}"
                    )
                else:
                    finite_window_text = (
                        f"finite-window n={int(np.count_nonzero(wmask))}\n"
                        f"ell*={ell_star_f:.3g}"
                    )

        audit = env_audit.get(k)
        corr_fit_text = "corr-gated fit: n/a"
        if audit is not None:
            corr = np.asarray(audit["corr_ratio"], dtype=float)
            ell_a = np.asarray(audit["ell"], dtype=float)
            log_a = np.asarray(audit["log_mu0"], dtype=float)
            amask = (
                _mask_log(ell_a, log_a)
                & np.isfinite(corr)
                & (corr <= float(args.corr_ratio_threshold))
            )
            if np.count_nonzero(amask) >= 6:
                xg = np.log(ell_a[amask] + 1e-12)
                yg = log_a[amask]
                slope, intercept = np.polyfit(xg, yg, deg=1)
                pred_g = intercept + slope * xg
                ss_res = float(np.sum((yg - pred_g) ** 2))
                ss_tot = float(np.sum((yg - float(np.mean(yg))) ** 2))
                r2_g = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                beta_g = max(0.0, -float(slope))
                ax.plot(xg, pred_g, color="#5b3f9f", linewidth=1.1, linestyle=":",
                        label=rf"transport-controlled power ($\beta={beta_g:.3f}$, $R^2={r2_g:.3f}$)")
                corr_fit_text = (
                    f"transport-controlled beta={beta_g:.3g}\n"
                    f"transport-controlled R2={r2_g:.3g}\n"
                    f"transport-controlled n={int(np.count_nonzero(amask))}"
                )
            elif corr.size:
                corr_fit_text = f"transport-controlled n={int(np.count_nonzero(amask))}"

        def _fmt_optional(text: str) -> str:
            # Compact the raw dict values for plot annotations.
            for token in ("None", "nan"):
                text = text.replace(token, "n/a")
            return text

        ann = (
            f"{_fmt_optional(finite_window_text)}\n"
            f"{_fmt_optional(corr_fit_text)}"
        )
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.2, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.88))

        ax.set_xlabel(r"$\log \ell$")
        ax.set_ylabel(r"$\log \hat f(\ell)$")
        ax.set_title(f"{k} envelope: finite-window power-law check")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"loglog_envelope_power_fit.png"),
            dpi=args.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---- Plot 4: Fit R^2 bar + AIC winner annotation (if JSONs exist)
    names, exp_r2, pow_r2, temp_r2 = [], [], [], []
    aic_winners = []
    ell_stars = []
    for k in keys:
        fit = env_fit.get(k)
        if not isinstance(fit, dict):
            continue
        e = fit.get("exp", {}) or {}
        p = fit.get("power", {}) or {}
        t = fit.get("tempered", {}) or {}
        if ("r2" in e) or ("r2" in p) or ("r2" in t):
            names.append(k)
            exp_r2.append(float(e.get("r2", np.nan)))
            pow_r2.append(float(p.get("r2", np.nan)))
            temp_r2.append(float(t.get("r2", np.nan)))
            # AIC winner
            # NOTE: aic dict uses "exponential" key, but fit sub-dicts use "exp"
            aic = fit.get("aic", {}) or {}
            winner = fit.get("envelope_winner", "")
            if not winner and aic:
                # Compute winner from AIC scores (keys: exponential, power, tempered)
                aic_vals = {m: aic.get(m, float("inf"))
                            for m in ["exponential", "power", "tempered"]}
                winner = min(aic_vals, key=aic_vals.get)
            aic_winners.append(winner)
            # ell_star from tempered fit
            ell_star = t.get("ell_star", np.nan)
            ell_stars.append(float(ell_star) if ell_star is not None else np.nan)

    if names:
        x = np.arange(len(names))
        width = 0.26
        plt.figure(figsize=(10.5, 4.8))
        bars_e = plt.bar(x - width, exp_r2, width=width, label="exp fit R$^2$")
        bars_p = plt.bar(x, pow_r2, width=width, label="power fit R$^2$")
        bars_t = plt.bar(x + width, temp_r2, width=width, label="tempered fit R$^2$")

        # Annotate AIC winner with a star marker
        for i, winner in enumerate(aic_winners):
            if winner == "mixed":
                continue
            if winner == "exponential":
                plt.plot(x[i] - width, min(exp_r2[i] + 0.03, 1.0), "*", color="gold", markersize=12)
            elif winner == "power":
                plt.plot(x[i], min(pow_r2[i] + 0.03, 1.0), "*", color="gold", markersize=12)
            elif winner == "tempered":
                plt.plot(x[i] + width, min(temp_r2[i] + 0.03, 1.0), "*", color="gold", markersize=12)

        # Annotate ell_star below x-axis labels
        for i, ell_s in enumerate(ell_stars):
            if np.isfinite(ell_s):
                plt.text(x[i], -0.08, rf"$\ell^*$={ell_s:.0f}", ha="center", fontsize=7,
                         color="purple", transform=plt.gca().get_xaxis_transform())

        plt.xticks(x, names, rotation=0, ha="center", fontsize=9)
        plt.ylim(0.0, 1.08)
        plt.ylabel(r"$R^2$")
        plt.title(r"Envelope fit quality comparison (final only; $\bigstar$ = AIC winner)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "envelope_fit_r2_bar.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    # ---- Plot 5: AIC comparison bar chart (if AIC data available)
    aic_names, aic_exp, aic_pow, aic_temp = [], [], [], []
    for k in keys:
        fit = env_fit.get(k)
        if not isinstance(fit, dict):
            continue
        aic = fit.get("aic", {}) or {}
        if aic:
            aic_names.append(k)
            aic_exp.append(float(aic.get("exponential", np.nan)))
            aic_pow.append(float(aic.get("power", np.nan)))
            aic_temp.append(float(aic.get("tempered", np.nan)))

    if aic_names:
        x = np.arange(len(aic_names))
        width = 0.26
        plt.figure(figsize=(10.5, 4.2))
        plt.bar(x - width, aic_exp, width=width, label="exp AIC")
        plt.bar(x, aic_pow, width=width, label="power AIC")
        plt.bar(x + width, aic_temp, width=width, label="tempered AIC")
        plt.xticks(x, aic_names, rotation=0, ha="center", fontsize=9)
        plt.ylabel("AIC")
        plt.title("Envelope AIC comparison (lower = better)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "envelope_aic_bar.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    # ---- Plot 6: Crossover residual diagnostic (Section 7) ----
    # One sub-panel per model with a valid crossover_diagnostic dict.
    cd_models = []
    for k in keys:
        fit = env_fit.get(k)
        if not isinstance(fit, dict):
            continue
        cd = fit.get("crossover_diagnostic", {})
        if cd.get("valid"):
            cd_models.append(k)

    if cd_models:
        ncols = min(len(cd_models), 3)
        nrows = (len(cd_models) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                                 squeeze=False)

        for idx, k in enumerate(cd_models):
            ax = axes[idx // ncols, idx % ncols]
            fit = env_fit[k]
            cd = fit["crossover_diagnostic"]
            d = env_data[k]
            ell = d["ell"]
            log_mu = np.log(np.maximum(d["mu"], 1e-30))

            if cd.get("mode") == "anti_collapsed":
                # Compute tempered-fit residuals for display
                tempered = fit.get("tempered", {})
                log_ell = np.log(ell + 1e-12)
                pred_temp = (tempered.get("a", 0) + tempered.get("d_log", 0) * log_ell
                             + tempered.get("b_ell", 0) * ell)
                resid = log_mu - pred_temp
                ell_star = cd["ell_star"]

                below_mask = ell <= ell_star
                above_mask = ell > ell_star

                ax.scatter(ell[below_mask], resid[below_mask], s=12, alpha=0.6,
                           color="steelblue", label=r"$\ell \leq \ell^*$", zorder=3)
                ax.scatter(ell[above_mask], resid[above_mask], s=12, alpha=0.6,
                           color="tomato", label=r"$\ell > \ell^*$", zorder=3)
                ax.axvline(ell_star, color="grey", ls="--", lw=1.0, alpha=0.7,
                           label=rf"$\ell^* = {ell_star:.1f}$")
                ax.axhline(0, color="black", lw=0.5, alpha=0.4)

                runs_p = cd.get("runs_below_p", float("nan"))
                sign_p = cd.get("sign_above_p", float("nan"))
                runs_ok = cd.get("runs_below_pass", False)
                sign_ok = cd.get("sign_above_pass", False)

                ann = (f"Below: runs p={runs_p:.3f} {'OK' if runs_ok else 'FAIL'}\n"
                       f"Above: sign p={sign_p:.3f} {'OK' if sign_ok else 'FAIL'}")
                ax.text(0.97, 0.97, ann, transform=ax.transAxes, va="top", ha="right",
                        fontsize=7, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

                ax.set_title(f"{k} — crossover residuals", fontsize=10)
                ax.set_ylabel("Tempered-fit residual")

            else:  # collapsed mode
                efit = fit.get("exp", {})
                pred_exp = efit.get("a", 0) + efit.get("b", 0) * ell
                resid = log_mu - pred_exp

                ax.scatter(ell, resid, s=12, alpha=0.6, color="grey", zorder=3)
                ax.axhline(0, color="black", lw=0.5, alpha=0.4)

                runs_p = cd.get("runs_exp_p", float("nan"))
                runs_ok = cd.get("runs_exp_pass", False)
                r2 = cd.get("r2_exp", float("nan"))
                ann = f"Exp R²={r2:.3f}\nRuns p={runs_p:.3f} {'OK' if runs_ok else 'FAIL'}"
                ax.text(0.97, 0.97, ann, transform=ax.transAxes, va="top", ha="right",
                        fontsize=7, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

                ax.set_title(f"{k} — exp residuals (collapsed)", fontsize=10)
                ax.set_ylabel("Exp-fit residual")

            ax.set_xlabel("Lag ℓ")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(True, alpha=0.2)

        # Blank out unused axes
        for idx in range(len(cd_models), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "envelope_crossover_diagnostic.png"),
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Saved envelope plots to: {outdir}")

if __name__ == "__main__":
    main()
