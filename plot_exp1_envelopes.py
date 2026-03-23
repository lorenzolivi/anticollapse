#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import json
import glob
import numpy as np
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

    # ---- Plot 4: Fit R^2 bar (if JSONs exist)
    names, exp_r2, pow_r2, temp_r2 = [], [], [], []
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

    if names:
        x = np.arange(len(names))
        width = 0.26
        plt.figure(figsize=(10.5, 4.2))
        plt.bar(x - width, exp_r2, width=width, label="exp fit R$^2$")
        plt.bar(x, pow_r2, width=width, label="power fit R$^2$")
        plt.bar(x + width, temp_r2, width=width, label="tempered fit R$^2$")
        plt.xticks(x, names, rotation=0, ha="center", fontsize=9)
        plt.ylim(0.0, 1.01)
        plt.ylabel(r"$R^2$")
        plt.title("Envelope fit quality comparison (final only)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "envelope_fit_r2_bar.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    print(f"[OK] Saved envelope plots to: {outdir}")

if __name__ == "__main__":
    main()
