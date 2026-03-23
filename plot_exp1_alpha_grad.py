#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alpha-gradient plotting — improved version
============================================
Key changes vs. previous version:
  - Replaces raw "grad_projection" histogram/KDE with proper diagnostic:
      * Histogram of gradient projections per model
      * Overlaid fitted symmetric alpha-stable PDF (using estimated alpha_hat, sigma_hat)
      * Overlaid Gaussian PDF (same variance) for comparison
  - Still plots alpha_hat(t) dynamics (kept from before)
  - Adds per-model diagnostic panels at final checkpoint
  - Uses scipy.stats.levy_stable if available; graceful fallback otherwise
"""

import os, re, argparse, json, csv, glob
import numpy as np
import matplotlib.pyplot as plt

CKPT_RE = re.compile(r"ckpt_(\d+)_")

# ============================================================
# Helpers
# ============================================================

def safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def read_alpha_samples_csv(path: str):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if getattr(data, "size", 0) == 0:
        return None
    if "grad_projection" not in data.dtype.names:
        return None
    s = np.array(data["grad_projection"], dtype=float)
    s = s[np.isfinite(s)]
    return s

def write_csv(path, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def find_model_dirs_recursive(indir: str):
    """
    A "model directory" is any directory that contains:
      - phase_trajectory.csv (dynamic) OR
      - checkpoint_alpha/ckpt_*_alpha_grad_samples.csv (dynamic) OR
      - <model>_alpha_grad_samples.csv (legacy)
    The model name is the directory basename.

    Skips seed_XXXX directories to avoid duplicate discovery when called
    from the optimizer-level directory instead of aggregated/.
    """
    import re
    _SEED_RE = re.compile(r"^seed_\d+$")
    model_dirs = []
    for root, dirs, files in os.walk(indir):
        # Skip seed directories to avoid duplicate model discovery
        dirs[:] = [d for d in dirs if not _SEED_RE.match(d)]
        base = os.path.basename(root)

        if "phase_trajectory.csv" in files:
            model_dirs.append(root)
            continue

        if os.path.isdir(os.path.join(root, "checkpoint_alpha")):
            pats = glob.glob(os.path.join(root, "checkpoint_alpha", "ckpt_*_alpha_grad_samples.csv"))
            if pats:
                model_dirs.append(root)
                continue

        legacy = os.path.join(root, f"{base}_alpha_grad_samples.csv")
        if os.path.exists(legacy):
            model_dirs.append(root)

    return sorted(set(model_dirs))

def latest_ckpt_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    best = None
    best_ep = -1
    for f in files:
        m = CKPT_RE.search(os.path.basename(f))
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best = f
    return best

def load_alpha_samples_latest_for_model(mdir: str):
    """
    Prefer latest dynamic checkpoint samples:
      checkpoint_alpha/ckpt_XXXX_alpha_grad_samples.csv
    Else fallback:
      <model>_alpha_grad_samples.csv
    """
    base = os.path.basename(mdir)
    dyn = latest_ckpt_file(os.path.join(mdir, "checkpoint_alpha", "ckpt_*_alpha_grad_samples.csv"))
    if dyn:
        s = read_alpha_samples_csv(dyn)
        jp = os.path.join(mdir, "checkpoint_alpha",
                          os.path.basename(dyn).replace("_alpha_grad_samples.csv", "_alpha_grad.json"))
        info = safe_load_json(jp) or {}
        return s, info, dyn

    legacy_csv = os.path.join(mdir, f"{base}_alpha_grad_samples.csv")
    legacy_json = os.path.join(mdir, f"{base}_alpha_grad.json")
    s = read_alpha_samples_csv(legacy_csv) if os.path.exists(legacy_csv) else None
    info = safe_load_json(legacy_json) or {}
    return s, info, legacy_csv if os.path.exists(legacy_csv) else None

def load_alpha_trajectory(mdir: str):
    """
    Preferred: phase_trajectory.csv with columns including:
      epoch, alpha_hat, sigma_alpha_hat, alpha_reliable (optional)
    Fallback: reconstruct from checkpoint_alpha/*_alpha_grad.json (sorted by ckpt epoch).
    Returns: epochs, alpha, sigma, reliable, source_str
    (reliable is an int array: 1 = reliable, 0 = unreliable; default all-1 for backward compat)
    """
    pt = os.path.join(mdir, "phase_trajectory.csv")
    if os.path.exists(pt):
        data = np.genfromtxt(pt, delimiter=",", names=True, dtype=None, encoding=None)
        if getattr(data, "size", 0) > 0 and ("epoch" in data.dtype.names) and ("alpha_hat" in data.dtype.names):
            ep = np.array(data["epoch"], dtype=int)
            a = np.array(data["alpha_hat"], dtype=float)
            s = np.array(data["sigma_alpha_hat"], dtype=float) if ("sigma_alpha_hat" in data.dtype.names) else np.full_like(a, np.nan)
            # reliability column (backward compat: default all-1)
            if "alpha_reliable" in data.dtype.names:
                rel = np.array(data["alpha_reliable"], dtype=int)
            else:
                rel = np.ones_like(ep, dtype=int)
            mask = np.isfinite(ep) & np.isfinite(a)
            ep, a, s, rel = ep[mask], a[mask], s[mask], rel[mask]
            order = np.argsort(ep)
            return ep[order], a[order], s[order], rel[order], "phase_trajectory.csv"

    # fallback: checkpoint_alpha jsons
    jps = glob.glob(os.path.join(mdir, "checkpoint_alpha", "ckpt_*_alpha_grad.json"))
    if not jps:
        return None, None, None, None, ""

    pairs = []
    for jp in jps:
        m = CKPT_RE.search(os.path.basename(jp))
        if not m:
            continue
        ep = int(m.group(1))
        info = safe_load_json(jp) or {}
        a = info.get("alpha_hat", np.nan)
        s = info.get("sigma_alpha_hat", np.nan)
        r = int(info.get("alpha_reliable", 1))
        pairs.append((ep, float(a), float(s), r))

    if not pairs:
        return None, None, None, None, ""

    pairs.sort(key=lambda t: t[0])
    ep = np.array([t[0] for t in pairs], dtype=int)
    a = np.array([t[1] for t in pairs], dtype=float)
    s = np.array([t[2] for t in pairs], dtype=float)
    rel = np.array([t[3] for t in pairs], dtype=int)
    mask = np.isfinite(ep) & np.isfinite(a)
    ep, a, s, rel = ep[mask], a[mask], s[mask], rel[mask]
    return ep, a, s, rel, "checkpoint_alpha/*.json"

def _try_load_alpha_se(mdir):
    """Load alpha_hat_se from phase_trajectory.csv for confidence bands."""
    pt = os.path.join(mdir, "phase_trajectory.csv")
    if not os.path.exists(pt):
        return None
    try:
        data = np.genfromtxt(pt, delimiter=",", names=True, dtype=None, encoding=None)
        if "alpha_hat_se" in (data.dtype.names or ()):
            arr = np.array(data["alpha_hat_se"], dtype=float)
            return arr[np.argsort(np.array(data["epoch"], dtype=int))]
    except Exception:
        pass
    return None

# ============================================================
# Stable distribution helpers
# ============================================================

_HAS_SCIPY = False
_LEVY_STABLE = None

def _init_scipy():
    global _HAS_SCIPY, _LEVY_STABLE
    try:
        from scipy.stats import levy_stable
        _HAS_SCIPY = True
        _LEVY_STABLE = levy_stable
    except ImportError:
        _HAS_SCIPY = False
        _LEVY_STABLE = None

_init_scipy()


def stable_pdf(x, alpha, sigma, loc=0.0):
    """Evaluate symmetric alpha-stable PDF. Returns None if scipy unavailable."""
    if not _HAS_SCIPY or _LEVY_STABLE is None:
        return None
    alpha = float(np.clip(alpha, 1.01, 2.0))  # levy_stable needs alpha > 1 for stability
    sigma = max(float(sigma), 1e-12)
    try:
        return _LEVY_STABLE.pdf(x, alpha, 0.0, loc=float(loc), scale=sigma)
    except Exception:
        return None


def gaussian_pdf(x, sigma, loc=0.0):
    """Simple Gaussian PDF."""
    sigma = max(float(sigma), 1e-12)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - loc) / sigma) ** 2)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Root folder (e.g., results/exp1)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--show_unreliable", action="store_true",
                    help="Show unreliable alpha estimates as gray dashed (default: filter them out)")
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_exp1")
    os.makedirs(outdir, exist_ok=True)

    model_dirs = find_model_dirs_recursive(indir)
    if not model_dirs:
        raise RuntimeError("No model directories found (phase_trajectory.csv / checkpoint_alpha / legacy alpha files).")

    # --------- 1) DYNAMICS: alpha_hat(t) ----------
    traj = {}
    traj_meta = {}
    for mdir in model_dirs:
        m = os.path.basename(mdir)
        ep, a, s, rel, src = load_alpha_trajectory(mdir)
        if ep is None or a is None or ep.size == 0:
            continue
        traj[m] = (ep, a, s, rel)
        traj_meta[m] = {"model_dir": mdir, "traj_src": src}

    show_unreliable = getattr(args, 'show_unreliable', False)

    if traj:
        models = sorted(traj.keys())

        plt.figure(figsize=(7.6, 4.8))
        for m in models:
            ep, a, _, rel = traj[m]
            reliable_mask = (rel == 1)
            # Plot reliable points
            if np.any(reliable_mask):
                line, = plt.plot(ep[reliable_mask], a[reliable_mask], linewidth=2.0, label=m)
                # +/- 1 SE band from K-direction estimation
                se_col = _try_load_alpha_se(traj_meta[m].get("model_dir", ""))
                if se_col is not None and se_col.size == a.size:
                    se_r = se_col[reliable_mask]
                    plt.fill_between(ep[reliable_mask], a[reliable_mask] - se_r,
                                     a[reliable_mask] + se_r, alpha=0.15, color=line.get_color())
            else:
                line, = plt.plot([], [], linewidth=2.0, label=m)
            # Optionally overlay unreliable points
            unreliable_mask = (rel == 0)
            if show_unreliable and np.any(unreliable_mask):
                plt.plot(ep[unreliable_mask], a[unreliable_mask],
                         linestyle="--", linewidth=1.0, alpha=0.5, color="gray",
                         marker="x", markersize=4)
        plt.xlabel("epoch")
        plt.ylabel(r"$\hat{\alpha}(t)$")
        title = r"Gradient tail index dynamics"
        if show_unreliable:
            title += r" (gray dashed = unreliable)"
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "alpha_hat_vs_epoch.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

        any_sigma = False
        for m in models:
            ep, _, sig, rel = traj[m]
            if np.any(np.isfinite(sig) & (rel == 1)):
                any_sigma = True
        if any_sigma:
            plt.figure(figsize=(7.6, 4.8))
            for m2 in models:
                ep2, _, sig2, rel2 = traj[m2]
                rmask = (rel2 == 1)
                if np.any(np.isfinite(sig2[rmask])):
                    plt.plot(ep2[rmask], sig2[rmask], linewidth=2.0, label=m2)
            plt.xlabel("epoch")
            plt.ylabel(r"$\hat{\sigma}_{\alpha}(t)$")
            plt.title(r"Gradient projection scale dynamics")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "sigma_alpha_hat_vs_epoch.png"), dpi=args.dpi, bbox_inches="tight")
            plt.close()

        rows = []
        for m in models:
            ep, a, sig, rel = traj[m]
            n_reliable = int(np.sum(rel == 1))
            rows.append([
                m,
                traj_meta[m].get("model_dir", ""),
                traj_meta[m].get("traj_src", ""),
                int(ep[-1]),
                float(a[-1]),
                float(sig[-1]) if (sig.size and np.isfinite(sig[-1])) else np.nan,
                int(ep.size),
                n_reliable,
            ])
        write_csv(
            os.path.join(outdir, "alpha_phase_trajectory_summary.csv"),
            ["model", "model_dir", "trajectory_source", "last_epoch", "alpha_hat_last",
             "sigma_alpha_hat_last", "n_checkpoints", "n_reliable"],
            rows
        )

    # --------- 2) DISTRIBUTIONS: histogram + stable fit overlay ----------
    samples_by_model = {}
    meta_by_model = {}

    for mdir in model_dirs:
        m = os.path.basename(mdir)
        s, info, src = load_alpha_samples_latest_for_model(mdir)
        if s is None or s.size == 0:
            continue
        samples_by_model[m] = s
        meta_by_model[m] = {
            "model_dir": mdir,
            "alpha_src": src,
            "alpha_hat": float((info or {}).get("alpha_hat", np.nan)),
            "sigma_alpha_hat": float((info or {}).get("sigma_alpha_hat", np.nan)),
            "alpha_reliable": int((info or {}).get("alpha_reliable", 1)),
        }

    if samples_by_model:
        models = sorted(samples_by_model.keys())
        n_models = len(models)

        # ---- Multi-panel figure: one panel per model ----
        ncols = min(3, n_models)
        nrows = max(1, (n_models + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

        for idx, m in enumerate(models):
            row_i = idx // ncols
            col_i = idx % ncols
            ax = axes[row_i][col_i]

            s = samples_by_model[m]
            meta = meta_by_model.get(m, {})
            alpha_hat = meta.get("alpha_hat", np.nan)
            sigma_hat = meta.get("sigma_alpha_hat", np.nan)

            # center the samples (remove mean for symmetric fit)
            s_mean = float(np.mean(s))
            s_centered = s - s_mean

            # histogram
            ax.hist(s, bins=args.bins, density=True, alpha=0.35, color="steelblue",
                    edgecolor="white", linewidth=0.3, label="data")

            # grid for overlays
            lo, hi = np.quantile(s, [0.001, 0.999])
            margin = 0.15 * (hi - lo)
            grid = np.linspace(lo - margin, hi + margin, 500)

            # fitted alpha-stable PDF
            if np.isfinite(alpha_hat) and np.isfinite(sigma_hat):
                stable_y = stable_pdf(grid, alpha_hat, sigma_hat, loc=s_mean)
                if stable_y is not None:
                    ax.plot(grid, stable_y, color="crimson", linewidth=2.0,
                            label=rf"$\alpha$-stable ($\hat{{\alpha}}={alpha_hat:.2f}$)")

                # Gaussian with matched scale (sigma_hat from stable fit)
                gauss_y = gaussian_pdf(grid, sigma_hat, loc=s_mean)
                ax.plot(grid, gauss_y, color="forestgreen", linewidth=1.5, linestyle="--",
                        label=rf"Gaussian ($\sigma={sigma_hat:.4f}$)")

            ax.set_xlabel("grad projection")
            ax.set_ylabel("density")
            rel_flag = meta.get("alpha_reliable", 1)
            rel_tag = "" if rel_flag == 1 else " [UNRELIABLE]"
            ax.set_title(f"{m}{rel_tag}")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2)

        # hide unused axes
        for idx in range(n_models, nrows * ncols):
            row_i = idx // ncols
            col_i = idx % ncols
            axes[row_i][col_i].set_visible(False)

        fig.suptitle("Gradient noise: histogram + fitted distributions (final checkpoint)", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "alpha_grad_stable_fit.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # ---- Overlay histogram (all models together, simple) ----
        plt.figure(figsize=(7.0, 4.6))
        for m in models:
            s = samples_by_model[m]
            plt.hist(s, bins=args.bins, density=True, alpha=0.25, label=m)
        plt.xlabel("grad projection samples")
        plt.ylabel("density")
        plt.title("Gradient noise proxy samples (histogram) — final checkpoint per model")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "alpha_grad_hist.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

        # ---- Summary CSV ----
        rows = []
        for m in models:
            s = samples_by_model[m]
            meta = meta_by_model.get(m, {})
            rows.append([
                m,
                meta.get("model_dir", ""),
                meta.get("alpha_src", ""),
                float(meta.get("alpha_hat", np.nan)),
                float(meta.get("sigma_alpha_hat", np.nan)),
                float(np.mean(s)) if s.size else np.nan,
                float(np.std(s)) if s.size else np.nan,
                int(s.size),
            ])

        write_csv(
            os.path.join(outdir, "alpha_grad_summary.csv"),
            ["model", "model_dir", "alpha_samples_source", "alpha_hat", "sigma_alpha_hat",
             "grad_proj_mean_emp", "grad_proj_std_emp", "n_samples"],
            rows
        )

    if (not traj) and (not samples_by_model):
        raise RuntimeError("No alpha trajectory or alpha sample data could be loaded.")

    print(f"[OK] Saved alpha plots to: {outdir}")

if __name__ == "__main__":
    main()
