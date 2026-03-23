#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Recursive discovery utilities
# ----------------------------

def find_model_dirs_with_learning_curve(indir: str):
    """
    Returns a list of dicts:
      [{"model": <name>, "dir": <abs/rel dir>, "csv": <full path>, "label": <plot label>}, ...]
    where <dir> is the directory that CONTAINS the learning_curve.csv.

    Skips seed_XXXX directories to avoid duplicate discovery when called
    from the optimizer-level directory instead of aggregated/.
    """
    import re
    _SEED_RE = re.compile(r"^seed_\d+$")
    out = []
    indir = os.path.abspath(indir)

    for root, dirs, files in os.walk(indir):
        # Skip seed directories to avoid duplicate model discovery
        dirs[:] = [d for d in dirs if not _SEED_RE.match(d)]
        # prefer exact <folder>/<folder>_learning_curve.csv
        folder_name = os.path.basename(root)
        exact = f"{folder_name}_learning_curve.csv"
        if exact in files:
            csv_path = os.path.join(root, exact)
            rel = os.path.relpath(root, indir)
            out.append({
                "model": folder_name,
                "dir": root,
                "csv": csv_path,
                "label": folder_name
            })
            continue

        # fallback: any *_learning_curve.csv in this folder
        lc_files = [f for f in files if f.endswith("_learning_curve.csv")]
        if lc_files:
            lc_files.sort()
            csv_path = os.path.join(root, lc_files[0])
            rel = os.path.relpath(root, indir)
            # infer model name from file prefix if possible
            inferred = os.path.basename(lc_files[0]).replace("_learning_curve.csv", "")
            model = inferred if inferred else folder_name
            out.append({
                "model": model,
                "dir": root,
                "csv": csv_path,
                "label": model
            })

    # de-duplicate by csv path
    uniq = {}
    for item in out:
        uniq[item["csv"]] = item
    out = list(uniq.values())

    # stable sort: by label
    out.sort(key=lambda d: d["label"])
    return out

def read_learning_curve(path: str):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if getattr(data, "size", 0) == 0:
        return None
    if data.shape == ():
        data = np.array([data])

    if ("epoch" not in data.dtype.names) or ("train_loss" not in data.dtype.names):
        return None

    epoch = np.array(data["epoch"], dtype=int)
    loss = np.array(data["train_loss"], dtype=float)
    mask = np.isfinite(epoch) & np.isfinite(loss)
    return epoch[mask], loss[mask]

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Experiment root folder (can be nested)")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <indir>/plots_exp1)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylog", type=int, default=0, help="If 1, use log-scale on y-axis")
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_exp1")
    os.makedirs(outdir, exist_ok=True)

    entries = find_model_dirs_with_learning_curve(indir)
    if not entries:
        raise RuntimeError(f"No learning-curve CSVs found under: {indir}")

    # Overlay plot
    plt.figure(figsize=(7.6, 4.8))
    any_data = False
    for e in entries:
        d = read_learning_curve(e["csv"])
        if d is None:
            continue
        epoch, loss = d
        if epoch.size == 0:
            continue
        any_data = True
        plt.plot(epoch, loss, linewidth=1.8, label=e["label"])

    if any_data:
        if args.ylog == 1:
            plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("train loss (MSE)")
        plt.title("Learning curves (train loss)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "learning_curves_train_loss.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # Per-model plots
    perdir = os.path.join(outdir, "learning_curves_train_loss_per_model")
    os.makedirs(perdir, exist_ok=True)

    for e in entries:
        d = read_learning_curve(e["csv"])
        if d is None:
            continue
        epoch, loss = d
        if epoch.size == 0:
            continue

        safe_name = e["label"].replace("/", "__").replace("\\", "__")
        plt.figure(figsize=(7.0, 4.4))
        plt.plot(epoch, loss, linewidth=2.0)
        if args.ylog == 1:
            plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("train loss (MSE)")
        plt.title(f"Learning curve ({e['label']})")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(perdir, f"learning_curve_{safe_name}.png"), dpi=args.dpi, bbox_inches="tight")
        plt.close()

    print(f"[OK] Saved learning-curve plots to: {outdir}")

if __name__ == "__main__":
    main()