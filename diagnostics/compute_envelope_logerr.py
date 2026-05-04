#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute log-space envelope error metrics against a reference width.

Default input layout:
    diagnostics/results_width/{arch}_H{width}_seed{seed}/envelope.csv

Each envelope CSV is expected to contain columns:
    lag, f_actual

Metrics:
    delta_inf    = sup_ell |log10 f_H(ell) - log10 f_ref(ell)|
    delta_2      = sqrt(mean_ell diff_ell^2)
    delta_2_w    = envelope-weighted RMSE, weights f_ref(ell)
    pearson_r    = Pearson correlation between log10 envelopes

Seed averaging is done in linear envelope space before taking logs, matching
the width-scaling pipeline.
"""

import argparse
import csv
import json
import math
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _parse_csv_list(value: str, cast):
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def load_envelope(root: Path, arch: str, width: int, seed: int):
    path = root / f"{arch}_H{width}_seed{seed}" / "envelope.csv"
    lags, fs = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lags.append(int(row["lag"]))
            fs.append(float(row["f_actual"]))
    return lags, fs


def seed_average(root: Path, arch: str, width: int, seeds):
    lags_ref = None
    stacked = []
    for seed in seeds:
        lags, fs = load_envelope(root, arch, width, seed)
        if lags_ref is None:
            lags_ref = lags
        elif lags != lags_ref:
            raise ValueError(f"lag grid mismatch for arch={arch}, H={width}, seed={seed}")
        stacked.append(fs)
    n_seeds = len(stacked)
    if n_seeds == 0 or lags_ref is None:
        raise ValueError(f"no envelopes found for arch={arch}, H={width}")
    n_lags = len(lags_ref)
    mean = [
        sum(stacked[i][lag_idx] for i in range(n_seeds)) / n_seeds
        for lag_idx in range(n_lags)
    ]
    return lags_ref, mean


def log_diffs(f_width, f_ref):
    out = []
    for a, b in zip(f_width, f_ref):
        if a <= 0 or b <= 0:
            raise ValueError(f"non-positive envelope value: a={a}, b={b}")
        out.append(math.log10(a) - math.log10(b))
    return out


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def compute_metrics(root: Path, arch: str, widths, seeds, ref_width: int, thresh: float):
    _, f_ref = seed_average(root, arch, ref_width, seeds)
    log_ref = [math.log10(v) for v in f_ref]
    idx_thresh = [i for i, v in enumerate(f_ref) if v >= thresh]
    if not idx_thresh:
        raise ValueError(f"no reference lags above threshold={thresh} for arch={arch}")
    weight_sum = sum(f_ref)

    per_width = []
    for width in widths:
        _, f_width = seed_average(root, arch, width, seeds)
        log_width = [math.log10(v) for v in f_width]
        diffs = log_diffs(f_width, f_ref)

        d_inf = max(abs(diffs[i]) for i in idx_thresh)
        d_2 = math.sqrt(sum(diffs[i] ** 2 for i in idx_thresh) / len(idx_thresh))
        d_2_w = math.sqrt(
            sum(f_ref[i] * diffs[i] ** 2 for i in range(len(f_ref))) / weight_sum
        )
        r = pearson(log_width, log_ref)
        per_width.append({
            "H": int(width),
            "delta_inf": d_inf,
            "delta_2": d_2,
            "delta_2_w": d_2_w,
            "pearson_r": r,
            "n_thresh_lags": len(idx_thresh),
            "f_head": f_width[:3],
            "f_tail": f_width[-3:],
        })
    return per_width


def main():
    parser = argparse.ArgumentParser(description="Compute width-scaling envelope log-error metrics.")
    parser.add_argument("--root", type=Path, default=THIS_DIR / "results_width",
                        help="Root directory containing {arch}_H{width}_seed{seed}/ folders.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path. Defaults to <root>/envelope_logerr.json.")
    parser.add_argument("--arches", default="diag,gru")
    parser.add_argument("--widths", default="16,32,64,128,256,512")
    parser.add_argument("--seeds", default="42,123,321,456,789")
    parser.add_argument("--ref_width", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out_path = args.out.expanduser().resolve() if args.out else root / "envelope_logerr.json"
    arches = _parse_csv_list(args.arches, str)
    widths = _parse_csv_list(args.widths, int)
    seeds = _parse_csv_list(args.seeds, int)

    results = {}
    for arch in arches:
        results[arch] = compute_metrics(
            root=root,
            arch=arch,
            widths=widths,
            seeds=seeds,
            ref_width=args.ref_width,
            thresh=args.threshold,
        )

    for arch in arches:
        n_thr = results[arch][0]["n_thresh_lags"] if results[arch] else 0
        print(f"\n=== {arch} (thresholded over {n_thr} lags) ===")
        print(f"{'H':>5} {'Delta_inf':>12} {'Delta_2':>12} {'Delta_2_w':>12} {'Pearson r':>12}")
        for row in results[arch]:
            print(
                f"{row['H']:>5} {row['delta_inf']:>12.6f} "
                f"{row['delta_2']:>12.6f} {row['delta_2_w']:>12.6f} "
                f"{row['pearson_r']:>12.6f}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
