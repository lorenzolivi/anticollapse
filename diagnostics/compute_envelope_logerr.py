"""
Compute log-space envelope error metrics against the H=512 reference:
  Delta_inf(H)     = sup_ell |log10 f_H(ell) - log10 f_512(ell)|      (thresholded)
  Delta_2(H)       = sqrt( (1/L) sum_ell diff_ell^2 )                 (thresholded)
  Delta_2_w(H)     = sqrt( sum_ell w_ell * diff_ell^2 / sum_ell w_ell ),
                     with w_ell = f_512(ell)  (envelope-weighted RMSE; no
                     threshold needed because deep-tail lags carry vanishing
                     weight).
  pearson_r(H)     = Pearson correlation of (log10 f_H, log10 f_512) over ell.

Delta_inf and Delta_2 are evaluated over lags with f_512(ell) >= THRESH (= 1e-3).
For DiagGate all 35 lags satisfy it; for GRU only the first 10 do (f decays to
~1e-42 by lag 140). Delta_2_w is summed over all 35 lags.

Data layout: diagnostics/results_width/{arch}_H{H}_seed{S}/envelope.csv
where arch in {"diag","gru"}, columns "lag,f_actual".

Seed averaging is done in linear space (mean envelope across 5 seeds) before
taking log10, to match the Pearson-r pipeline.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path("/sessions/jolly-friendly-volta/mnt/anticollapse/diagnostics/results_width")
ARCHES = ["diag", "gru"]
WIDTHS = [16, 32, 64, 128, 256, 512]
SEEDS = [42, 123, 321, 456, 789]
THRESH = 1e-3


def load_envelope(arch: str, H: int, seed: int):
    path = ROOT / f"{arch}_H{H}_seed{seed}" / "envelope.csv"
    lags, fs = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lags.append(int(row["lag"]))
            fs.append(float(row["f_actual"]))
    return lags, fs


def seed_average(arch: str, H: int):
    lags_ref = None
    stacked = []
    for s in SEEDS:
        lags, fs = load_envelope(arch, H, s)
        if lags_ref is None:
            lags_ref = lags
        else:
            assert lags == lags_ref, (arch, H, s)
        stacked.append(fs)
    n_seeds = len(stacked)
    L = len(lags_ref)
    mean = [sum(stacked[i][l] for i in range(n_seeds)) / n_seeds for l in range(L)]
    return lags_ref, mean


def log_diffs(f_H, f_ref):
    out = []
    for a, b in zip(f_H, f_ref):
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
    return sxy / math.sqrt(sxx * syy)


def main():
    results = {}
    for arch in ARCHES:
        _, f_ref = seed_average(arch, 512)
        log_ref = [math.log10(v) for v in f_ref]
        idx_thresh = [i for i, v in enumerate(f_ref) if v >= THRESH]
        weight_sum = sum(f_ref)
        per_width = []
        for H in WIDTHS:
            _, f_H = seed_average(arch, H)
            log_H = [math.log10(v) for v in f_H]
            diffs = log_diffs(f_H, f_ref)
            # thresholded metrics
            d_inf = max(abs(diffs[i]) for i in idx_thresh)
            d_2 = math.sqrt(sum(diffs[i] ** 2 for i in idx_thresh) / len(idx_thresh))
            # weighted (no threshold) RMSE
            d_2_w = math.sqrt(
                sum(f_ref[i] * diffs[i] ** 2 for i in range(len(f_ref))) / weight_sum
            )
            # Pearson over all 35 lags
            r = pearson(log_H, log_ref)
            per_width.append(
                dict(
                    H=H,
                    delta_inf=d_inf,
                    delta_2=d_2,
                    delta_2_w=d_2_w,
                    pearson_r=r,
                    n_thresh_lags=len(idx_thresh),
                    f_head=f_H[:3],
                    f_tail=f_H[-3:],
                )
            )
        results[arch] = per_width

    # pretty print
    for arch in ARCHES:
        n_thr = results[arch][0]["n_thresh_lags"]
        print(f"\n=== {arch}  (thresholded over {n_thr} lags; weighted over 35) ===")
        print(
            f"{'H':>5} {'Delta_inf':>12} {'Delta_2':>12} "
            f"{'Delta_2_w':>12} {'Pearson r':>12}"
        )
        for row in results[arch]:
            print(
                f"{row['H']:>5} {row['delta_inf']:>12.6f} "
                f"{row['delta_2']:>12.6f} {row['delta_2_w']:>12.6f} "
                f"{row['pearson_r']:>12.6f}"
            )

    out_path = (
        "/sessions/jolly-friendly-volta/mnt/anticollapse/diagnostics/"
        "results_width/envelope_logerr.json"
    )
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
