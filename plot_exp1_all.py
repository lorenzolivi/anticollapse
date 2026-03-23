#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess, argparse, datetime

SCRIPTS = [
    "plot_exp1_envelopes.py",
    "plot_exp1_tau_spectrum.py",
    "plot_exp1_alpha_grad.py",
    "plot_exp1_phase_summary.py",
    "plot_exp1_learning_curves.py",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Experiment root folder (e.g., exp1/ containing baselines/ and lstmgru/)")
    ap.add_argument("--outdir", default=None, help="Plot output folder (default: <indir>/plots_exp1)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ylog", type=int, default=0, help="If 1, pass --ylog 1 to learning-curves script.")
    ap.add_argument("--min_r2", type=float, default=None, help="Forward to plot_exp1_phase_summary.py")
    ap.add_argument("--min_beta_r2", type=float, default=None, help="Forward to plot_exp1_tau_spectrum.py")
    ap.add_argument("--hist_bins", type=int, default=None, help="Forward to plot_exp1_tau_spectrum.py")
    ap.add_argument("--bins", type=int, default=None, help="Forward to plot_exp1_alpha_grad.py")
    ap.add_argument("--debug", type=int, default=None, help="Forward to plot_exp1_envelopes.py")
    ap.add_argument("--tau_cap", type=float, default=None, help="Forward to plot_exp1_tau_spectrum.py")
    args = ap.parse_args()

    indir = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir or os.path.join(indir, "plots_exp1"))
    os.makedirs(outdir, exist_ok=True)

    # Ensure scripts run relative to *this* file, not the CWD of the caller.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(outdir, "plot_driver_log.txt")
    with open(log_path, "w") as lf:
        lf.write(f"Plot driver run at {datetime.datetime.now()}\n")
        lf.write(f"indir={indir}\n")
        lf.write(f"outdir={outdir}\n")
        lf.write(f"dpi={args.dpi}\n")
        lf.write(f"ylog={args.ylog}\n\n")

    ok, fail = [], []

    for script in SCRIPTS:
        script_path = os.path.join(script_dir, script)

        # ---- Missing script safety check ----
        if not os.path.exists(script_path):
            print(f"[SKIP] Missing script: {script_path}")
            fail.append(script)
            with open(log_path, "a") as lf:
                lf.write(f"[FAIL] {script}\n")
                lf.write(f"missing script: {script_path}\n\n")
            continue
        # -------------------------------------

        cmd = [sys.executable, script_path, "--indir", indir, "--outdir", outdir, "--dpi", str(args.dpi)]

        # R8: per-script optional flags
        if script == "plot_exp1_learning_curves.py":
            cmd += ["--ylog", str(args.ylog)]
        if script == "plot_exp1_phase_summary.py" and args.min_r2 is not None:
            cmd += ["--min_r2", str(args.min_r2)]
        if script == "plot_exp1_tau_spectrum.py":
            if args.min_beta_r2 is not None:
                cmd += ["--min_beta_r2", str(args.min_beta_r2)]
            if args.hist_bins is not None:
                cmd += ["--hist_bins", str(args.hist_bins)]
            if args.tau_cap is not None:
                cmd += ["--tau_cap", str(args.tau_cap)]
        if script == "plot_exp1_alpha_grad.py" and args.bins is not None:
            cmd += ["--bins", str(args.bins)]
        if script == "plot_exp1_envelopes.py" and args.debug is not None:
            cmd += ["--debug", str(args.debug)]

        print("[RUN]", " ".join(cmd), flush=True)
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)
            ok.append(script)
            with open(log_path, "a") as lf:
                lf.write(f"[OK] {script}\n")
                if r.stdout.strip():
                    lf.write(r.stdout + "\n")
                if r.stderr.strip():
                    lf.write(r.stderr + "\n")
                lf.write("\n")
        except subprocess.CalledProcessError as e:
            fail.append(script)
            with open(log_path, "a") as lf:
                lf.write(f"[FAIL] {script}\n")
                lf.write(f"returncode={e.returncode}\n")
                if e.stdout:
                    lf.write(e.stdout + "\n")
                if e.stderr:
                    lf.write(e.stderr + "\n")
                lf.write("\n")
            continue

    print(f"[DONE] ok={len(ok)} fail={len(fail)}")
    print(f"Log saved to: {log_path}")
    if fail:
        print("Failed scripts:", ", ".join(fail))

if __name__ == "__main__":
    main()