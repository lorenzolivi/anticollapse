#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline envelope re-computation.

Walks a results directory (e.g. ``results/exp1_constgate_full/adamw``) and, for
every ``seed_NNNN/<model>/analysis_checkpoint/final.pt`` it finds, re-runs the
final-envelope analysis pipeline from saved checkpoints — without retraining.

This is needed whenever the macro-envelope semantics in
``diagnostics.compute_macro_envelope_comparison`` changes (e.g. switching from
the legacy ``mu0 + mu1`` kernel to the canonical zero-order ``mu0`` envelope).
The new run will overwrite:

    <seed>/<model>/<model>_envelope.csv
    <seed>/<model>/<model>_envelope_fit.json
    <seed>/<model>/<model>_envelope_fit_curves.csv
    <seed>/<model>/<model>_envelope_audit.csv         (new file)
    <seed>/<model>/<model>_gelr_envelope_compare.csv
    <seed>/<model>/<model>_gelr_fit.json
    <seed>/<model>/<model>_gelr_fit_curves.csv
    <seed>/<model>/<model>_adaptive_base_rates.csv
    <seed>/<model>/<model>_final_phase.json

The original ``analysis_checkpoint/final.pt`` is not modified.

Usage
-----
Regenerate all seeds for Path B (full run, ConstGate only):

    python regenerate_envelopes.py \
        --results_root results/exp1_constgate_full/adamw \
        --models const

Regenerate Path A (full):

    python regenerate_envelopes.py \
        --results_root results/exp1_constgate_inject_full/adamw \
        --models const

Add ``--dry_run`` to print what would be re-run without doing the work.
"""

import argparse
import glob
import json
import os
import sys
from typing import List, Optional


def _build_args_namespace(cli_args: dict, outdir: str, models: str,
                          analysis_only: bool = True):
    """Reconstruct an argparse Namespace from a saved cli_args dict.

    The training run wrote a complete CLI args dictionary to
    ``<seed>/cli_args.json``.  We use that to seed an analysis-only call,
    overriding only ``outdir``, ``models``, and ``analysis_only``.
    """
    ns = argparse.Namespace()
    for k, v in cli_args.items():
        setattr(ns, k, v)
    ns.outdir = outdir
    ns.models = models
    ns.analysis_only = bool(analysis_only)

    # Defensive defaults — older cli_args dumps may be missing fields
    # introduced after the original run.
    _defaults = {
        "device": "cpu",
        "alpha_n_grad_batches_ckpt": 256,
        "alpha_grad_batch_size": 256,
        "alpha_use_grad_clip": False,
        "alpha_n_directions": 5,
        "alpha_method": "ecf",
        "min_samples_alpha": 500,
        "task_variant": "fixed",
        "task_alpha": 1.0,
        "task_lag_min": 8,
        "task_lag_max": 384,
        "task_K": 8,
        "task_coeff_base": 0.6,
        "task_coeff_decay": 0.85,
        "task_lag_seed": 20260410,
        "inject_alpha_noise": 0.0,
        "inject_alpha": 1.6,
        "inject_grad_seed_offset": 1729,
        "save_checkpoint_ccdf": False,
        "save_final_envelope": False,
        "save_analysis_checkpoint": False,
        "save_model_checkpoints": False,
    }
    for k, v in _defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def find_seed_dirs(results_root: str) -> List[str]:
    """Find all seed_NNNN directories under results_root that contain
    a cli_args.json and at least one model with an analysis_checkpoint."""
    pattern = os.path.join(results_root, "seed_*")
    candidates = sorted(glob.glob(pattern))
    seed_dirs = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, "cli_args.json")):
            continue
        seed_dirs.append(path)
    return seed_dirs


def regenerate_one(seed_dir: str, models: str, device: Optional[str]) -> bool:
    """Run --analysis_only style regeneration for one seed dir.

    Returns True on success, False on missing inputs / failure.
    """
    cli_path = os.path.join(seed_dir, "cli_args.json")
    if not os.path.isfile(cli_path):
        print(f"[skip] no cli_args.json in {seed_dir}", flush=True)
        return False
    with open(cli_path, "r") as jf:
        cli_args = json.load(jf)

    # Sanity-check that at least one of the requested models has a
    # saved analysis_checkpoint.
    have_any = False
    for mname in [m.strip().lower() for m in models.split(",") if m.strip()]:
        ckpt = os.path.join(seed_dir, mname, "analysis_checkpoint", "final.pt")
        if os.path.isfile(ckpt):
            have_any = True
            break
    if not have_any:
        print(f"[skip] no analysis_checkpoint under {seed_dir}", flush=True)
        return False

    ns = _build_args_namespace(cli_args, outdir=seed_dir, models=models,
                               analysis_only=True)
    if device is not None:
        ns.device = device

    # Import lazily so --dry_run works without importing torch.
    from run_phase_trajectory import main as run_phase_trajectory_main
    # run_phase_trajectory.main() reads from parse_args; override it here.
    import run_phase_trajectory as _run_phase_trajectory
    _orig_parse = _run_phase_trajectory.parse_args
    _run_phase_trajectory.parse_args = lambda: ns
    try:
        run_phase_trajectory_main()
    finally:
        _run_phase_trajectory.parse_args = _orig_parse
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", type=str, required=True,
                    help="Path to <results>/<optimizer> (the directory "
                         "containing the seed_NNNN folders).")
    ap.add_argument("--models", type=str, default="const",
                    help="Comma-separated model names to regenerate "
                         "(default: const)")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (e.g. cpu, cuda, mps). "
                         "Default: whatever cli_args.json had, or cpu.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Just list what would be regenerated.")
    args = ap.parse_args()

    if not os.path.isdir(args.results_root):
        print(f"results_root does not exist: {args.results_root}", file=sys.stderr)
        sys.exit(2)

    seed_dirs = find_seed_dirs(args.results_root)
    if not seed_dirs:
        print(f"no seed_* dirs found under {args.results_root}", file=sys.stderr)
        sys.exit(2)

    print(f"[regenerate_envelopes] found {len(seed_dirs)} seed dirs", flush=True)
    for sd in seed_dirs:
        print(f"  - {sd}", flush=True)

    if args.dry_run:
        print("[dry_run] no changes made.", flush=True)
        return

    n_ok = 0
    n_fail = 0
    for sd in seed_dirs:
        print(f"\n[regenerate_envelopes] === {sd} ===", flush=True)
        try:
            ok = regenerate_one(sd, args.models, args.device)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[error] {sd}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n[regenerate_envelopes] done. ok={n_ok} fail={n_fail}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
