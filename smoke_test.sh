#!/usr/bin/env bash
# ===========================================================================
# smoke_test.sh — Medium-scale validation run for Anti-Collapse
# ===========================================================================
#
# Runs a medium-scale version of the experiment pipeline that is serious
# enough to exercise the Section 7 crossover diagnostic and cross-seed
# aggregation, while still being quick enough to iterate on a Mac.
# Use this before launching the full paper-scale simulations on DGX.
#
# Usage:
#   ./smoke_test.sh exp1          # Experiment 1 only
#   ./smoke_test.sh exp2          # Experiment 2 only
#   ./smoke_test.sh all           # Both experiments
#   ./smoke_test.sh --help        # Show this help message
#
# Environment:
#   OUTDIR   — Root output directory  (default: results/smoke_test)
#   DEVICE   — Compute device         (default: auto)
#   DPI      — Plot resolution        (default: 150)
#
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-results/smoke_test}"
DEVICE="${DEVICE:-auto}"
DPI="${DPI:-150}"

# Medium-scale settings: serious enough to exercise the Section 7 crossover
# diagnostic and cross-seed aggregation, quick enough to iterate on a Mac.
# See README/notes for the three tiers (smoke / medium / paper).
SEEDS="42,123"
EPOCHS=150
CHECKPOINT_EVERY=30
H=128
T=256
NSEQ_TRAIN=1500
NSEQ_DIAG=750
BATCH_SIZE=128
DIAG_BATCH_SIZE=128
ALPHA_N_GRAD_BATCHES=64
ALPHA_GRAD_BATCH_SIZE=128
ALPHA_N_DIRECTIONS=3
TAU_FIT_NUM_LAGS=16
NUM_LAGS=48
BETA_BOOTSTRAP_B=500
BETA_BOOTSTRAP_CI=0.90
PHASE_R2_THRESHOLD=0.90

usage() {
    cat <<'EOF'
smoke_test.sh — Quick validation run (reduced grid) for Anti-Collapse

Usage:
    ./smoke_test.sh exp1          Run Experiment 1 (phase dynamics)
    ./smoke_test.sh exp2          Run Experiment 2 (ablation study)
    ./smoke_test.sh all           Run both experiments
    ./smoke_test.sh --help        Show this help message

Environment variables:
    OUTDIR   Root output directory  (default: results/smoke_test)
    DEVICE   Compute device         (default: auto)
    DPI      Plot resolution        (default: 150)

This script uses a medium-scale grid (2 seeds, 150 epochs, H=128,
num_lags=48, bootstrap B=500) — serious enough to exercise the
Section 7 crossover diagnostic and cross-seed aggregation, quick
enough to iterate on a Mac.  Typical runtime on M5+MPS is
~15-25 min (exp1) and ~25-40 min (exp2).
EOF
    exit 0
}

run_exp1() {
    echo "=== Smoke Test: Experiment 1 (Phase Dynamics) ==="
    python "$SCRIPT_DIR/main_exp1.py" \
        --outdir "$OUTDIR/exp1" \
        --seeds "$SEEDS" \
        --models const,shared,diag,gru,lstm \
        --epochs "$EPOCHS" \
        --checkpoint_every "$CHECKPOINT_EVERY" \
        --H "$H" \
        --T "$T" \
        --Nseq_train "$NSEQ_TRAIN" \
        --Nseq_diag "$NSEQ_DIAG" \
        --batch_size "$BATCH_SIZE" \
        --diag_batch_size "$DIAG_BATCH_SIZE" \
        --alpha_n_grad_batches_ckpt "$ALPHA_N_GRAD_BATCHES" \
        --alpha_grad_batch_size "$ALPHA_GRAD_BATCH_SIZE" \
        --alpha_n_directions "$ALPHA_N_DIRECTIONS" \
        --tau_fit_num_lags "$TAU_FIT_NUM_LAGS" \
        --num_lags "$NUM_LAGS" \
        --beta_bootstrap_B "$BETA_BOOTSTRAP_B" \
        --beta_bootstrap_ci "$BETA_BOOTSTRAP_CI" \
        --phase_r2_threshold "$PHASE_R2_THRESHOLD" \
        --device "$DEVICE" \
        --dpi "$DPI" \
        --save_final_envelope \
        --save_checkpoint_ccdf
    echo "=== Experiment 1 smoke test DONE ==="
}

run_exp2() {
    echo "=== Smoke Test: Experiment 2 (Ablation Study) ==="
    python "$SCRIPT_DIR/main_exp2.py" \
        --outdir "$OUTDIR/exp2" \
        --seeds "$SEEDS" \
        --models diag,gru,lstm \
        --epochs "$EPOCHS" \
        --checkpoint_every "$CHECKPOINT_EVERY" \
        --H "$H" \
        --T "$T" \
        --Nseq_train "$NSEQ_TRAIN" \
        --Nseq_diag "$NSEQ_DIAG" \
        --batch_size "$BATCH_SIZE" \
        --diag_batch_size "$DIAG_BATCH_SIZE" \
        --alpha_n_grad_batches_ckpt "$ALPHA_N_GRAD_BATCHES" \
        --alpha_grad_batch_size "$ALPHA_GRAD_BATCH_SIZE" \
        --alpha_n_directions "$ALPHA_N_DIRECTIONS" \
        --tau_fit_num_lags "$TAU_FIT_NUM_LAGS" \
        --num_lags "$NUM_LAGS" \
        --beta_bootstrap_B "$BETA_BOOTSTRAP_B" \
        --beta_bootstrap_ci "$BETA_BOOTSTRAP_CI" \
        --phase_r2_threshold "$PHASE_R2_THRESHOLD" \
        --device "$DEVICE" \
        --dpi "$DPI" \
        --batch_ablation_values "2048" \
        --clip_ablation_values "0.1" \
        --winsorize_ablation_values "95" \
        --warmup_epochs 10 \
        --save_final_envelope \
        --save_checkpoint_ccdf
    echo "=== Experiment 2 smoke test DONE ==="
}

# Parse argument
case "${1:-}" in
    exp1)
        run_exp1
        ;;
    exp2)
        run_exp2
        ;;
    all)
        run_exp1
        run_exp2
        ;;
    --help|-h|help)
        usage
        ;;
    "")
        echo "Error: specify exp1, exp2, or all. Use --help for details." >&2
        exit 1
        ;;
    *)
        echo "Error: unknown argument '$1'. Use --help for details." >&2
        exit 1
        ;;
esac
