#!/usr/bin/env bash
# ===========================================================================
# anticollapse.sh — Full paper-scale simulation for Anti-Collapse
# ===========================================================================
#
# Runs the complete experiment pipeline at paper-quality settings.
# Intended for DGX / multi-GPU cluster execution.
#
# Usage:
#   ./anticollapse.sh exp1        # Experiment 1 only
#   ./anticollapse.sh exp2        # Experiment 2 only
#   ./anticollapse.sh all         # Both experiments
#   ./anticollapse.sh --help      # Show this help message
#
# Environment:
#   OUTDIR   — Root output directory  (default: results)
#   DEVICE   — Compute device         (default: auto → cuda if available)
#   DPI      — Plot resolution        (default: 600)
#   SEEDS    — Comma-separated seeds  (default: 42,123,321)
#
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-results}"
DEVICE="${DEVICE:-auto}"
DPI="${DPI:-600}"
SEEDS="${SEEDS:-42,123,321}"

# Paper-scale settings
EPOCHS=1000
CHECKPOINT_EVERY=50
H=512
T=1024
D=16
NSEQ_TRAIN=8000
NSEQ_DIAG=8000
BATCH_SIZE=512
DIAG_BATCH_SIZE=256
ALPHA_N_GRAD_BATCHES=256
ALPHA_GRAD_BATCH_SIZE=256
ALPHA_N_DIRECTIONS=5
TAU_FIT_NUM_LAGS=24
NUM_LAGS=128
BETA_BOOTSTRAP_B=2000
BETA_BOOTSTRAP_CI=0.90
PHASE_R2_THRESHOLD=0.90

usage() {
    cat <<'EOF'
anticollapse.sh — Full paper-scale simulation for Anti-Collapse

Usage:
    ./anticollapse.sh exp1        Run Experiment 1 (phase dynamics)
    ./anticollapse.sh exp2        Run Experiment 2 (ablation study)
    ./anticollapse.sh all         Run both experiments
    ./anticollapse.sh --help      Show this help message

Environment variables:
    OUTDIR   Root output directory  (default: results)
    DEVICE   Compute device         (default: auto)
    DPI      Plot resolution        (default: 600)
    SEEDS    Comma-separated seeds  (default: 42,123,321)

Paper-scale defaults:
    H=512, T=1024, D=16, epochs=1000, checkpoint_every=50
    3 seeds, B=2000 bootstrap resamples, 90% stability interval
    5 projection directions, 256 gradient batches per checkpoint
EOF
    exit 0
}

run_exp1() {
    echo "=== Anti-Collapse: Experiment 1 (Phase Dynamics) ==="
    echo "    Seeds: $SEEDS | Device: $DEVICE | Output: $OUTDIR/exp1"
    python "$SCRIPT_DIR/main_exp1.py" \
        --outdir "$OUTDIR/exp1" \
        --seeds "$SEEDS" \
        --models const,shared,diag,gru,lstm \
        --epochs "$EPOCHS" \
        --checkpoint_every "$CHECKPOINT_EVERY" \
        --H "$H" \
        --T "$T" \
        --D "$D" \
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
        --orth_init \
        --save_final_envelope \
        --save_checkpoint_ccdf \
        --save_model_checkpoints
    echo "=== Experiment 1 DONE ==="
}

run_exp2() {
    echo "=== Anti-Collapse: Experiment 2 (Ablation Study) ==="
    echo "    Seeds: $SEEDS | Device: $DEVICE | Output: $OUTDIR/exp2"
    python "$SCRIPT_DIR/main_exp2.py" \
        --outdir "$OUTDIR/exp2" \
        --seeds "$SEEDS" \
        --models diag,gru,lstm \
        --epochs "$EPOCHS" \
        --checkpoint_every "$CHECKPOINT_EVERY" \
        --H "$H" \
        --T "$T" \
        --D "$D" \
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
        --batch_ablation_values "2048,4096,8192" \
        --clip_ablation_values "0.1,0.01,0.001" \
        --winsorize_ablation_values "95,90,80" \
        --warmup_epochs 200 \
        --orth_init \
        --save_final_envelope \
        --save_checkpoint_ccdf
    echo "=== Experiment 2 DONE ==="
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
