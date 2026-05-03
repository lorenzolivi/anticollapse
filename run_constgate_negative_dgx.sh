#!/usr/bin/env bash
#
# Launch the ConstGate structural-negative-control experiment.
#
# Purpose:
#   This is main empirical evidence, not drift-diagnostic validation.
#   ConstGate + AdamW + heavy-tailed-lag is run at the same scale as the
#   positive controls to test the route-versus-realizability distinction:
#   forcing and drift summaries remain measurable, but the achievable
#   time-scale spectrum is expected to stay too narrow to support a
#   power-law envelope.
#
# Usage (may be run from any folder):
#     ./run_constgate_negative_dgx.sh smoke   # 1 seed, small scale
#     ./run_constgate_negative_dgx.sh full    # 5 seeds, full scale
#
# Outputs:
#   - All results are written under results/<outdir>/.
#   - Logs and PID files are written next to the result directory:
#       results/<outdir>.log
#       results/<outdir>.pid
#
# Environment overrides:
#   OUTDIR, SEEDS, DEVICE
#   H, T, EPOCHS, CHECKPOINT_EVERY, NSEQ_TRAIN, NSEQ_DIAG
#   TASK_ALPHA, TASK_K, TASK_LAG_MIN, TASK_LAG_MAX
#   ALPHA_N_DIRECTIONS, ALPHA_N_GRAD_BATCHES, ALPHA_GRAD_BATCH_SIZE

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_EXP1="$PROJECT_ROOT/main_exp1.py"
RESULTS_DIR="$PROJECT_ROOT/results"
cd "$PROJECT_ROOT"

MODE="${1:-}"
DEVICE="${DEVICE:-auto}"

case "$MODE" in
    smoke)
        OUTDIR="${OUTDIR:-constgate_negative_smoketest_v1}"
        SEEDS="${SEEDS:-1873}"
        H="${H:-256}"
        T="${T:-512}"
        EPOCHS="${EPOCHS:-200}"
        CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
        NSEQ_TRAIN="${NSEQ_TRAIN:-2000}"
        NSEQ_DIAG="${NSEQ_DIAG:-1000}"
        TASK_ALPHA="${TASK_ALPHA:-0.8}"
        TASK_K="${TASK_K:-10}"
        TASK_LAG_MIN="${TASK_LAG_MIN:-8}"
        TASK_LAG_MAX="${TASK_LAG_MAX:-256}"
        ALPHA_N_DIRECTIONS="${ALPHA_N_DIRECTIONS:-4}"
        ALPHA_N_GRAD_BATCHES="${ALPHA_N_GRAD_BATCHES:-64}"
        ALPHA_GRAD_BATCH_SIZE="${ALPHA_GRAD_BATCH_SIZE:-128}"
        ;;
    full)
        OUTDIR="${OUTDIR:-constgate_negative_v1}"
        # Distinct from the default drift-validation seed set.
        SEEDS="${SEEDS:-1873,3041,5119,6427,8369}"
        H="${H:-512}"
        T="${T:-1280}"
        EPOCHS="${EPOCHS:-1400}"
        CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
        NSEQ_TRAIN="${NSEQ_TRAIN:-8000}"
        NSEQ_DIAG="${NSEQ_DIAG:-6000}"
        TASK_ALPHA="${TASK_ALPHA:-0.6}"
        TASK_K="${TASK_K:-16}"
        TASK_LAG_MIN="${TASK_LAG_MIN:-8}"
        TASK_LAG_MAX="${TASK_LAG_MAX:-640}"
        ALPHA_N_DIRECTIONS="${ALPHA_N_DIRECTIONS:-16}"
        ALPHA_N_GRAD_BATCHES="${ALPHA_N_GRAD_BATCHES:-128}"
        ALPHA_GRAD_BATCH_SIZE="${ALPHA_GRAD_BATCH_SIZE:-256}"
        ;;
    *)
        echo "Usage: $0 {smoke|full}" >&2
        exit 2
        ;;
esac

OPTIMIZER="${OPTIMIZER:-adamw}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TAU_FIT_LAG_MIN="${TAU_FIT_LAG_MIN:-64}"
TAU_FIT_LAG_MAX="${TAU_FIT_LAG_MAX:-384}"
TAU_FIT_NUM_LAGS="${TAU_FIT_NUM_LAGS:-32}"

mkdir -p "$RESULTS_DIR"
if [[ "$OUTDIR" = /* ]]; then
    RUN_DIR="$OUTDIR"
else
    RUN_DIR="$RESULTS_DIR/$OUTDIR"
fi
LOGFILE="${RUN_DIR}.log"
PIDFILE="${RUN_DIR}.pid"

# Safety: refuse to clobber a still-running instance.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A ${MODE} ConstGate-negative run is already in progress (PID $(cat "$PIDFILE"))." >&2
    echo "Kill it first or remove ${PIDFILE}." >&2
    exit 1
fi

# Safety: refuse to overwrite an existing non-empty output directory.
if [[ -d "$RUN_DIR" && -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
    echo "Output directory '${RUN_DIR}' already exists and is not empty." >&2
    echo "Move or delete it before re-running, to keep results reproducible." >&2
    exit 1
fi

if [[ ! -f "$MAIN_EXP1" ]]; then
    echo "Could not find main_exp1.py at $MAIN_EXP1" >&2
    exit 1
fi

echo "Launching ${MODE} ConstGate-negative experiment:"
echo "  outdir : ${RUN_DIR}/"
echo "  log    : ${LOGFILE}"
echo "  pidfile: ${PIDFILE}"
echo "  models : const  (structural negative control)"
echo "  seeds  : ${SEEDS}"
echo "  task   : alpha=${TASK_ALPHA}, K=${TASK_K}, lags=[${TASK_LAG_MIN}, ${TASK_LAG_MAX}]"
echo "  alpha  : K=${ALPHA_N_DIRECTIONS}, batches=${ALPHA_N_GRAD_BATCHES}"
echo "  H=${H}, T=${T}, epochs=${EPOCHS}, checkpoint_every=${CHECKPOINT_EVERY}"

nohup python "$MAIN_EXP1" \
    --outdir "$RUN_DIR" \
    --models "const" \
    --seeds "$SEEDS" \
    --H "$H" \
    --T "$T" \
    --Nseq_train "$NSEQ_TRAIN" \
    --Nseq_diag "$NSEQ_DIAG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip "$GRAD_CLIP" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --optimizer "$OPTIMIZER" \
    --task_variant "heavy_tail" \
    --task_alpha "$TASK_ALPHA" \
    --task_lag_min "$TASK_LAG_MIN" \
    --task_lag_max "$TASK_LAG_MAX" \
    --task_K "$TASK_K" \
    --alpha_n_directions "$ALPHA_N_DIRECTIONS" \
    --alpha_n_grad_batches_ckpt "$ALPHA_N_GRAD_BATCHES" \
    --alpha_grad_batch_size "$ALPHA_GRAD_BATCH_SIZE" \
    --tau_fit_lag_min "$TAU_FIT_LAG_MIN" \
    --tau_fit_lag_max "$TAU_FIT_LAG_MAX" \
    --tau_fit_num_lags "$TAU_FIT_NUM_LAGS" \
    --save_checkpoint_ccdf \
    --save_final_envelope \
    --device "$DEVICE" \
    --skip_plot \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "Started PID ${PID}. Tail the log with:"
echo "  tail -f ${LOGFILE}"
echo
echo "Expected deliverable for the manuscript matrix:"
echo "  ${RUN_DIR}/${OPTIMIZER}/aggregated/const/phase_trajectory.csv  (envelope class, beta, alpha)"
echo "  ${RUN_DIR}/${OPTIMIZER}/aggregated/const/const_envelope_fit.json (final envelope class)"
echo "  ${RUN_DIR}/${OPTIMIZER}/aggregated/const/checkpoint_tau_ccdf/    (checkpoint CCDF audit)"
echo "  ${RUN_DIR}/${OPTIMIZER}/seed_*/const/                            (per-seed checkpoint dumps)"
echo "  envelope class: expected exponential"
echo "  alpha_hat:      measured forcing proxy; may or may not be heavy-tailed"
echo "  zeta range:     expected narrow (capacity bounded by fixed gates)"
