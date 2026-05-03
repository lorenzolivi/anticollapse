#!/usr/bin/env bash
#
# Launch the strengthened drift-validation pipeline on the DGX.
#
# Usage (run from the diagnostics/ folder):
#     ./run_drift_dgx.sh smoke   # short smoke test, 1 seed, small models
#     ./run_drift_dgx.sh full    # full validation, stressed task, GRU + DiagGate
#
# Both modes:
#   - Write all outputs under diagnostics/<outdir>/
#   - Use --device auto (resolves to CUDA on the DGX)
#   - Never copy figures into latex/figures; all artifacts stay in the outdir
#   - Run in the background via nohup and store the PID in <outdir>.pid
#
# The full mode is intended to replace the previous GRU-only drift smoke test.
# It validates the far-left-tail closure model-by-model, using:
#   - GRU + DiagGate by default;
#   - heavier task-side lag pressure than the main experiments;
#   - dense checkpointing for zeta-transition estimates;
#   - multi-projection alpha estimation (K=16 by default).
#
# Environment overrides:
#   OUTDIR, MODELS, SEEDS, DEVICE
#   H, T, EPOCHS, CHECKPOINT_EVERY, NSEQ_TRAIN, NSEQ_DIAG
#   TASK_ALPHA, TASK_K, TASK_LAG_MIN, TASK_LAG_MAX
#   ALPHA_N_DIRECTIONS, ALPHA_N_GRAD_BATCHES, ALPHA_GRAD_BATCH_SIZE
#   TAIL_Q_LOW_SWEEP, TAIL_Q_LOW_PRIMARY, TAIL_BOOTSTRAP_B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
DEVICE="${DEVICE:-auto}"

case "$MODE" in
    smoke)
        OUTDIR="${OUTDIR:-drift_smoketest_v2}"
        MODELS="${MODELS:-diag,gru}"
        SEEDS="${SEEDS:-42}"
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
        TAIL_BOOTSTRAP_B="${TAIL_BOOTSTRAP_B:-500}"
        ;;
    full)
        OUTDIR="${OUTDIR:-drift_validation_v2}"
        MODELS="${MODELS:-diag,gru}"
        SEEDS="${SEEDS:-42,123,321,456,789}"
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
        TAIL_BOOTSTRAP_B="${TAIL_BOOTSTRAP_B:-3000}"
        ;;
    *)
        echo "Usage: $0 {smoke|full}" >&2
        exit 2
        ;;
esac

N_BINS="${N_BINS:-24}"
LATE_FRACTION="${LATE_FRACTION:-0.35}"
MIN_LATE_CHECKPOINTS="${MIN_LATE_CHECKPOINTS:-4}"
TAIL_Q_LOW_PRIMARY="${TAIL_Q_LOW_PRIMARY:-0.10}"
TAIL_Q_LOW_SWEEP="${TAIL_Q_LOW_SWEEP:-0.03,0.05,0.10,0.15,0.20}"
TAIL_TRIM_FRACTION="${TAIL_TRIM_FRACTION:-0.1}"
TAIL_CI_LEVEL="${TAIL_CI_LEVEL:-0.90}"
TAU_FIT_LAG_MIN="${TAU_FIT_LAG_MIN:-64}"
TAU_FIT_LAG_MAX="${TAU_FIT_LAG_MAX:-384}"
TAU_FIT_NUM_LAGS="${TAU_FIT_NUM_LAGS:-32}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

LOGFILE="${OUTDIR}.log"
PIDFILE="${OUTDIR}.pid"
IFS=',' read -r -a MODEL_ARRAY <<< "$MODELS"

# Safety: refuse to clobber a still-running instance.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A ${MODE} run is already in progress (PID $(cat "$PIDFILE"))." >&2
    echo "Kill it first or remove ${PIDFILE}." >&2
    exit 1
fi

# Safety: refuse to overwrite existing non-empty per-model outputs, while still
# allowing a failed model to be retried after deleting only that model subdir.
for MODEL in "${MODEL_ARRAY[@]}"; do
    MODEL="$(echo "$MODEL" | xargs)"
    if [[ -z "$MODEL" ]]; then
        continue
    fi
    MODEL_OUTDIR="${OUTDIR}/${MODEL}"
    if [[ -d "$MODEL_OUTDIR" && -n "$(ls -A "$MODEL_OUTDIR" 2>/dev/null)" ]]; then
        echo "Output directory '${MODEL_OUTDIR}' already exists and is not empty." >&2
        echo "Move/delete that model subdir, or change OUTDIR/MODELS before re-running." >&2
        exit 1
    fi
done

echo "Launching ${MODE} run:"
echo "  outdir : diagnostics/${OUTDIR}/"
echo "  log    : diagnostics/${LOGFILE}"
echo "  pidfile: diagnostics/${PIDFILE}"
echo "  models : ${MODELS}"
echo "  seeds  : ${SEEDS}"
echo "  task   : alpha=${TASK_ALPHA}, K=${TASK_K}, lags=[${TASK_LAG_MIN}, ${TASK_LAG_MAX}]"
echo "  alpha  : K=${ALPHA_N_DIRECTIONS}, batches=${ALPHA_N_GRAD_BATCHES}"

nohup bash -c '
    set -euo pipefail
    cd "$0"
    for MODEL in "$@"; do
        MODEL="$(echo "$MODEL" | xargs)"
        if [[ -z "$MODEL" ]]; then
            continue
        fi
        echo "============================================================"
        echo "Drift validation model=${MODEL}"
        echo "============================================================"
        python run_drift_validation.py \
            --outdir "'"$OUTDIR"'/${MODEL}" \
            --model "${MODEL}" \
            --seeds "'"$SEEDS"'" \
            --H "'"$H"'" \
            --T "'"$T"'" \
            --Nseq_train "'"$NSEQ_TRAIN"'" \
            --Nseq_diag "'"$NSEQ_DIAG"'" \
            --epochs "'"$EPOCHS"'" \
            --batch_size "'"$BATCH_SIZE"'" \
            --lr "'"$LR"'" \
            --weight_decay "'"$WEIGHT_DECAY"'" \
            --grad_clip "'"$GRAD_CLIP"'" \
            --checkpoint_every "'"$CHECKPOINT_EVERY"'" \
            --task_alpha "'"$TASK_ALPHA"'" \
            --task_lag_min "'"$TASK_LAG_MIN"'" \
            --task_lag_max "'"$TASK_LAG_MAX"'" \
            --task_K "'"$TASK_K"'" \
            --alpha_n_directions "'"$ALPHA_N_DIRECTIONS"'" \
            --alpha_n_grad_batches_ckpt "'"$ALPHA_N_GRAD_BATCHES"'" \
            --alpha_grad_batch_size "'"$ALPHA_GRAD_BATCH_SIZE"'" \
            --tau_fit_lag_min "'"$TAU_FIT_LAG_MIN"'" \
            --tau_fit_lag_max "'"$TAU_FIT_LAG_MAX"'" \
            --tau_fit_num_lags "'"$TAU_FIT_NUM_LAGS"'" \
            --late_fraction "'"$LATE_FRACTION"'" \
            --min_late_checkpoints "'"$MIN_LATE_CHECKPOINTS"'" \
            --n_bins "'"$N_BINS"'" \
            --tail_q_low_primary "'"$TAIL_Q_LOW_PRIMARY"'" \
            --tail_q_low_sweep "'"$TAIL_Q_LOW_SWEEP"'" \
            --tail_trim_fraction "'"$TAIL_TRIM_FRACTION"'" \
            --tail_bootstrap_B "'"$TAIL_BOOTSTRAP_B"'" \
            --tail_ci_level "'"$TAIL_CI_LEVEL"'" \
            --device "'"$DEVICE"'"
    done
' "$SCRIPT_DIR" "${MODEL_ARRAY[@]}" > "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "Started PID ${PID}. Tail the log with:"
echo "  tail -f diagnostics/${LOGFILE}"
