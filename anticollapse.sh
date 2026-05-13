#!/usr/bin/env bash
#
# Main-text experiment launcher for the Anti-Collapse paper.
#
# The manuscript now has three empirical experiments:
#   exp1: ConstGate structural negative control
#   exp2: dynamical phase trajectory across the capacity ladder
#   exp3: stochastic-forcing ablation
#
# Usage:
#   ./anticollapse.sh exp1 [smoke|full]
#   ./anticollapse.sh exp2 [smoke|full]
#   ./anticollapse.sh exp3 [smoke|full]
#   ./anticollapse.sh all  [smoke|full]
#
# The optional profile defaults to "full".  The full profile is sequential
# and DGX-Spark-safe in memory, but it is intentionally long-running.
# Use EXP2_MODELS / EXP3_MODELS to split publication runs into smaller jobs.
# Runs launch in the background by default, with .log and .pid files next to
# the output directory. Set FOREGROUND=1 for an interactive foreground run.
#
# Output:
#   results/exp1_constgate_<profile>/
#   results/exp2_phase_<profile>/
#   results/exp3_forcing_<profile>/
#
# Common environment overrides:
#   RESULTS_DIR or OUTDIR, DEVICE, SEEDS, H, T, EPOCHS, CHECKPOINT_EVERY
#   NSEQ_TRAIN, NSEQ_DIAG, TASK_ALPHA, TASK_K, TASK_LAG_MIN, TASK_LAG_MAX
#   ALPHA_N_DIRECTIONS, ALPHA_N_GRAD_BATCHES, ALPHA_GRAD_BATCH_SIZE
#   EXP2_MODELS, EXP3_MODELS, SKIP_PLOT, FOREGROUND

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${OUTDIR:-$SCRIPT_DIR/results}}"
DEVICE="${DEVICE:-auto}"
DPI="${DPI:-600}"

COMMAND="${1:-}"
PROFILE="${2:-${PROFILE:-full}}"

usage() {
    cat <<'EOF'
anticollapse.sh - main-text Anti-Collapse experiment launcher

Usage:
    ./anticollapse.sh exp1        [smoke|full]   ConstGate structural negative control
    ./anticollapse.sh exp1_inject [smoke|full]   Path A: ConstGate + α-stable forcing injection
    ./anticollapse.sh exp2        [smoke|full]   phase trajectory / capacity ladder
    ./anticollapse.sh exp3        [smoke|full]   stochastic-forcing ablation
    ./anticollapse.sh all         [smoke|full]   run exp1, exp2, exp3 sequentially
    ./anticollapse.sh --help                     show this help

Examples:
    ./anticollapse.sh exp1 smoke
    ./anticollapse.sh exp1_inject smoke
    INJECT_ALPHA=1.4 INJECT_ALPHA_NOISE=2.0 ./anticollapse.sh exp1_inject full
    ./anticollapse.sh exp2 full
    EXP2_MODELS=diag,gru ./anticollapse.sh exp2 full
    EXP3_MODELS=diag EXP3_CONDITIONS=baseline,clip_ablation ./anticollapse.sh exp3 full

Profiles:
    smoke: 1 seed, small model/task, pipeline integrity only.
    full:  5 seeds, H=512, stressed heavy-tailed-lag task, publication scale.

Outputs default to results/ under the project root. Set RESULTS_DIR or OUTDIR
to redirect all experiment folders.

Runs launch simulations in the background by default and create .log/.pid files
next to the output directory. Final envelope/regime analysis is deferred to
plot_all.sh. Set FOREGROUND=1 for an interactive foreground run.
EOF
    exit 0
}

case "$COMMAND" in
    exp1|exp1_inject|exp2|exp3|all|--help|-h|help|"") ;;
    *)
        echo "Error: unknown command '$COMMAND'." >&2
        usage
        ;;
esac

if [[ "$COMMAND" == "--help" || "$COMMAND" == "-h" || "$COMMAND" == "help" || -z "$COMMAND" ]]; then
    usage
fi

case "$PROFILE" in
    smoke)
        SEEDS="${SEEDS:-47}"
        H="${H:-256}"
        T="${T:-512}"
        D="${D:-16}"
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
        BETA_BOOTSTRAP_B="${BETA_BOOTSTRAP_B:-500}"
        EXP2_MODELS="${EXP2_MODELS:-shared,diag,gru}"
        EXP3_MODELS="${EXP3_MODELS:-diag}"
        EXP3_BATCH_VALUES="${EXP3_BATCH_VALUES:-1024}"
        EXP3_CLIP_VALUES="${EXP3_CLIP_VALUES:-0.1}"
        EXP3_WINSOR_VALUES="${EXP3_WINSOR_VALUES:-95}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
        ;;
    full)
        SEEDS="${SEEDS:-47,83,12,69,31}"
        H="${H:-512}"
        T="${T:-1280}"
        D="${D:-16}"
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
        BETA_BOOTSTRAP_B="${BETA_BOOTSTRAP_B:-2000}"
        EXP2_MODELS="${EXP2_MODELS:-shared,diag,gru,lstm}"
        EXP3_MODELS="${EXP3_MODELS:-diag}"
        EXP3_BATCH_VALUES="${EXP3_BATCH_VALUES:-2048,4096,8192}"
        EXP3_CLIP_VALUES="${EXP3_CLIP_VALUES:-0.1,0.01,0.001}"
        EXP3_WINSOR_VALUES="${EXP3_WINSOR_VALUES:-95,90,80}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-200}"
        ;;
    *)
        echo "Error: profile must be smoke or full, got '$PROFILE'." >&2
        exit 2
        ;;
esac

OPTIMIZER="${OPTIMIZER:-adamw}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-256}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MOMENTUM="${MOMENTUM:-0.9}"
TAU_FIT_LAG_MIN="${TAU_FIT_LAG_MIN:-64}"
TAU_FIT_LAG_MAX="${TAU_FIT_LAG_MAX:-384}"
TAU_FIT_NUM_LAGS="${TAU_FIT_NUM_LAGS:-32}"
TAU_CCDF_QMIN="${TAU_CCDF_QMIN:-0.75}"
TAU_CCDF_QMAX="${TAU_CCDF_QMAX:-0.995}"
LAG_MIN="${LAG_MIN:-4}"
LAG_MAX="${LAG_MAX:-384}"
NUM_LAGS="${NUM_LAGS:-128}"
BETA_BOOTSTRAP_CI="${BETA_BOOTSTRAP_CI:-0.90}"
PHASE_R2_THRESHOLD="${PHASE_R2_THRESHOLD:-0.90}"
POWER_WINDOW_BETA_MIN="${POWER_WINDOW_BETA_MIN:-0.10}"
POWER_WINDOW_MIN_POINTS="${POWER_WINDOW_MIN_POINTS:-8}"
POWER_WINDOW_MIN_FRACTION="${POWER_WINDOW_MIN_FRACTION:-0.05}"
TASK_COEFF_BASE="${TASK_COEFF_BASE:-0.6}"
TASK_COEFF_DECAY="${TASK_COEFF_DECAY:-0.85}"
TASK_LAG_SEED="${TASK_LAG_SEED:-20260410}"
EXP3_CONDITIONS="${EXP3_CONDITIONS:-baseline,batch_ablation,clip_ablation,winsorize_ablation}"
SAVE_MODEL_CHECKPOINTS="${SAVE_MODEL_CHECKPOINTS:-0}"
SKIP_PLOT="${SKIP_PLOT:-1}"

EXP1_OUTDIR="${EXP1_OUTDIR:-$RESULTS_DIR/exp1_constgate_${PROFILE}}"
EXP1_INJECT_OUTDIR="${EXP1_INJECT_OUTDIR:-$RESULTS_DIR/exp1_constgate_inject_${PROFILE}}"
EXP2_OUTDIR="${EXP2_OUTDIR:-$RESULTS_DIR/exp2_phase_${PROFILE}}"
EXP3_OUTDIR="${EXP3_OUTDIR:-$RESULTS_DIR/exp3_forcing_${PROFILE}}"
ALL_RUN_STEM="${ALL_RUN_STEM:-$RESULTS_DIR/anticollapse_all_${PROFILE}}"

# Heavy-tailed gradient-noise injection (Path A) settings. Used only when
# COMMAND == exp1_inject. INJECT_ALPHA_NOISE=0 disables (no-op); >0 enables.
INJECT_ALPHA_NOISE="${INJECT_ALPHA_NOISE:-1.0}"
INJECT_ALPHA="${INJECT_ALPHA:-1.6}"
INJECT_GRAD_SEED_OFFSET="${INJECT_GRAD_SEED_OFFSET:-1729}"

mkdir -p "$RESULTS_DIR"

common_exp_args=(
    --seeds "$SEEDS"
    --H "$H"
    --T "$T"
    --D "$D"
    --Nseq_train "$NSEQ_TRAIN"
    --Nseq_diag "$NSEQ_DIAG"
    --epochs "$EPOCHS"
    --checkpoint_every "$CHECKPOINT_EVERY"
    --batch_size "$BATCH_SIZE"
    --diag_batch_size "$DIAG_BATCH_SIZE"
    --optimizer "$OPTIMIZER"
    --momentum "$MOMENTUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --grad_clip "$GRAD_CLIP"
    --task_variant heavy_tail
    --task_alpha "$TASK_ALPHA"
    --task_lag_min "$TASK_LAG_MIN"
    --task_lag_max "$TASK_LAG_MAX"
    --task_K "$TASK_K"
    --task_coeff_base "$TASK_COEFF_BASE"
    --task_coeff_decay "$TASK_COEFF_DECAY"
    --task_lag_seed "$TASK_LAG_SEED"
    --alpha_n_grad_batches_ckpt "$ALPHA_N_GRAD_BATCHES"
    --alpha_grad_batch_size "$ALPHA_GRAD_BATCH_SIZE"
    --alpha_n_directions "$ALPHA_N_DIRECTIONS"
    --tau_fit_lag_min "$TAU_FIT_LAG_MIN"
    --tau_fit_lag_max "$TAU_FIT_LAG_MAX"
    --tau_fit_num_lags "$TAU_FIT_NUM_LAGS"
    --tau_ccdf_qmin "$TAU_CCDF_QMIN"
    --tau_ccdf_qmax "$TAU_CCDF_QMAX"
    --lag_min "$LAG_MIN"
    --lag_max "$LAG_MAX"
    --num_lags "$NUM_LAGS"
    --beta_bootstrap_B "$BETA_BOOTSTRAP_B"
    --beta_bootstrap_ci "$BETA_BOOTSTRAP_CI"
    --phase_r2_threshold "$PHASE_R2_THRESHOLD"
    --power_window_beta_min "$POWER_WINDOW_BETA_MIN"
    --power_window_min_points "$POWER_WINDOW_MIN_POINTS"
    --power_window_min_fraction "$POWER_WINDOW_MIN_FRACTION"
    --device "$DEVICE"
    --dpi "$DPI"
    --orth_init
    --save_checkpoint_ccdf
    --save_analysis_checkpoint
)

if [[ "$SKIP_PLOT" == "1" || "$SKIP_PLOT" == "true" ]]; then
    common_exp_args+=(--skip_plot)
fi

is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

launcher_artifacts() {
    case "$COMMAND" in
        exp1)
            RUN_LABEL="Experiment 1: ConstGate structural negative control"
            RUN_LOGFILE="${EXP1_OUTDIR}.log"
            RUN_PIDFILE="${EXP1_OUTDIR}.pid"
            TARGET_DIRS=("$EXP1_OUTDIR")
            ;;
        exp1_inject)
            RUN_LABEL="Experiment 1 (Path A): ConstGate with α-stable forcing injection"
            RUN_LOGFILE="${EXP1_INJECT_OUTDIR}.log"
            RUN_PIDFILE="${EXP1_INJECT_OUTDIR}.pid"
            TARGET_DIRS=("$EXP1_INJECT_OUTDIR")
            ;;
        exp2)
            RUN_LABEL="Experiment 2: dynamical phase trajectory"
            RUN_LOGFILE="${EXP2_OUTDIR}.log"
            RUN_PIDFILE="${EXP2_OUTDIR}.pid"
            TARGET_DIRS=("$EXP2_OUTDIR")
            ;;
        exp3)
            RUN_LABEL="Experiment 3: stochastic-forcing ablation"
            RUN_LOGFILE="${EXP3_OUTDIR}.log"
            RUN_PIDFILE="${EXP3_OUTDIR}.pid"
            TARGET_DIRS=("$EXP3_OUTDIR")
            ;;
        all)
            RUN_LABEL="Experiments 1-3: complete empirical campaign"
            RUN_LOGFILE="${ALL_RUN_STEM}.log"
            RUN_PIDFILE="${ALL_RUN_STEM}.pid"
            TARGET_DIRS=("$EXP1_OUTDIR" "$EXP2_OUTDIR" "$EXP3_OUTDIR")
            ;;
    esac
}

launcher_artifacts

if [[ "${ANTICOLLAPSE_INTERNAL_RUN:-0}" != "1" ]] && ! is_true "${FOREGROUND:-0}"; then
    if [[ -f "$RUN_PIDFILE" ]] && kill -0 "$(cat "$RUN_PIDFILE")" 2>/dev/null; then
        echo "A ${COMMAND}/${PROFILE} run is already in progress (PID $(cat "$RUN_PIDFILE"))." >&2
        echo "Kill it first or remove ${RUN_PIDFILE}." >&2
        exit 1
    fi

    for TARGET_DIR in "${TARGET_DIRS[@]}"; do
        if [[ -d "$TARGET_DIR" && -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
            echo "Output directory '${TARGET_DIR}' already exists and is not empty." >&2
            echo "Move/delete it, or choose a different OUTDIR/EXP*_OUTDIR before re-running." >&2
            exit 1
        fi
    done

    mkdir -p "$(dirname "$RUN_LOGFILE")" "$(dirname "$RUN_PIDFILE")"

    echo "Launching ${RUN_LABEL}"
    echo "  profile: ${PROFILE}"
    echo "  log    : ${RUN_LOGFILE}"
    echo "  pidfile: ${RUN_PIDFILE}"
    echo "  tail   : tail -f ${RUN_LOGFILE}"
    echo "  stop   : kill \$(cat ${RUN_PIDFILE})"

    nohup env ANTICOLLAPSE_INTERNAL_RUN=1 "$SCRIPT_DIR/anticollapse.sh" "$COMMAND" "$PROFILE" \
        > "$RUN_LOGFILE" 2>&1 &

    PID=$!
    echo "$PID" > "$RUN_PIDFILE"
    echo "Started PID ${PID}."
    exit 0
fi

run_exp1() {
    echo "============================================================"
    echo "Experiment 1: ConstGate structural negative control"
    echo "============================================================"
    echo "outdir: $EXP1_OUTDIR"
    run_python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP1_OUTDIR" \
        --models const \
        "${common_exp_args[@]}"
}

run_exp1_inject() {
    echo "============================================================"
    echo "Experiment 1 (Path A): ConstGate with α-stable forcing injection"
    echo "============================================================"
    echo "outdir:               $EXP1_INJECT_OUTDIR"
    echo "inject_alpha_noise:   $INJECT_ALPHA_NOISE"
    echo "inject_alpha:         $INJECT_ALPHA"
    echo "inject_grad_seed_off: $INJECT_GRAD_SEED_OFFSET"
    run_python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP1_INJECT_OUTDIR" \
        --models const \
        --inject_alpha_noise "$INJECT_ALPHA_NOISE" \
        --inject_alpha "$INJECT_ALPHA" \
        --inject_grad_seed_offset "$INJECT_GRAD_SEED_OFFSET" \
        "${common_exp_args[@]}"
}

run_exp2() {
    echo "============================================================"
    echo "Experiment 2: dynamical phase trajectory"
    echo "============================================================"
    echo "models: $EXP2_MODELS"
    echo "outdir: $EXP2_OUTDIR"
    exp2_args=("${common_exp_args[@]}")
    if [[ "$SAVE_MODEL_CHECKPOINTS" == "1" || "$SAVE_MODEL_CHECKPOINTS" == "true" ]]; then
        exp2_args+=(--save_model_checkpoints)
    fi
    run_python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP2_OUTDIR" \
        --models "$EXP2_MODELS" \
        "${exp2_args[@]}"
}

run_exp3() {
    echo "============================================================"
    echo "Experiment 3: stochastic-forcing ablation"
    echo "============================================================"
    echo "models: $EXP3_MODELS"
    echo "conditions: $EXP3_CONDITIONS"
    echo "outdir: $EXP3_OUTDIR"
    run_python "$SCRIPT_DIR/main_forcing_ablation.py" \
        --outdir "$EXP3_OUTDIR" \
        --models "$EXP3_MODELS" \
        --conditions "$EXP3_CONDITIONS" \
        --batch_ablation_values "$EXP3_BATCH_VALUES" \
        --clip_ablation_values "$EXP3_CLIP_VALUES" \
        --winsorize_ablation_values "$EXP3_WINSOR_VALUES" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        "${common_exp_args[@]}"
}

ACTIVE_CHILD=""

terminate_active_child() {
    if [[ -n "${ACTIVE_CHILD:-}" ]] && kill -0 "$ACTIVE_CHILD" 2>/dev/null; then
        kill "$ACTIVE_CHILD" 2>/dev/null || true
    fi
    exit 130
}

trap terminate_active_child TERM INT

run_python() {
    if [[ "$COMMAND" == "all" ]]; then
        python "$@" &
        ACTIVE_CHILD=$!
        set +e
        wait "$ACTIVE_CHILD"
        local status=$?
        set -e
        ACTIVE_CHILD=""
        return "$status"
    else
        exec python "$@"
    fi
}

echo "Anti-Collapse empirical launcher"
echo "profile: $PROFILE"
echo "seeds:   $SEEDS"
echo "task:    heavy_tail alpha=$TASK_ALPHA K=$TASK_K lags=[$TASK_LAG_MIN,$TASK_LAG_MAX]"
echo "scale:   H=$H T=$T epochs=$EPOCHS checkpoint_every=$CHECKPOINT_EVERY"
echo "results: $RESULTS_DIR"

case "$COMMAND" in
    exp1)
        run_exp1
        ;;
    exp1_inject)
        run_exp1_inject
        ;;
    exp2)
        run_exp2
        ;;
    exp3)
        run_exp3
        ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        ;;
esac
