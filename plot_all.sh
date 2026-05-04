#!/usr/bin/env bash
#
# Root plotting / analysis launcher for Anti-Collapse main-text experiments.
#
# This is the plotting-side counterpart to anticollapse.sh:
#   anticollapse.sh -> launch simulations
#   plot_all.sh     -> re-aggregate and plot existing results
#
# Usage:
#   ./plot_all.sh exp1 [smoke|full]
#   ./plot_all.sh exp2 [smoke|full]
#   ./plot_all.sh exp3 [smoke|full]
#   ./plot_all.sh all  [smoke|full]
#
# The optional profile defaults to "full" and selects the default result
# directories created by anticollapse.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${OUTDIR:-$SCRIPT_DIR/results}}"
DPI="${DPI:-600}"

COMMAND="${1:-}"
PROFILE="${2:-${PROFILE:-full}}"

usage() {
    cat <<'EOF'
plot_all.sh - plotting / analysis launcher for Anti-Collapse

Usage:
    ./plot_all.sh exp1 [smoke|full]   plot ConstGate negative-control results
    ./plot_all.sh exp2 [smoke|full]   plot phase-trajectory/capacity results
    ./plot_all.sh exp3 [smoke|full]   plot forcing-ablation results
    ./plot_all.sh all  [smoke|full]   plot all three result sets
    ./plot_all.sh --help              show this help

Environment overrides:
    RESULTS_DIR or OUTDIR   root result directory (default: ./results)
    EXP1_OUTDIR             explicit Experiment 1 result directory
    EXP2_OUTDIR             explicit Experiment 2 result directory
    EXP3_OUTDIR             explicit Experiment 3 result directory
    SEEDS                   comma-separated seeds
    EXP2_MODELS             default shared,diag,gru,lstm
    EXP3_MODELS             default diag
    DPI                     default 600

No simulation is run. This script calls main_exp1.py/main_exp2.py with
--plot_only to regenerate aggregations and plots from existing per-seed output.
EOF
    exit 0
}

case "$COMMAND" in
    exp1|exp2|exp3|all|--help|-h|help|"") ;;
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
        SEEDS="${SEEDS:-42}"
        EXP2_MODELS="${EXP2_MODELS:-shared,diag,gru}"
        EXP3_MODELS="${EXP3_MODELS:-diag}"
        EXP3_BATCH_VALUES="${EXP3_BATCH_VALUES:-1024}"
        EXP3_CLIP_VALUES="${EXP3_CLIP_VALUES:-0.1}"
        EXP3_WINSOR_VALUES="${EXP3_WINSOR_VALUES:-95}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
        ;;
    full)
        SEEDS="${SEEDS:-42,123,321,456,789}"
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
EXP1_OUTDIR="${EXP1_OUTDIR:-$RESULTS_DIR/exp1_constgate_${PROFILE}}"
EXP2_OUTDIR="${EXP2_OUTDIR:-$RESULTS_DIR/exp2_phase_${PROFILE}}"
EXP3_OUTDIR="${EXP3_OUTDIR:-$RESULTS_DIR/exp3_forcing_${PROFILE}}"

EXP3_CONDITIONS="${EXP3_CONDITIONS:-baseline,batch_ablation,clip_ablation,winsorize_ablation}"

check_dir() {
    local dir="$1"
    local label="$2"
    if [[ ! -d "$dir" ]]; then
        echo "Missing ${label} result directory: ${dir}" >&2
        echo "Run the corresponding experiment first or set EXP*_OUTDIR." >&2
        exit 1
    fi
}

plot_exp1() {
    check_dir "$EXP1_OUTDIR" "Experiment 1"
    echo "============================================================"
    echo "Plotting Experiment 1: ConstGate structural negative control"
    echo "============================================================"
    echo "outdir: $EXP1_OUTDIR"
    python "$SCRIPT_DIR/main_exp1.py" \
        --outdir "$EXP1_OUTDIR" \
        --seeds "$SEEDS" \
        --models const \
        --optimizer "$OPTIMIZER" \
        --dpi "$DPI" \
        --plot_only
}

plot_exp2() {
    check_dir "$EXP2_OUTDIR" "Experiment 2"
    echo "============================================================"
    echo "Plotting Experiment 2: dynamical phase trajectory"
    echo "============================================================"
    echo "models: $EXP2_MODELS"
    echo "outdir: $EXP2_OUTDIR"
    python "$SCRIPT_DIR/main_exp1.py" \
        --outdir "$EXP2_OUTDIR" \
        --seeds "$SEEDS" \
        --models "$EXP2_MODELS" \
        --optimizer "$OPTIMIZER" \
        --dpi "$DPI" \
        --plot_only
}

plot_exp3() {
    check_dir "$EXP3_OUTDIR" "Experiment 3"
    echo "============================================================"
    echo "Plotting Experiment 3: stochastic-forcing ablation"
    echo "============================================================"
    echo "models: $EXP3_MODELS"
    echo "outdir: $EXP3_OUTDIR"
    python "$SCRIPT_DIR/main_exp2.py" \
        --outdir "$EXP3_OUTDIR" \
        --seeds "$SEEDS" \
        --models "$EXP3_MODELS" \
        --optimizer "$OPTIMIZER" \
        --conditions "$EXP3_CONDITIONS" \
        --batch_ablation_values "$EXP3_BATCH_VALUES" \
        --clip_ablation_values "$EXP3_CLIP_VALUES" \
        --winsorize_ablation_values "$EXP3_WINSOR_VALUES" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --dpi "$DPI" \
        --plot_only
}

case "$COMMAND" in
    exp1)
        plot_exp1
        ;;
    exp2)
        plot_exp2
        ;;
    exp3)
        plot_exp3
        ;;
    all)
        plot_exp1
        plot_exp2
        plot_exp3
        ;;
esac
