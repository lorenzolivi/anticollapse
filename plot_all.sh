#!/usr/bin/env bash
#
# Root plotting / analysis launcher for Anti-Collapse main-text experiments.
#
# This is the plotting-side counterpart to anticollapse.sh:
#   anticollapse.sh -> launch simulations
#   plot_all.sh     -> run deferred analysis, re-aggregate, and plot existing results
#
# Usage:
#   ./plot_all.sh exp1        [smoke|full]
#   ./plot_all.sh exp1_inject [smoke|full]
#   ./plot_all.sh exp1_figs   [smoke|full]
#   ./plot_all.sh exp2        [smoke|full]
#   ./plot_all.sh exp2_figs   [smoke|full]
#   ./plot_all.sh exp3        [smoke|full]
#   ./plot_all.sh exp3_figs   [smoke|full]
#   ./plot_all.sh all         [smoke|full]
#
# The optional profile defaults to "full" and selects the default result
# directories created by anticollapse.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${OUTDIR:-$SCRIPT_DIR/results}}"
DPI="${DPI:-600}"

# Local matplotlib cache to avoid the per-run "Matplotlib created a temporary
# config/cache directory" warning when running on shared systems.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

COMMAND="${1:-}"
PROFILE="${2:-${PROFILE:-full}}"

usage() {
    cat <<'EOF'
plot_all.sh - plotting / analysis launcher for Anti-Collapse

Usage:
    ./plot_all.sh exp1        [smoke|full]   plot ConstGate negative-control results
    ./plot_all.sh exp1_inject [smoke|full]   plot ConstGate injected-forcing results
    ./plot_all.sh exp1_figs   [smoke|full]   regenerate ONLY the Exp 1 Path A vs Path B
                                             comparison figures (drift, final spectrum)
                                             and mirror per-path envelope plots into
                                             $EXP1_FIG_DIR — skips main_phase_trajectory and drift
    ./plot_all.sh exp2        [smoke|full]   plot phase-trajectory/capacity results
                                             (capacity ladder: shared,diag,gru,lstm)
    ./plot_all.sh exp2_figs   [smoke|full]   regenerate ONLY the Exp 2 capacity-ladder
                                             comparison figures (forcing, beta_env, drift,
                                             final spectrum, Delta zeta,
                                             phase-summary, t_cross) and mirror
                                             per-architecture envelope plots into
                                             $EXP2_FIG_DIR — skips main_phase_trajectory and drift
    ./plot_all.sh exp3        [smoke|full]   plot forcing-ablation results
    ./plot_all.sh exp3_figs   [smoke|full]   regenerate ONLY the Exp 3 ablation
                                             comparison figures and markdown summary
                                             into $EXP3_FIG_DIR — skips main_forcing_ablation
    ./plot_all.sh all         [smoke|full]   plot exp1, exp1_inject, exp2, exp3 result sets
    ./plot_all.sh --help                     show this help

Environment overrides:
    RESULTS_DIR or OUTDIR   root result directory (default: ./results)
    EXP1_OUTDIR             explicit Experiment 1 result directory
    EXP1_INJECT_OUTDIR      explicit Experiment 1 injected-forcing directory
    EXP1_FIG_DIR            Overleaf-bound folder for exp1 figures (default: results/exp1_figures)
    EXP1_FIG_DPI            DPI for exp1 figures (default: 400)
    EXP2_OUTDIR             explicit Experiment 2 result directory
    EXP2_FIG_DIR            Overleaf-bound folder for exp2 figures (default: results/exp2_figures)
    EXP2_FIG_DPI            DPI for exp2 figures (default: 400)
    EXP3_OUTDIR             explicit Experiment 3 result directory
    EXP3_FIG_DIR            Overleaf-bound folder for exp3 figures (default: results/exp3_figures)
    EXP3_FIG_DPI            DPI for exp3 figures (default: 400)
    SEEDS                   comma-separated seeds
    EXP2_MODELS             default shared,diag,gru,lstm
    EXP3_MODELS             default diag
    DPI                     default 600 (general fallback; per-experiment FIG_DPI overrides take precedence)
    SKIP_DRIFT=1            skip the far-left drift diagnostic regardless of profile
    RUN_DRIFT_ON_SMOKE=1    enable drift diagnostic on the smoke profile (off by default)
    SKIP_EXP1_COMPARISON=1  skip the exp1 path-comparison plot block
    SKIP_EXP2_COMPARISON=1  skip the exp2 capacity-ladder comparison plot block
    DRIFT_*                 override drift-script knobs (see body of script)

No simulation is run. This script calls main_phase_trajectory.py/main_forcing_ablation.py with
--plot_only to reload saved analysis checkpoints, compute final envelope /
regime diagnostics, and regenerate aggregations and plots from existing
per-seed output. For exp1 / exp1_inject the far-left drift plateau is also
regenerated as part of the canonical analysis pass (one of the manuscript's
reported observables); coupling it here prevents stale drift
artifacts after re-running the analysis on updated checkpoints.
EOF
    exit 0
}

case "$COMMAND" in
    exp1|exp1_inject|exp1_figs|exp2|exp2_figs|exp3|exp3_figs|all|--help|-h|help|"") ;;
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
        EXP2_MODELS="${EXP2_MODELS:-shared,diag,gru}"
        EXP3_MODELS="${EXP3_MODELS:-diag}"
        EXP3_BATCH_VALUES="${EXP3_BATCH_VALUES:-1024}"
        EXP3_CLIP_VALUES="${EXP3_CLIP_VALUES:-0.1}"
        EXP3_WINSOR_VALUES="${EXP3_WINSOR_VALUES:-95}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
        ;;
    full)
        SEEDS="${SEEDS:-47,83,12,69,31}"
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
EXP1_INJECT_OUTDIR="${EXP1_INJECT_OUTDIR:-$RESULTS_DIR/exp1_constgate_inject_${PROFILE}}"
EXP1_FIG_DIR="${EXP1_FIG_DIR:-$RESULTS_DIR/exp1_figures}"
EXP1_FIG_DPI="${EXP1_FIG_DPI:-400}"
EXP2_OUTDIR="${EXP2_OUTDIR:-$RESULTS_DIR/exp2_phase_${PROFILE}}"
# Exp 2 Overleaf-bound figure folder; all paper-bound exp2 PNGs are
# written or mirrored here so a single folder can be copy-pasted to the
# manuscript repo. 400 dpi is more than enough for the figure types we
# use; override EXP2_FIG_DPI to change.
EXP2_FIG_DIR="${EXP2_FIG_DIR:-$RESULTS_DIR/exp2_figures}"
EXP2_FIG_DPI="${EXP2_FIG_DPI:-400}"
EXP3_OUTDIR="${EXP3_OUTDIR:-$RESULTS_DIR/exp3_forcing_${PROFILE}}"
EXP3_FIG_DIR="${EXP3_FIG_DIR:-$RESULTS_DIR/exp3_figures}"
EXP3_FIG_DPI="${EXP3_FIG_DPI:-400}"

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

# ------------------------------------------------------------
# Far-left drift diagnostic (one of the five common diagnostic
# columns of the empirical campaign). Coupled into plot_all.sh
# so re-running the analysis on updated checkpoints regenerates
# the drift artifacts in lockstep, preventing stale drift output.
#
# Skipped on the smoke profile by default (single-seed bootstrap
# is not statistically informative); set RUN_DRIFT_ON_SMOKE=1 to
# enable it for pipeline-integrity coverage. SKIP_DRIFT=1 is a
# universal escape hatch.
# ------------------------------------------------------------
DRIFT_LATE_FRACTION="${DRIFT_LATE_FRACTION:-0.35}"
DRIFT_MIN_LATE_CHECKPOINTS="${DRIFT_MIN_LATE_CHECKPOINTS:-4}"
DRIFT_N_BINS="${DRIFT_N_BINS:-24}"
DRIFT_TAIL_Q_LOW_PRIMARY="${DRIFT_TAIL_Q_LOW_PRIMARY:-0.10}"
DRIFT_TAIL_Q_LOW_SWEEP="${DRIFT_TAIL_Q_LOW_SWEEP:-0.03,0.05,0.10,0.15,0.20}"
DRIFT_TAIL_TRIM_FRACTION="${DRIFT_TAIL_TRIM_FRACTION:-0.1}"
DRIFT_TAIL_BOOTSTRAP_B="${DRIFT_TAIL_BOOTSTRAP_B:-3000}"
DRIFT_TAIL_CI_LEVEL="${DRIFT_TAIL_CI_LEVEL:-0.90}"

run_drift_diagnostic() {
    local outdir="$1"
    local model="$2"
    local label="$3"

    if [[ "${SKIP_DRIFT:-0}" == "1" || "${SKIP_DRIFT:-}" == "true" ]]; then
        echo "[$label] drift diagnostic: SKIPPED (SKIP_DRIFT=1)"
        return 0
    fi
    if [[ "$PROFILE" == "smoke" && "${RUN_DRIFT_ON_SMOKE:-0}" != "1" ]]; then
        echo "[$label] drift diagnostic: SKIPPED on smoke profile " \
             "(set RUN_DRIFT_ON_SMOKE=1 to override; smoke single-seed bootstrap is not informative)"
        return 0
    fi

    local opt_dir="$outdir/$OPTIMIZER"
    if [[ ! -d "$opt_dir" ]]; then
        echo "[$label] drift diagnostic: no $opt_dir directory; skipping"
        return 0
    fi

    echo "[$label] drift diagnostic on $opt_dir (model=$model)"
    python "$SCRIPT_DIR/diagnostics/run_restoring_drift.py" \
        --input_dir "$opt_dir" \
        --model "$model" \
        --outdir "$opt_dir/drift_${model}" \
        --late_fraction "$DRIFT_LATE_FRACTION" \
        --min_late_checkpoints "$DRIFT_MIN_LATE_CHECKPOINTS" \
        --n_bins "$DRIFT_N_BINS" \
        --tail_q_low_primary "$DRIFT_TAIL_Q_LOW_PRIMARY" \
        --tail_q_low_sweep "$DRIFT_TAIL_Q_LOW_SWEEP" \
        --tail_trim_fraction "$DRIFT_TAIL_TRIM_FRACTION" \
        --tail_bootstrap_B "$DRIFT_TAIL_BOOTSTRAP_B" \
        --tail_ci_level "$DRIFT_TAIL_CI_LEVEL"
}

# ------------------------------------------------------------
# Exp 1 Path A vs Path B comparison plots
# (drift plateau, final time-scale spectrum; forcing / Delta zeta are audit-only).
# Requires BOTH Path B (EXP1_OUTDIR) and Path A (EXP1_INJECT_OUTDIR)
# to be on disk; skips with a clean message if either is missing.
# ------------------------------------------------------------
run_exp1_path_comparison() {
    local label="$1"

    if [[ "${SKIP_EXP1_COMPARISON:-0}" == "1" || "${SKIP_EXP1_COMPARISON:-}" == "true" ]]; then
        echo "[$label] exp1 path-comparison plots: SKIPPED (SKIP_EXP1_COMPARISON=1)"
        return 0
    fi

    local path_b="$EXP1_OUTDIR/$OPTIMIZER"
    local path_a="$EXP1_INJECT_OUTDIR/$OPTIMIZER"
    if [[ ! -d "$path_b" ]]; then
        echo "[$label] exp1 path-comparison plots: SKIPPED " \
             "(Path B directory $path_b not found; run anticollapse.sh exp1 first)"
        return 0
    fi
    if [[ ! -d "$path_a" ]]; then
        echo "[$label] exp1 path-comparison plots: SKIPPED " \
             "(Path A directory $path_a not found; run anticollapse.sh exp1_inject first)"
        return 0
    fi

    echo "[$label] exp1 path-comparison plots: Path B=$path_b  Path A=$path_a"
    echo "[$label]   outdir=$EXP1_FIG_DIR  dpi=$EXP1_FIG_DPI"

    # Locate the per-path drift JSONs. Both legacy names (drift_const,
    # drift_constgate) are accepted. A loud, prefixed warning is emitted
    # for any path whose drift JSON is missing — the silent fallback used
    # to let missing artifacts slip past unnoticed.
    local has_drift=1
    local pb_drift=""
    local pa_drift=""
    for cand in drift_const drift_constgate; do
        [[ -z "$pb_drift" && -f "$path_b/$cand/tail_saturation.json" ]] && pb_drift="$path_b/$cand/tail_saturation.json"
        [[ -z "$pa_drift" && -f "$path_a/$cand/tail_saturation.json" ]] && pa_drift="$path_a/$cand/tail_saturation.json"
    done
    if [[ -z "$pb_drift" ]]; then
        has_drift=0
        echo "[$label] !! WARNING: Path B drift tail_saturation.json is MISSING under $path_b/drift_const(gate)/"
        echo "[$label] !! WARNING:   exp1_drift_plateau.png will NOT be regenerated."
        echo "[$label] !! WARNING:   Fix:  ./plot_all.sh exp1 full         (drift will run automatically)"
        echo "[$label] !! WARNING:    or:  python $SCRIPT_DIR/diagnostics/run_restoring_drift.py --input_dir $path_b --model const --outdir $path_b/drift_const \\"
        echo "[$label] !! WARNING:         --late_fraction $DRIFT_LATE_FRACTION --min_late_checkpoints $DRIFT_MIN_LATE_CHECKPOINTS --n_bins $DRIFT_N_BINS \\"
        echo "[$label] !! WARNING:         --tail_q_low_primary $DRIFT_TAIL_Q_LOW_PRIMARY --tail_q_low_sweep $DRIFT_TAIL_Q_LOW_SWEEP \\"
        echo "[$label] !! WARNING:         --tail_trim_fraction $DRIFT_TAIL_TRIM_FRACTION --tail_bootstrap_B $DRIFT_TAIL_BOOTSTRAP_B --tail_ci_level $DRIFT_TAIL_CI_LEVEL"
    fi
    if [[ -z "$pa_drift" ]]; then
        has_drift=0
        echo "[$label] !! WARNING: Path A drift tail_saturation.json is MISSING under $path_a/drift_const(gate)/"
        echo "[$label] !! WARNING:   exp1_drift_plateau.png will NOT be regenerated."
        echo "[$label] !! WARNING:   Fix:  ./plot_all.sh exp1_inject full  (drift will run automatically)"
    fi

    if [[ "$has_drift" == "1" ]]; then
        python "$SCRIPT_DIR/plot_exp1_path_comparison.py" \
            --path_b "$path_b" \
            --path_a "$path_a" \
            --outdir "$EXP1_FIG_DIR" \
            --dpi "$EXP1_FIG_DPI" \
            --which all
    else
        echo "[$label]   plotting final spectrum only (drift panel skipped due to missing JSON)"
        python "$SCRIPT_DIR/plot_exp1_path_comparison.py" \
            --path_b "$path_b" \
            --path_a "$path_a" \
            --outdir "$EXP1_FIG_DIR" \
            --dpi "$EXP1_FIG_DPI" \
            --which spectrum
    fi

    # Mirror the per-path envelope plot into the comparison-figures folder
    # so everything for Exp 1 lives in one place. We only mirror the one
    # PNG the manuscript actually references; the diagnostic
    # envelope_aic_bar / envelope_crossover_diagnostic / envelope_fit_r2_bar
    # / envelope_mu_vs_ell plots stay under the per-path plots/ folder for
    # audits and are not paper-bound.
    mkdir -p "$EXP1_FIG_DIR"
    # Only Path A's envelope is mirrored: for ConstGate the canonical
    # mu_0 kernel is trajectory-independent (frozen gate), so Path A and
    # Path B produce byte-identical envelopes. Showing both panels would
    # be redundant. The manuscript's consolidated 3-panel
    # fig:exp1_summary references this single PNG for both paths.
    for fname in log_envelope_vs_ell.png; do
        local stem="${fname%.png}"
        if [[ -f "$path_a/plots/$fname" ]]; then
            cp "$path_a/plots/$fname" "$EXP1_FIG_DIR/exp1_${stem}_pathA.png"
            echo "[$label]   mirrored $fname -> exp1_${stem}_pathA.png"
        fi
    done

    # Regenerate the consolidated Path A vs Path B markdown summary
    # whenever either path's aggregates change. The writer is idempotent
    # and harmless on partial data.
    echo "[$label] writing consolidated exp1 results summary"
    python "$SCRIPT_DIR/write_exp1_summary.py" \
        --path_a "$path_a" \
        --path_b "$path_b" \
        --model const
}

plot_exp1() {
    check_dir "$EXP1_OUTDIR" "Experiment 1"
    echo "============================================================"
    echo "Plotting Experiment 1: ConstGate structural negative control"
    echo "============================================================"
    echo "outdir: $EXP1_OUTDIR"
    python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP1_OUTDIR" \
        --seeds "$SEEDS" \
        --models const \
        --optimizer "$OPTIMIZER" \
        --dpi "$DPI" \
        --plot_only
    run_drift_diagnostic "$EXP1_OUTDIR" "const" "exp1"
    run_exp1_path_comparison "exp1"
}

plot_exp1_inject() {
    check_dir "$EXP1_INJECT_OUTDIR" "Experiment 1 injected-forcing"
    echo "============================================================"
    echo "Plotting Experiment 1 (Path A): ConstGate injected forcing"
    echo "============================================================"
    echo "outdir: $EXP1_INJECT_OUTDIR"
    python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP1_INJECT_OUTDIR" \
        --seeds "$SEEDS" \
        --models const \
        --optimizer "$OPTIMIZER" \
        --dpi "$DPI" \
        --plot_only
    run_drift_diagnostic "$EXP1_INJECT_OUTDIR" "const" "exp1_inject"
    run_exp1_path_comparison "exp1_inject"
}

# ------------------------------------------------------------
# Exp 2 capacity-ladder comparison plots
# (forcing trajectory, beta_env, drift plateau, final spectrum,
#  Delta zeta, phase summary, t_cross), plus per-architecture envelope
# mirroring into the Overleaf-bound figure folder.
#
# Reads only from already-aggregated files under
# $EXP2_OUTDIR/$OPTIMIZER/aggregated/<arch>/, so this helper can
# run standalone (no main_phase_trajectory.py invocation) once an exp2
# simulation has finished.
# ------------------------------------------------------------
run_exp2_phase_ladder() {
    local label="$1"

    if [[ "${SKIP_EXP2_COMPARISON:-0}" == "1" || "${SKIP_EXP2_COMPARISON:-}" == "true" ]]; then
        echo "[$label] exp2 capacity-ladder plots: SKIPPED (SKIP_EXP2_COMPARISON=1)"
        return 0
    fi

    local opt_dir="$EXP2_OUTDIR/$OPTIMIZER"
    if [[ ! -d "$opt_dir" ]]; then
        echo "[$label] exp2 capacity-ladder plots: SKIPPED (directory $opt_dir not found; run anticollapse.sh exp2 first)"
        return 0
    fi

    echo "[$label] exp2 capacity-ladder plots: $opt_dir"
    echo "[$label]   outdir=$EXP2_FIG_DIR  dpi=$EXP2_FIG_DPI  architectures=$EXP2_MODELS"
    IFS=',' read -r -a _arch_list <<< "$EXP2_MODELS"

    local has_any_drift=0
    for arch in "${_arch_list[@]}"; do
        arch="$(echo "$arch" | xargs)"
        [[ -z "$arch" ]] && continue
        if [[ -f "$opt_dir/drift_${arch}/tail_saturation.json" ]]; then
            has_any_drift=1
        fi
    done

    if [[ "$has_any_drift" == "1" ]]; then
        python "$SCRIPT_DIR/plot_exp2_phase_ladder.py" \
            --exp2_dir "$opt_dir" \
            --outdir "$EXP2_FIG_DIR" \
            --architectures "$EXP2_MODELS" \
            --dpi "$EXP2_FIG_DPI" \
            --which all
    else
        echo "[$label]   no exp2 drift tail_saturation.json files found; plotting non-drift figures only"
        for fig in forcing beta_env spectrum delta_zeta phase_summary t_cross; do
            python "$SCRIPT_DIR/plot_exp2_phase_ladder.py" \
                --exp2_dir "$opt_dir" \
                --outdir "$EXP2_FIG_DIR" \
                --architectures "$EXP2_MODELS" \
                --dpi "$EXP2_FIG_DPI" \
                --which "$fig"
        done
    fi

    # Regenerate the markdown results summary from the same aggregates the
    # plots are driven from, so the LaTeX section and the figures stay in
    # lockstep. The writer is idempotent and harmless on partial data.
    echo "[$label] writing exp2 results summary"
    python "$SCRIPT_DIR/write_exp2_summary.py" \
        --exp2_dir "$opt_dir" \
        --architectures "$EXP2_MODELS" \
        --profile "$PROFILE" \
        --seeds "$SEEDS" \
        --output "$RESULTS_DIR/exp2_phase_${PROFILE}_results_summary.md"

    # Generate and mirror per-architecture envelope plots into the comparison-
    # figures folder so all Overleaf-bound exp2 figures live in one place.
    # Naming convention: exp2_<stem>_<arch>.png (e.g.
    # exp2_log_envelope_vs_ell_diag.png).
    mkdir -p "$EXP2_FIG_DIR"
    for arch in "${_arch_list[@]}"; do
        arch="$(echo "$arch" | xargs)"  # trim whitespace
        [[ -z "$arch" ]] && continue
        local arch_agg_dir="$opt_dir/aggregated/$arch"
        local arch_plot_dir="$EXP2_FIG_DIR/_per_arch_${arch}"
        if [[ ! -d "$arch_agg_dir" ]]; then
            echo "[$label]   no aggregated dir for $arch ($arch_agg_dir); skipping envelope mirror"
            continue
        fi
        if [[ ! -f "$arch_agg_dir/${arch}_envelope.csv" && ! -f "$arch_agg_dir/${arch}_envelope_fit_curves.csv" ]]; then
            echo "[$label]   no envelope artifacts for $arch; skipping envelope mirror"
            continue
        fi
        python "$SCRIPT_DIR/plot_exp1_envelopes.py" \
            --indir "$arch_agg_dir" \
            --outdir "$arch_plot_dir" \
            --dpi "$EXP2_FIG_DPI" \
            --debug 0
        # Only mirror the one envelope PNG the manuscript actually
        # references for exp2; the auxiliary envelope_aic_bar / fit_r2 /
        # mu_vs_ell / crossover_diagnostic plots stay under the per-arch
        # plots folder for audits and are not paper-bound.
        for fname in log_envelope_vs_ell.png; do
            local stem="${fname%.png}"
            if [[ -f "$arch_plot_dir/$fname" ]]; then
                cp "$arch_plot_dir/$fname" "$EXP2_FIG_DIR/exp2_${stem}_${arch}.png"
                echo "[$label]   mirrored $arch/$fname -> exp2_${stem}_${arch}.png"
            fi
        done
    done
}

# ------------------------------------------------------------
# Exp 3 forcing-ablation comparison plots and markdown summary.
#
# Reads from $EXP3_OUTDIR/$OPTIMIZER/aggregated after main_forcing_ablation.py has
# aggregated the condition ladders. This can run standalone once the
# simulation/aggregation exists.
# ------------------------------------------------------------
run_exp3_forcing_ablation() {
    local label="$1"
    local opt_dir="$EXP3_OUTDIR/$OPTIMIZER"
    local agg_dir="$opt_dir/aggregated"

    if [[ ! -d "$agg_dir" ]]; then
        echo "[$label] exp3 ablation plots: SKIPPED (directory $agg_dir not found; run anticollapse.sh exp3 first)"
        return 0
    fi

    echo "[$label] exp3 ablation plots: $agg_dir"
    echo "[$label]   outdir=$EXP3_FIG_DIR  dpi=$EXP3_FIG_DPI  models=$EXP3_MODELS"
    mkdir -p "$EXP3_FIG_DIR"

    python "$SCRIPT_DIR/plot_exp3_forcing_ablation.py" \
        --agg_dir "$agg_dir" \
        --outdir "$EXP3_FIG_DIR" \
        --models "$EXP3_MODELS" \
        --dpi "$EXP3_FIG_DPI" \
        --which all

    echo "[$label] writing exp3 results summary"
    python "$SCRIPT_DIR/write_exp3_summary.py" \
        --agg_dir "$agg_dir" \
        --models "$EXP3_MODELS" \
        --output "$RESULTS_DIR/exp3_forcing_${PROFILE}_results_summary.md"
}

plot_exp2() {
    check_dir "$EXP2_OUTDIR" "Experiment 2"
    echo "============================================================"
    echo "Plotting Experiment 2: dynamical phase trajectory (capacity ladder)"
    echo "============================================================"
    echo "models: $EXP2_MODELS"
    echo "outdir: $EXP2_OUTDIR"
    python "$SCRIPT_DIR/main_phase_trajectory.py" \
        --outdir "$EXP2_OUTDIR" \
        --seeds "$SEEDS" \
        --models "$EXP2_MODELS" \
        --optimizer "$OPTIMIZER" \
        --dpi "$DPI" \
        --plot_only
    # Per-architecture drift diagnostic. We loop over the comma-separated
    # EXP2_MODELS list so each rung of the capacity ladder gets its own
    # drift_<arch>/ output, which the capacity-ladder comparison plot then
    # picks up. SKIP_DRIFT=1 / RUN_DRIFT_ON_SMOKE=1 still apply.
    IFS=',' read -r -a _arch_list <<< "$EXP2_MODELS"
    for arch in "${_arch_list[@]}"; do
        arch="$(echo "$arch" | xargs)"
        [[ -z "$arch" ]] && continue
        run_drift_diagnostic "$EXP2_OUTDIR" "$arch" "exp2:$arch"
    done
    run_exp2_phase_ladder "exp2"
}

plot_exp3() {
    check_dir "$EXP3_OUTDIR" "Experiment 3"
    echo "============================================================"
    echo "Plotting Experiment 3: stochastic-forcing ablation"
    echo "============================================================"
    echo "models: $EXP3_MODELS"
    echo "outdir: $EXP3_OUTDIR"
    python "$SCRIPT_DIR/main_forcing_ablation.py" \
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
    run_exp3_forcing_ablation "exp3"
}

case "$COMMAND" in
    exp1)
        plot_exp1
        ;;
    exp1_inject)
        plot_exp1_inject
        ;;
    exp1_figs)
        # Standalone: only regenerate the Path A vs Path B comparison
        # figures + mirror the per-path envelope plots, without re-running
        # main_phase_trajectory.py --plot_only or the drift diagnostic. Reads from the
        # already-on-disk EXP1_OUTDIR and EXP1_INJECT_OUTDIR.
        echo "============================================================"
        echo "Regenerating Exp 1 path-comparison figures (no main_phase_trajectory re-run)"
        echo "============================================================"
        run_exp1_path_comparison "exp1_figs"
        ;;
    exp2)
        plot_exp2
        ;;
    exp2_figs)
        # Standalone: only regenerate the capacity-ladder comparison
        # figures + mirror the per-architecture envelope plots, without
        # re-running main_phase_trajectory.py --plot_only or the drift diagnostic.
        # Reads from the already-on-disk EXP2_OUTDIR.
        echo "============================================================"
        echo "Regenerating Exp 2 capacity-ladder figures (no main_phase_trajectory re-run)"
        echo "============================================================"
        run_exp2_phase_ladder "exp2_figs"
        ;;
    exp3)
        plot_exp3
        ;;
    exp3_figs)
        echo "============================================================"
        echo "Regenerating Exp 3 forcing-ablation figures (no main_forcing_ablation re-run)"
        echo "============================================================"
        run_exp3_forcing_ablation "exp3_figs"
        ;;
    all)
        plot_exp1
        plot_exp1_inject
        plot_exp2
        plot_exp3
        ;;
esac
