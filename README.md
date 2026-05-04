# Anti-Collapse Dynamics in Gated Recurrent Neural Networks

Code accompanying the ongoing project:

**Lorenzo Livi**  
*Anti-Collapse Dynamics and the Emergence of Multi-Time-Scale Learning in Recurrent Neural Networks*

Publication metadata such as public manuscript links, DOI, and release citation
will be added later. The repository metadata is intentionally kept in template
form until those details are finalized.

This repository contains the experimental and diagnostic code for the
anti-collapse project. The focus is on how broad time-scale spectra emerge
during training, how they relate to heavy-tailed stochastic forcing, and how
that phase structure can be measured empirically in gated recurrent networks.

## Overview

The project studies three coupled empirical questions:

1. Whether a structural negative control can satisfy route diagnostics while
   failing to realize anti-collapse.
2. How the training trajectory moves in the phase plane defined by:
   - `alpha`: the tail index of stochastic gradient fluctuations
   - `beta`: the tail exponent of the effective time-scale spectrum
3. Whether suppressing stochastic forcing contracts or reverses anti-collapse.

The main pipeline is organized around three main-text experiments:

- **Experiment 1**: structural negative control (`ConstGate` + AdamW)
- **Experiment 2**: observe phase trajectories during training across
  the architecture--optimizer capacity ladder
- **Experiment 3**: intervene on stochastic forcing through batch-size,
  gradient clipping, and winsorization ablations

The code also includes a sidecar diagnostics suite in `diagnostics/`
for validating structural assumptions behind the theory.

## Repository Structure

```text
.
├── anticollapse.sh                 # Unified main-text experiment launcher
├── plot_all.sh                     # Unified plotting / analysis launcher
├── run_exp1.py                     # Phase-trajectory runner used by Exp 1/2
├── run_exp2.py                     # Forcing-ablation runner used by Exp 3
├── main_exp1.py                    # Multi-seed phase-trajectory orchestration
├── main_exp2.py                    # Multi-seed forcing-ablation orchestration
├── models.py                       # Shared gated RNN architectures
├── transport.py                    # Transport-factor / mu_{t,ell} computation
├── diagnostics.py                  # Main diagnostics pipeline
├── data.py                         # Synthetic delayed-regression task
├── alpha_utils.py                  # Stable-tail estimation utilities
├── seed_utils.py                   # CSV discovery, loading, aggregation helpers
├── plot_exp1_*.py                  # Phase-trajectory plotting scripts
├── plot_exp2_ablation.py           # Exp 3 forcing-ablation plotting script
├── diagnostics/
│   ├── run_mixture.py              # Diagnostic 1: mixture-of-exponentials validation
│   ├── run_width_scaling.py        # Diagnostic 2: population-concentration validation
│   ├── run_restoring_drift.py      # Diagnostic 3: restoring-drift + far-tail closure
│   ├── run_drift_validation.py     # Diagnostic 3 orchestrator (train + drift diagnostic + figure)
│   ├── diag_utils.py               # Shared helpers for diagnostics-side runs
│   └── analysis_*.md               # Analysis notes for diagnostics results
└── README.md
```

Legacy monolithic experiment scripts (previously
`run_anticollapse_exp1_phase_trajectory_baselines.py`,
`run_anticollapse_exp1_phase_trajectory_lstmgru.py`,
`run_anticollapse_exp2_forcing_ablation.py`) have been deleted.
The modular `run_exp1.py` and `run_exp2.py` runners are the only entry points.

## Requirements

```bash
pip install -r requirements.txt
```

The code requires Python 3.9+.

Core dependencies:

- PyTorch
- NumPy
- Matplotlib
- SciPy

`SciPy` is required for parts of the diagnostics and validation suite.

## Hardware Notes

The project is designed to run on:

- CUDA GPUs for large production runs
- Apple Silicon via `--device auto` for laptop-scale runs
- CPU for smaller smoke tests and post-processing

On a MacBook Pro, `--device auto` will usually select `mps` when available.
If `mps` becomes unstable for a long run, rerun with `--device cpu`.

## Running The Main Experiments

Use the unified main-text launcher for publication and smoke runs:

```bash
./anticollapse.sh exp1 smoke   # ConstGate structural negative control
./anticollapse.sh exp2 smoke   # phase trajectory / capacity ladder
./anticollapse.sh exp3 smoke   # stochastic forcing ablation
./anticollapse.sh all  smoke   # all three smoke pipelines

./anticollapse.sh exp1 full
./anticollapse.sh exp2 full
./anticollapse.sh exp3 full
```

The full profile uses the heavy-tailed-lag task and writes under
`results/exp1_constgate_full`, `results/exp2_phase_full`, and
`results/exp3_forcing_full` by default.  On DGX Spark, prefer splitting
large runs by model:

```bash
EXP2_MODELS=diag,gru ./anticollapse.sh exp2 full
EXP2_MODELS=lstm ./anticollapse.sh exp2 full
EXP3_MODELS=diag ./anticollapse.sh exp3 full
```

`anticollapse.sh` launches in the background by default. Each run creates
a log and pid file next to its result directory, for example:

```text
results/exp2_phase_full.log
results/exp2_phase_full.pid
```

Monitor with `tail -f results/exp2_phase_full.log`. Stop with
`kill $(cat results/exp2_phase_full.pid)`. Set `FOREGROUND=1` for an
interactive foreground run.

Existing results can be re-aggregated and plotted without rerunning
simulations via:

```bash
./plot_all.sh exp1 full
./plot_all.sh exp2 full
./plot_all.sh exp3 full
```

The root orchestration scripts remain available for direct use:
`main_exp1.py` is the phase-trajectory engine used by main-text
Experiments 1 and 2, while `main_exp2.py` is the forcing-ablation
engine used by main-text Experiment 3.

### Experiment 1: structural negative control

ConstGate + AdamW on the heavy-tailed-lag task:

```bash
python main_exp1.py \
  --outdir results/exp1_constgate_full \
  --seeds 42,123,321,456,789 \
  --models const \
  --optimizer adamw \
  --task_variant heavy_tail --task_alpha 0.6 --task_K 16 \
  --task_lag_min 8 --task_lag_max 640 \
  --epochs 1400 --checkpoint_every 20 \
  --H 512 --T 1280 --D 16 \
  --Nseq_train 8000 --Nseq_diag 6000 \
  --alpha_n_directions 16 --alpha_n_grad_batches_ckpt 128 \
  --tau_fit_lag_min 64 --tau_fit_lag_max 384 --tau_fit_num_lags 32 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto
```

### Experiment 2: dynamical phase trajectory

Capacity ladder on the same task:

```bash
python main_exp1.py \
  --outdir results/exp2_phase_full \
  --seeds 42,123,321,456,789 \
  --models shared,diag,gru,lstm \
  --optimizer adamw \
  --task_variant heavy_tail --task_alpha 0.6 --task_K 16 \
  --task_lag_min 8 --task_lag_max 640 \
  --epochs 1400 --checkpoint_every 20 \
  --H 512 --T 1280 --D 16 \
  --Nseq_train 8000 --Nseq_diag 6000 \
  --alpha_n_directions 16 --alpha_n_grad_batches_ckpt 128 \
  --tau_fit_lag_min 64 --tau_fit_lag_max 384 --tau_fit_num_lags 32 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto
```

### Experiment 3: stochastic forcing ablation

Warm-start causal ablation:

```bash
python main_exp2.py \
  --outdir results/exp3_forcing_full \
  --seeds 42,123,321,456,789 \
  --models diag \
  --conditions baseline,batch_ablation,clip_ablation,winsorize_ablation \
  --batch_ablation_values 2048,4096,8192 \
  --clip_ablation_values 0.1,0.01,0.001 \
  --winsorize_ablation_values 95,90,80 \
  --warmup_epochs 200 \
  --task_variant heavy_tail --task_alpha 0.6 --task_K 16 \
  --task_lag_min 8 --task_lag_max 640 \
  --epochs 1400 --checkpoint_every 20 \
  --H 512 --T 1280 --D 16 \
  --Nseq_train 8000 --Nseq_diag 6000 \
  --alpha_n_directions 16 --alpha_n_grad_batches_ckpt 128 \
  --tau_fit_lag_min 64 --tau_fit_lag_max 384 --tau_fit_num_lags 32 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto
```

## Running The Diagnostics Validation

The sidecar validation scripts live in `diagnostics/` and are designed to be
invoked from within that folder. Each diagnostic
targets a specific structural assumption of the anti-collapse theory.

### Diagnostic 1 — Mixture-of-exponentials validation (`run_mixture.py`)

Tests whether the first-order diagonal mixture

    f_mix(ℓ) = (1/H) Σ_q exp(-ℓ/τ_q)

faithfully reproduces the transport envelope

    f_transport(ℓ) = (1/H) Σ_q |μ^{(q)}_{t,ℓ}|

computed from the actual per-neuron transport factors. Under adaptive
optimizers it additionally checks the lag-dependent Rayleigh-weighted
mixture against the GELR envelope. This is the empirical check of the
representation that the τ_q spectrum is the object one should be modelling
in the first place. Outputs include per-run envelopes, per-neuron (τ_q, r²)
tables, neuron traces, and a cross-run summary.

```bash
python run_mixture.py \
  --device auto \
  --outdir results_mixture_rerun \
  --H 64 --T 300 --D 8 --Nseq 256 --epochs 200 \
  --lr 1e-3 --batch_size 32 \
  --max_lag 140 --fit_lag_min 16 \
  --seeds 42,123,321,456,789 \
  --archs diag,gru,lstm \
  --optimizers sgd,adam,rmsprop
```

### Diagnostic 2 — Population concentration / large-width limit (`run_width_scaling.py`)

Tests whether the empirical log-τ spectrum and the intensive envelope
`f(ℓ) = (1/H)‖μ(ℓ)‖₁` stabilize as `H → ∞`. Concretely it reports:

- Wasserstein distance between the τ spectrum at width `H` and at the
  largest width (spectral convergence),
- pairwise Pearson correlation of `f(ℓ)` across widths (envelope collapse),
- cross-seed variance of `f(ℓ)` as a function of width (concentration,
  expected to scale like `1/H`).

This is the empirical check that the thermodynamic-limit description used
throughout the theory is a valid asymptote at the widths we actually train
at.

```bash
python run_width_scaling.py \
  --device auto \
  --outdir results_width_rerun \
  --T 300 --D 8 --Nseq 128 --epochs 200 \
  --lr 1e-3 --batch_size 32 --optimizer sgd \
  --max_lag 140 --fit_lag_min 16 \
  --widths 16,32,64,128,256,512 \
  --seeds 42,123,321,456,789 \
  --archs diag,gru
```

### Diagnostic 3 — Restoring drift and far-left-tail closure (`run_restoring_drift.py`)

Tests the two components of the drift closure used in the stochastic
log-spectrum model. From saved late-training checkpoints, define

    ζ_q(t) = -log τ_q(t),

and estimate the conditional drift
`F̂(ζ) ≈ E[Δζ | ζ_t ≈ ζ]` by binning the transitions between consecutive
checkpoints in the late-training window. The diagnostic reports:

1. **Bulk restoring drift.** Linear-slope estimate of `F̂` near the
   late-training bulk median, reported as `κ̂ = -slope` with a Spearman
   correlation and an "inward fraction" statistic relative to the bulk.
2. **Moment stabilization.** Late-training cross-neuron mean and variance
   of `ζ_q(t)` should level off rather than drift indefinitely.
3. **Far-left-tail closure (new).** Targeted empirical check of
   `F(ζ) = κ + o(1)` as `ζ → -∞`. On the slice
   `{ζ ≤ quantile_{q_low}(ζ)}` the script computes a trimmed-mean
   plateau `κ̂_tail`, the slope of `F̂` inside the slice, and a
   constant-vs-linear BIC comparison. Confidence intervals for all three
   quantities are obtained via a block bootstrap whose blocks are
   individual checkpoint transitions (the natural unit of dependence for
   non-i.i.d. neurons). A `q_low` sweep over `{0.05, 0.10, 0.15}` is run
   by default. Results are written to `tail_saturation.{json,csv}` and a
   combined figure `conditional_drift_with_tail.png` overlays `F̂(ζ)`,
   the shaded tail slice, and the `κ̂_tail` plateau with its CI band.

A typical invocation on an existing phase-trajectory run directory is:

```bash
python run_restoring_drift.py \
  --input_dir ../results/exp2_phase_full/adamw \
  --model gru \
  --outdir results_restoring_drift_gru \
  --late_fraction 0.3 \
  --n_bins 18 \
  --tail_q_low_primary 0.10 \
  --tail_q_low_sweep 0.05,0.10,0.15 \
  --tail_trim_fraction 0.1 \
  --tail_bootstrap_B 2000 \
  --tail_ci_level 0.90
```

#### Orchestrator: `run_drift_validation.py`

For the concrete paper figure, the recommended entry point is the
orchestrator, which wraps training + diagnostic + figure export:

1. trains a single architecture (default GRU) for 3 seeds on the
   heavy-tailed-lag task variant (truncated-Pareto lag distribution with
   tail index `α_task`, set via `--task_variant heavy_tail`) with dense
   checkpointing,
2. invokes `run_restoring_drift.py` with the far-tail saturation
   diagnostic enabled,
3. copies `conditional_drift_with_tail.png` into `latex/figures/` as
   `drift_validation.png` for inclusion in the appendix.

```bash
python run_drift_validation.py --outdir results_drift_validation
# diagnostic-only rerun on the same training output:
python run_drift_validation.py --outdir results_drift_validation --skip_train
```

## Outputs

Typical outputs include:

- per-model learning curves
- `phase_trajectory.csv`
- checkpoint tau spectra and CCDF fits
- checkpoint alpha diagnostics
- final transport and GELR envelope files
- multi-seed aggregated CSVs and plots

Generated results are intentionally kept out of git.

## Notes On The Paper Files

The repository may contain a local `latex/` copy of the manuscript for working
context, but that directory is intentionally
ignored by git because the paper is managed separately on Overleaf.

## Citation

This repository currently ships with template citation metadata only.
Public manuscript links, DOI information, and the final preferred citation
will be added later.

The repository also includes a machine-readable `CITATION.cff` file for
software metadata.

## License

This project is released under the MIT License. See `LICENSE`.
