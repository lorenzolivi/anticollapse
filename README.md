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

The project studies two coupled empirical questions:

1. How the training trajectory moves in the phase plane defined by:
   - `alpha`: the tail index of stochastic gradient fluctuations
   - `beta`: the tail exponent of the effective time-scale spectrum
2. Whether suppressing stochastic forcing causally reverses anti-collapse.

The main pipeline is organized around two experiments:

- **Experiment 1**: observe phase trajectories during training across
  `ConstGate`, `SharedGate`, `DiagGate`, `GRU`, and `LSTM`
- **Experiment 2**: intervene on stochastic forcing through batch-size,
  gradient clipping, and winsorization ablations

The code also includes a sidecar diagnostics suite in
[`diagnostics/`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/diagnostics)
for validating structural assumptions behind the theory.

## Repository Structure

```text
.
├── run_exp1.py                     # Unified Exp 1 runner
├── run_exp2.py                     # Unified Exp 2 runner
├── main_exp1.py                    # Multi-seed Exp 1 orchestration + aggregation
├── main_exp2.py                    # Multi-seed Exp 2 orchestration + aggregation
├── models.py                       # Shared gated RNN architectures
├── transport.py                    # Transport-factor / mu_{t,ell} computation
├── diagnostics.py                  # Main diagnostics pipeline
├── data.py                         # Synthetic delayed-regression task
├── alpha_utils.py                  # Stable-tail estimation utilities
├── seed_utils.py                   # CSV discovery, loading, aggregation helpers
├── plot_exp1_*.py                  # Exp 1 plotting scripts
├── plot_exp2_ablation.py           # Exp 2 plotting script
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

### Experiment 1: phase trajectories

Single-seed direct run:

```bash
python run_exp1.py \
  --outdir results/exp1_manual/seed_0042 \
  --models const,shared,diag,gru,lstm \
  --H 512 --T 1024 --D 16 \
  --Nseq_train 8000 --Nseq_diag 8000 \
  --epochs 1000 --batch_size 512 --lr 1e-3 --grad_clip 1.0 \
  --checkpoint_every 50 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto --seed 42 --w_seed 1042
```

Multi-seed orchestration:

```bash
python main_exp1.py \
  --outdir results/exp1 \
  --seeds 42,123,321 \
  --models const,shared,diag,gru,lstm \
  --optimizer adamw \
  --epochs 1000 \
  --H 512 --T 1024 --D 16 \
  --Nseq_train 8000 --Nseq_diag 8000 \
  --batch_size 512 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto
```

### Experiment 2: stochastic forcing ablation

Warm-start causal ablation:

```bash
python main_exp2.py \
  --outdir results/exp2_warm_start \
  --seeds 42,123,321 \
  --models diag,gru,lstm \
  --conditions baseline,batch_ablation,clip_ablation,winsorize_ablation \
  --batch_ablation_values 2048,4096,8192 \
  --clip_ablation_values 0.1,0.01,0.001 \
  --winsorize_ablation_values 95,90,80 \
  --warmup_epochs 250 \
  --epochs 1000 \
  --H 512 --T 1024 --D 16 \
  --Nseq_train 8000 --Nseq_diag 8000 \
  --batch_size 512 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
  --save_final_envelope --save_checkpoint_ccdf \
  --device auto
```

## Running The Diagnostics Validation

The sidecar validation scripts live in
[`diagnostics/`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/diagnostics)
and are designed to be invoked from within that folder. Each diagnostic
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

A typical invocation on an existing Exp 1 run directory is:

```bash
python run_restoring_drift.py \
  --input_dir ../results/exp1/adamw \
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

The repository may contain a local
[`latex/`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/latex)
copy of the manuscript for working context, but that directory is intentionally
ignored by git because the paper is managed separately on Overleaf.

## Citation

This repository currently ships with template citation metadata only.
Public manuscript links, DOI information, and the final preferred citation
will be added later.

The repository also includes a machine-readable
[`CITATION.cff`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/CITATION.cff)
file for software metadata.

## License

This project is released under the MIT License. See
[LICENSE](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/LICENSE).
