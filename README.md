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
├── CLI_examples.txt                # End-to-end command examples
├── diagnostics/
│   ├── run_mixture.py              # Mixture-of-exponentials validation
│   ├── run_width_scaling.py        # Population-concentration validation
│   ├── run_restoring_drift.py      # Drift diagnostics
│   ├── diag_utils.py               # Shared helpers for diagnostics-side runs
│   └── analysis_*.md               # Analysis notes for diagnostics results
└── README.md
```

Legacy one-off scripts such as
[`run_anticollapse_exp1_phase_trajectory_baselines.py`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/run_anticollapse_exp1_phase_trajectory_baselines.py),
[`run_anticollapse_exp1_phase_trajectory_lstmgru.py`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/run_anticollapse_exp1_phase_trajectory_lstmgru.py),
and
[`run_anticollapse_exp2_forcing_ablation.py`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/run_anticollapse_exp2_forcing_ablation.py)
are kept for reference, but the unified `run_exp1.py` and `run_exp2.py` runners
are the primary entry points.

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

The main workflow is documented in
[`CLI_examples.txt`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/CLI_examples.txt).

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
[`diagnostics/`](/Users/lorenzo/university/research/projects/dynamical_theory_learning/anticollapse/diagnostics).

From that folder, the two main validation runs are:

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
