# Anti-Collapse Dynamics in Gated Recurrent Neural Networks

Code for the empirical validation in:

**Lorenzo Livi**  
*Anti-Collapse Dynamics and the Emergence of Multi-Time-Scale Learning in Recurrent Neural Networks*

Preprint: [arXiv:2606.29519](https://arxiv.org/abs/2606.29519).

This repository contains the training, diagnostic, and plotting pipeline for
the two main-text experiments. The experiments test when trained gated
recurrent networks realize a broad spectrum of effective time scales and the
corresponding slowly decaying macroscopic envelope.

## Experiments

The public replication pipeline is organized around two experiments:

- **Experiment 1: structural negative control.** ConstGate is trained with
  frozen gates, both spontaneously and with an externally supplied heavy-tailed
  slow-mode update driver. The goal is to test whether forcing alone is
  sufficient when the architecture cannot realize heterogeneous trainable time
  scales.
- **Experiment 2: access route through trainable diagonal gates.** SharedGate
  and DiagGate are trained on the same heavy-tailed-lag task. SharedGate is the
  trainable-but-collapsed reference; DiagGate is the minimal per-unit gate
  architecture used to test the proposed access route.

The paper-scale runs are resource-bounded: they are the largest systematic
multi-seed, multi-checkpoint experiments we could run with the full diagnostic
suite on the available compute. The code is written so that researchers with
larger compute budgets can scale the same protocols to more seeds, wider
models, longer horizons, and denser diagnostics.

## Repository Layout

```text
.
├── anticollapse.sh                  # main experiment launcher
├── plot_all.sh                      # deferred analysis and figure launcher
├── main_phase_trajectory.py         # multi-seed orchestrator for Exp1/Exp2
├── run_phase_trajectory.py          # single-seed runner
├── models.py                        # ConstGate, SharedGate, DiagGate, GRU, LSTM
├── data.py                          # synthetic delayed-regression task
├── diagnostics.py                   # core spectrum/envelope/forcing diagnostics
├── transport.py                     # transport-factor computations
├── alpha_utils.py                   # stable-tail utilities
├── seed_utils.py                    # aggregation helpers
├── plot_exp1_path_comparison.py     # Exp1 paper figures
├── plot_exp1_envelopes.py           # envelope figure helper
├── plot_exp2_phase_ladder.py        # Exp2 paper figures
├── write_exp1_summary.py            # Exp1 markdown summary
├── write_exp2_summary.py            # Exp2 markdown summary
└── diagnostics/
    ├── run_restoring_drift.py       # far-left restoring-drift diagnostic
    ├── run_zeta_residual_forcing.py # drift-subtracted log-rate residual test
    └── validate_forcing_tail_estimators.py
```

Generated results, figures, logs, LaTeX sources, caches, and local validation
outputs are intentionally excluded from git.

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Core dependencies:

- PyTorch
- NumPy
- SciPy
- Matplotlib

The launchers use `--device auto`. On CUDA systems this selects CUDA; on Apple
Silicon it can select MPS for smaller runs; CPU remains available for smoke
tests and post-processing.

## Quick Start

Run smoke tests first. Smoke runs validate the pipeline only; they should not
be interpreted as empirical evidence for the manuscript claims.

```bash
./anticollapse.sh exp1 smoke
./plot_all.sh exp1 smoke

./anticollapse.sh exp2 smoke
./plot_all.sh exp2 smoke
```

Full paper-scale runs:

```bash
./anticollapse.sh exp1 full
./plot_all.sh exp1 full

./anticollapse.sh exp2 full
./plot_all.sh exp2 full
```

Run both experiments sequentially:

```bash
./anticollapse.sh all full
./plot_all.sh all full
```

## Launcher Behavior

`anticollapse.sh` starts long simulations in the background by default and
writes a log and PID file next to the result directory:

```text
results/exp1_constgate_full.log
results/exp1_constgate_full.pid
results/exp1_constgate_inject_full.log
results/exp1_constgate_inject_full.pid
results/exp2_phase_full.log
results/exp2_phase_full.pid
```

Monitor a run:

```bash
tail -f results/exp2_phase_full.log
```

Stop a run:

```bash
kill $(cat results/exp2_phase_full.pid)
```

Use `FOREGROUND=1` for an interactive foreground run:

```bash
FOREGROUND=1 ./anticollapse.sh exp2 smoke
```

`plot_all.sh` does not train models. It reloads saved checkpoints and
regenerates deferred final-envelope analysis, far-left drift diagnostics,
drift-subtracted log-rate residual diagnostics, figures, and markdown
summaries.

## Default Full Configuration

The full profile uses:

- seeds: `47,83,12,69,31,104,218,337,451,592`
- hidden width: `H=512`
- sequence length: `T=1280`
- input dimension: `D=16`
- epochs: `1400`
- checkpoint schedule: epoch `1`, then every `40` epochs
- training set size: `Nseq_train=8000`
- diagnostic set size: `Nseq_diag=6000`
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`
- gradient clipping: global L2 clip at `1.0`
- task: heavy-tailed-lag synthetic regression with tail index `0.6`,
  `K=16`, lag range `[8,640]`

Important environment overrides:

```bash
SEEDS=47,83 EXP2_MODELS=diag ./anticollapse.sh exp2 full
H=256 EPOCHS=400 ./anticollapse.sh exp1 smoke
RESULTS_DIR=/path/to/results ./plot_all.sh exp2 full
```

## Outputs

Default output directories:

```text
results/exp1_constgate_<profile>/
results/exp1_constgate_inject_<profile>/
results/exp2_phase_<profile>/
results/exp1_figures/
results/exp2_figures/
```

Main generated summaries:

```text
results/exp1_results_summary.md
results/exp2_phase_<profile>_results_summary.md
```

Main paper-facing figure folders:

```text
results/exp1_figures/
results/exp2_figures/
```

The figures in these folders are the intended files to copy into the
manuscript project.

## Re-running Analysis Without Training

If simulations are already complete, regenerate analysis and figures only:

```bash
./plot_all.sh exp1 full
./plot_all.sh exp2 full
```

To regenerate only the comparison figures from existing aggregates:

```bash
./plot_all.sh exp1_figs full
./plot_all.sh exp2_figs full
```

## Notes On Scaling

The expensive part of the pipeline is not just training. The diagnostics
include per-checkpoint time-scale extraction, calibrated forcing-tail
estimation, bootstrap summaries, envelope reconstruction, far-left drift
estimation, and residual forcing tests. Increasing width, seeds, checkpoint
density, or diagnostic bootstrap counts can substantially increase runtime.

For larger compute budgets, the cleanest scaling knobs are:

- `SEEDS`
- `H`
- `EPOCHS`
- `NSEQ_TRAIN`
- `NSEQ_DIAG`
- `CHECKPOINT_EVERY`
- diagnostic bootstrap settings exposed in `anticollapse.sh` and `plot_all.sh`

## Citation

If you use this code, please cite the accompanying preprint:

```bibtex
@misc{livi2026anticollapse,
  title = {Anti-Collapse Dynamics and the Emergence of Multi-Time-Scale Learning in Recurrent Neural Networks},
  author = {Livi, Lorenzo},
  year = {2026},
  eprint = {2606.29519},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2606.29519}
}
```

The repository also includes `CITATION.cff` for citation managers and GitHub's
citation widget.

## License

MIT License. See `LICENSE`.
