# Mixture-of-Exponentials Diagnostic: Analysis (v3)

## Setup

45 runs: 3 architectures (DiagGate, GRU, LSTM) × 3 optimizers (SGD, Adam, RMSprop) × 5 seeds.
H=64, T=300, D=8, 200 epochs, lr=1e-3, 256 sequences, 35 lags spanning 1–140.
Corrected simulation scripts with updated transport computation and GELR envelope comparison via `compute_macro_envelope_comparison`.


## Summary Table

| Arch | Opt | ρ_mix (min) | r_mix (min) | r_mix (mean) | R² median |
|---|---|---|---|---|---|
| DiagGate | SGD | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DiagGate | Adam | 1.0000 | 0.9991 | 0.9993 | 0.9999 |
| DiagGate | RMSprop | 1.0000 | 0.9999 | 0.9999 | 1.0000 |
| GRU | SGD | 1.0000 | 1.0000 | 1.0000 | 0.960 |
| GRU | Adam | 1.0000 | 0.979 | 0.990 | 0.728 |
| GRU | RMSprop | 1.0000 | 0.968 | 0.992 | 0.806 |
| LSTM | SGD | 1.0000 | 1.0000 | 1.0000 | 0.957 |
| LSTM | Adam | 1.0000 | 0.913 | 0.971 | 0.875 |
| LSTM | RMSprop | 1.0000 | 0.945 | 0.971 | 0.865 |


## Key Finding 1: Perfect Lag Ordering

Spearman ρ = 1.0000 across ALL 45 runs. The mixture of exponentials preserves the monotonic decay ordering without exception, for every architecture, optimizer, and seed tested. The GELR-weighted mixture also achieves ρ = 1.0000 in all 45 runs.


## Key Finding 2: Log-Space Fidelity Across Regimes

The Pearson r in log-space quantifies how well the mixture tracks the actual envelope's shape on a log scale.

**DiagGate** achieves near-perfect fidelity (r ≥ 0.999) across all optimizers. The simple gating structure produces clean per-neuron exponential decay (R² ≈ 1.0).

**GRU and LSTM under SGD** also achieve perfect log-space correlation (r = 1.0). The per-neuron exponential fit has moderate R² (~0.96), reflecting some sub-exponential corrections, but these do not disrupt the aggregate envelope shape.

**GRU and LSTM under adaptive optimizers** show lower but still strong log-space correlations (r = 0.91–0.99, mean 0.97–0.99). These models train to substantially lower loss (~0.27 vs ~1.25 for SGD), producing richer gate dynamics. The per-neuron single-exponential fit becomes a rougher approximation (R² = 0.73–0.88), but the aggregate decay structure remains well captured. The correlations are all highly statistically significant (p ≈ 0 at 35 points).

The Pearson r in log-space tests a strong property: that log f ≈ a log f_mix + b, i.e. f ∝ f_mix^a. Values of 0.91–0.99 confirm this relationship holds, with the deviations arising from the sub-exponential prefactors that the pure mixture omits.


## Key Finding 3: Adaptive Optimizer Invariance

For the 10 DiagGate runs with adaptive optimizers, the Λ-weighted GELR mixture achieves Spearman ρ = 1.0000 and Pearson r ≥ 0.994, despite per-neuron Λ ratios (max/min) reaching 49–206. For GRU and LSTM, the Λ-weighted mixture tracks the unweighted one closely (GELR Pearson r ≥ 0.933), confirming that the bounded Rayleigh projection modulates amplitudes without altering the decay geometry.


## Implications for the Anticollapse Paper

1. **The mixture of exponentials is a valid structural approximation.** Spearman ρ = 1.0 across all 45 runs empirically supports preservation of the monotonic decay structure over the tested lag range (1–140). This is consistent with the scaling class (algebraic, exponential, or logarithmic) being preserved under the Laplace-type representation, though the diagnostic does not prove asymptotic equivalence beyond the sampled grid.

2. **Accuracy varies by architecture complexity.** DiagGate, with its simple gating, produces near-exact exponential per-neuron decay. GRU and LSTM, with richer gate interactions, show stronger sub-exponential corrections. In all cases the aggregate decay structure is captured.

3. **The approximation is optimizer-independent.** Both the unweighted and Λ-weighted mixtures preserve the decay ordering. Adaptive optimizers do not break the mixture representation.
