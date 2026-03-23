# Large-Width Population Concentration Diagnostic: Analysis (v3)

## Setup

60 runs: 2 architectures (DiagGate, GRU) × 6 widths (H = 16, 32, 64, 128, 256, 512) × 5 seeds.
SGD optimizer, T=300, D=8, 200 epochs, lr=1e-3, 128 sequences.
Corrected simulation scripts with updated transport computation.


## Spectral Convergence

**DiagGate:**

| H | W₁ to ref | KS stat | τ median (mean ± std) |
|---|---|---|---|
| 16 | 0.114 | 0.452 | 184.5 ± 9.0 |
| 32 | 0.104 | 0.357 | 201.8 ± 7.4 |
| 64 | 0.059 | 0.264 | 197.1 ± 3.2 |
| 128 | 0.037 | 0.181 | 200.0 ± 2.4 |
| 256 | 0.013 | 0.083 | 198.9 ± 0.5 |
| 512 | — | — | 199.1 ± 0.4 |

**GRU:**

| H | W₁ to ref | KS stat | τ median (mean ± std) |
|---|---|---|---|
| 16 | 0.009 | 0.562 | 1.812 ± 0.010 |
| 32 | 0.006 | 0.427 | 1.801 ± 0.003 |
| 64 | 0.006 | 0.472 | 1.802 ± 0.001 |
| 128 | 0.004 | 0.331 | 1.798 ± 0.000 |
| 256 | 0.002 | 0.225 | 1.797 ± 0.000 |
| 512 | — | — | 1.797 ± 0.000 |

The Wasserstein-1 distance decreases monotonically toward the reference for both architectures. The DiagGate spectrum (which has wider spread) shows clearer convergence; the GRU spectrum is already very narrow at H=16, so the W₁ values are small throughout.

Cross-seed variability of the median τ scales as:
- DiagGate: std(τ_med) ~ H^{−1.01} — essentially exact 1/H concentration
- GRU: std(τ_med) ~ H^{−1.31}


## Envelope Collapse

The intensive envelope f(ℓ) = (1/H)||μ||₁ is width-independent:

- DiagGate: Pearson r = 1.000000 at ALL widths (including H=16)
- GRU: Pearson r ≥ 0.999910 at all widths, reaching 1.000000 at H=512

The population-limit representation provides a faithful description of the envelope decay even at H=16.


## Variance Scaling

Cross-seed variance of f(ℓ) decreases with width:

| H | DiagGate var(f) | GRU var(f) |
|---|---|---|
| 16 | 2.4e-05 | 1.2e-05 |
| 32 | 1.5e-05 | 2.1e-06 |
| 64 | 9.8e-06 | 1.2e-06 |
| 128 | 3.3e-06 | 3.9e-07 |
| 256 | 1.1e-07 | 6.2e-08 |
| 512 | 9.5e-08 | 9.6e-09 |

Variance scaling slopes:
- **DiagGate: var(f) ~ H^{−1.79}**
- **GRU: var(f) ~ H^{−1.94}**

Both faster than the 1/H rate expected from simple mean-field concentration. The H=256→512 step for DiagGate shows saturation (var barely decreases), likely reflecting irreducible seed-dependent variance in the training dynamics.


## Implications for the Anticollapse Paper

1. **Population concentration is empirically validated.** The τ spectrum converges (W₁ decreasing), the cross-seed variability decays as H^{−1} or faster, and the intensive envelope is width-independent (r ≥ 0.99991) across all widths tested.

2. **The Laplace-type representation is meaningful at practical widths.** Even H=16 produces an envelope nearly indistinguishable from the population limit.

3. **Variance scaling at least as fast as 1/H.** DiagGate shows slope ≈ −1.8, GRU shows slope ≈ −1.9, both consistent with (faster than) mean-field predictions.

4. **Both architectures are in the concentrated regime.** GRU has a very narrow τ spectrum (q90/median < 1.02). DiagGate is broader (q90/median ≈ 1.06–1.28, max/median ≈ 1.15–1.64 across widths) but still not in the broad anti-collapse regime. Testing the population limit in the power-law-tail regime remains for the DGX experiments.
