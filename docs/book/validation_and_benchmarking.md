# Validation and Benchmarking

![Solver validation stack](figures/solver_validation_stack.svg)

## Scope

Validation covers analytical benchmarks, method of manufactured solutions, k-Wave/k-wave-python parity, pykwavers bindings, Apollo transform parity, GPU/CPU equivalence, benchmark reports, and figure generation. Code ownership maps to `kwavers::solver::validation`, `pykwavers/examples`, `benchmarks`, and `apollo`.

## Theorem: Correlation Is Not Error Sufficiency

Two nonzero vectors can have Pearson correlation `1` while their amplitudes differ by a constant scale factor.

### Proof Sketch

Let `y = alpha x` with `alpha > 0` and nonconstant `x`. The centered vectors are also scalar multiples, so correlation is `1`, but `||x - y||` is nonzero when `alpha != 1`.

## Algorithm: Validation Report Contract

1. Compare raw fields or traces before derived images.
2. Report absolute error, relative error, correlation, RMS ratio, and phase metrics when applicable.
3. Cache only reproducible reference artifacts.
4. Track 1-D, 2-D, and 3-D benchmark times and memory separately.

## Implementation Targets

- Keep k-Wave, k-wave-python, k-wave-julia, kwavers, and pykwavers comparisons explicitly labeled.
- Do not weaken workloads to improve benchmark results.
- Use Apollo FFT parity tests for transform-level discrepancies before solver-level attribution.

## Research Anchors

- k-Wave MATLAB toolbox: http://www.k-wave.org/
- k-wave-python documentation: https://k-wave-python.readthedocs.io/
- k-Wave.jl repository: https://github.com/JClingo/k-wave-julia
