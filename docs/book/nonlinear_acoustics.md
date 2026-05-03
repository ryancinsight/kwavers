# Nonlinear Acoustics

## Scope

Nonlinear acoustics covers Westervelt, Kuznetsov, KZK, Burgers, harmonic generation, shock capturing, absorption, and thermoviscous losses. Code ownership maps to `kwavers::solver::forward::nonlinear`, `kwavers::physics::acoustics::wave_propagation::nonlinear`, and Apollo-backed spectral operators.

## Theorem: Quadratic Nonlinearity Generates Harmonics

If a pressure field contains `p(t) = A cos(omega t)`, then a quadratic term contains

```text
p(t)^2 = A^2/2 + A^2/2 cos(2 omega t).
```

### Proof Sketch

The trigonometric identity `cos^2(x) = (1 + cos(2x))/2` shows that a quadratic acoustic term creates a DC component and a second harmonic.

## Algorithm: Nonlinear Solver Acceptance

1. State whether the model is Westervelt, Kuznetsov, KZK, Burgers, or another equation.
2. Validate linear-limit behavior when nonlinearity tends to zero.
3. Validate harmonic generation against analytical or literature cases.
4. Track shock-capturing activation and absorption separately.

## Implementation Targets

- Keep nonlinear source terms separate from linear propagation operators.
- Reuse spectral scratch buffers and caller-owned arrays.
- Preserve conservation diagnostics as optional instrumentation, not as solver state mutation.

## Research Anchors

- Mixed approximation of Kuznetsov and Westervelt equations: https://doi.org/10.1016/j.apnum.2023.12.001
- Westervelt drug-delivery modeling context: https://arxiv.org/abs/2412.07490
