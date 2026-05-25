# ADR-008: Finite-Window PSTD Born Forward Boundary

## Status

Accepted.

## Context

The Ali 2025 reduced breast-UST probe separates two contracts:

- stationary frequency-domain Helmholtz/CBS operators;
- finite-window PSTD acquisition with source differencing, k-space source correction, and trailing-cycle demodulation.

`PstdSpectralConvergentBornOperator` shares the homogeneous PSTD modal symbol,
but it remains a stationary CBS operator. The scattering-increment diagnostic
showed that applying finite-window source/bin transfer to only the stationary
source projection over-amplifies heterogeneous increments by approximately
`985-989x` on the determined `(4,4,3)` probe.

## Decision

Add a solver-owned forward diagnostic:

`solver::inverse::fwi::frequency_domain::simulate_pstd_finite_window_born_observation`.

The component implements the first-order finite-window PSTD recurrence directly:

```text
p0[n+1] = (2 - theta^2) p0[n] - p0[n-1] + S[n]
ps[n+1] = (2 - theta^2) ps[n] - ps[n-1]
          - chi * (p0[n+1] - 2 p0[n] + p0[n-1])
chi = (s^2 - s0^2) / s0^2
```

The PyO3 function
`simulate_breast_fwi_pstd_finite_window_born_observation` is a conversion-only
wrapper over the Rust solver component.

## Consequences

- CBS remains the owner of stationary Helmholtz fixed-point algebra.
- Finite-window scattering is not exposed as a `HelmholtzForwardOperator` until
  the matching adjoint-gradient theorem is implemented.
- Clinical replication code can compare finite-window heterogeneous increments
  without Python owning propagation math.

## Verification

- Value-semantic Rust test: exact linearity in normalized slowness-squared
  contrast.
- Domain rejection test: off-grid PSTD ring geometry fails at the solver
  boundary.
- PyO3 compile gate: `cargo check -p pykwavers --lib`.

## Residual Risk

The Rust forward diagnostic is implemented, but the reduced Ali 2025 report has
not yet been regenerated with this model included in the clinical comparison
artifact. That integration must remain Rust/PyO3-owned; Python may only
orchestrate serialization and plotting.
