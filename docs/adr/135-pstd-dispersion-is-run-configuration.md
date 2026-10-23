# ADR 135: Power-law dispersion is part of every PSTD run's configuration

Status: Accepted

## Context

The CPU PSTD solver has offered `PowerLawDispersion::Omitted` (k-Wave's
`alpha_mode = 'no_dispersion'`) through `PSTDConfig`. Neither the GPU PSTD
runner nor the simulation crate's PSTD dispatch exposed it:

- the GPU runner computed `η = 2α₀c₀^y·tan(πy/2)` unconditionally and rejected
  `y = 1`, the exponent tissue tables give for brain;
- the dispatch hardcoded causal dispersion and replaced the configured
  exponent with the medium's exponent sampled at the grid origin.

A downstream consumer modelling `y = 1` tissue could therefore run only on the
CPU, and only by building the solver directly.

The CPU solver also resolved a nearly uniform exponent field by sampling voxel
`[0, 0, 0]`. It built the uniform operators even when it then built the
stratified ones.

## Decision

1. The dispersion choice is a field of each PSTD run's configuration:
   - `GpuPstdRunConfig.power_law_dispersion`;
   - `NonlinearConfig.power_law_dispersion`, serde default `Causal`, so stored
     configurations keep their meaning;
   - an argument of `PstdMediumSnapshot::from_medium`, which validates the
     exponent against it.

   `Omitted` sets `η = 0` and admits `y = 1`. `Causal` keeps its `y = 1`
   rejection, and the error names `Omitted`.
2. The dispatch passes the configured exponent through. The solver's own
   resolution owns heterogeneous exponents.
3. The absorption kernel holds one `ExponentSpectrum`: either the uniform
   symbols with their exponent, or the strata. A field collapses to the
   midpoint of its range only when every exponent-dependent factor, including
   causal `tan(πy/2)`, moves by no more than the bound the operators already
   accept (`strata::collapses_to_uniform`). Otherwise the strata reproduce each
   voxel's exponent.

## Rejected alternatives

- **Choose the mode from the exponent (no dispersion whenever `y = 1`).** This
  silently changes the physics at one exponent value and hides the choice
  k-Wave makes explicit.
- **A separate GPU entry point for absorption without dispersion.** That would
  duplicate the runner for a one-factor difference.
- **Keep origin sampling and document it.** The exponent used would depend on
  where the grid starts, not on the medium.

## Consequences

This is a breaking change for `kwavers-gpu` and `kwavers-simulation` struct
literals and for `from_medium` callers. Adding
`power_law_dispersion: PowerLawDispersion::Causal` keeps the previous
behaviour. The Python bindings still carry their own GPU operator construction
and the origin-sampled exponent. KW-PY-GPU-ABSORPTION tracks moving them onto
this seam.

Overturning evidence: a dispersion form other than these two that a consumer
needs. That would turn the enum into a model trait.

Delivered by ryancinsight/kwavers#959.
