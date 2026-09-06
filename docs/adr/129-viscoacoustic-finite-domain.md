# ADR 129: The viscoacoustic domain is finite-and-positive, and the absorbing layer is fallible

- Status: Accepted
- Date: 2026-09-06
- Item: `backlog.md#kw-viscoacoustic-finite-domain-2026-08-31`
- Related: [ADR 116](116-workspace-clippy-floor.md) (the error contract style
  this follows), PR #708 (the active-axis state layout the validation now
  guards ahead of)

## Context

`ViscoacousticMemorySolver::new` and `new_heterogeneous` validated their
positive-valued parameters with bare `v <= 0.0` comparisons, so `NaN` and
positive infinity passed every guard and reached retained coefficient,
wavenumber, and state construction. A `NaN` density produced a `NaN` `inv_rho`
coefficient field (or scalar) that no later step rejects; a `NaN` time step
exponentiated into the relaxation decay directly. The relaxation-arm guards
had the same hole. These are safe public input paths — no `unsafe`, no
invariant beyond calling a public constructor.

`enable_absorbing_layer` compounded two defects. It accepted a non-finite
`gamma_max`, which produced a `NaN`-laden decay field (`exp(-NaN·dt)`), and
its per-axis extent test computed `2 * thickness`: a `usize` thickness above
`usize::MAX / 2` wrapped the guard to a small number, the test read false,
and `n - thickness` underflowed into an enormous index — reachable from a
safe public call with a single extreme argument.

## Decision

Every scalar and field element a viscoacoustic constructor accepts as a
physical quantity is validated as **finite and strictly positive** before any
allocation or state mutation. The predicate is
`v.is_finite() && v > 0.0`; the `ΔM` field keeps its lossless-voxel zero but
gains the finiteness half (`v.is_finite() && v >= 0.0`). Signed zero stays
rejected: `-0.0` is finite, but it is not positive, and the physical quantity
is.

`enable_absorbing_layer` returns `KwaversResult<()>`. A non-finite
`gamma_max` is rejected before the retained decay field is touched, so a
rejected reconfiguration leaves the existing damping state and subsequent
pressure evolution untouched — the same rejection-preserves-state contract
`reserve_sensor_samples` already holds. The extent test compares the
thickness against `n.div_ceil(2)` instead of doubling it, which rejects
exactly the axes the wrapped expression rejected, for every `usize` input,
without overflow. An axis too short to host a layer keeps its established
zero-damping behavior; when no axis can host one, the retained state is the
established no-layer `None` rather than an all-ones decay grid.

## Consequences

- `enable_absorbing_layer` changes return type (`()` → `KwaversResult<()>`).
  The empirical SemVer-gate verdict (`cargo-semver-checks` against
  `origin/main`: 223 pass, 31 skip) is **no semver update required**: a
  function that begins returning a value where it returned `()` before is
  source-compatible — existing call statements still compile, now with an
  `unused_must_use` warning instead of a silent success they cannot inspect.
  The break is contractual, not syntactic: a call that could previously
  silently accept a non-finite rate and no-op into a `NaN` decay field now
  has an error path the caller can (and the lint nudges it to) handle. There
  is no compatibility wrapper and no silent fallback; the two first-party
  callers (one example, three test sites) were updated in the same change.
- Rejection allocates no solver state. The allocation-contract harness pins
  that a rejected construction retains only the diagnostic message (under
  1 KiB), not the 327,680-byte retained footprint a warm 1-D build carries.
- Accepted finite behavior is unchanged: every parameter vector the old
  guards accepted, the new guards accept, and the numerical results are
  bit-identical. The warm-construction allocation counts and retained bytes
  at the probe shapes still match the #708 figures.
- The table-driven structural tests cover `NaN`, both infinities, signed
  zero (where applicable), and finite negatives for every constructor
  parameter, relaxation field, and the absorbing-layer rate — including the
  extreme-`usize` thickness cases that previously underflowed.
