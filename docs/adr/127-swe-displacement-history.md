# 127. Retain displacement-only SWE history

- Status: Accepted
- Date: 2026-08-30
- Item: `KW-SWE-DISPLACEMENT-HISTORY-2026-08-30` in `backlog.md`

## Context

`ShearWaveElastography::generate_shear_wave` documents a history of displacement
fields, but returns `Vec<ElasticWaveField>`. Each retained field owns six dense
arrays: three displacement components and three velocity components. The
high-level harmonic-analysis and elasticity-reconstruction callers consume only
displacement. Retaining velocities therefore doubles the dominant history
payload without preserving a value those callers observe.

For `H` snapshots over `P = nx * ny * nz` grid points, the existing payload is
`H * 6 * P * size_of::<f64>()` bytes. A displacement-only history is
`H * 3 * P * size_of::<f64>()`. The bounded 16³, 41-snapshot workflow retains
7.6875 MiB of array payload through the existing type and 3.84375 MiB through
the displacement contract, excluding array and vector headers.

The solver also had two independent history loops: normal body-force propagation
cloned complete fields, while point-force propagation exposed an internal
observer used by elastic FWI. Adding another complete propagation loop for the
high-level path would duplicate snapshot cadence and final-state behavior.

## Decision

Publish `ElasticDisplacementSnapshot` as the timed displacement checkpoint:
`ux`, `uy`, `uz`, and `time`. It owns each component because every retained
snapshot must survive mutation of the solver's current field.

Replace the high-level `generate_shear_wave` result with
`Vec<ElasticDisplacementSnapshot>` and make `reconstruct_elasticity` consume that
history. Full `ElasticWaveField` histories remain available at the lower-level
solver boundary for consumers that require velocities.

Normal body-force propagation uses one generic, statically dispatched
saved-snapshot collector. The existing full-history path projects by cloning the
complete field; the displacement path projects only the three displacement
arrays and time. Both projections use the same initial, periodic, and final
snapshot decisions and the same integration loop. Header capacity is derived
and reserved before field or body-force allocation for either projected type.

The point-force observer remains separate because it records every completed
step and has a different force-application contract. Its existing internal
displacement snapshot type becomes the same public snapshot used by the normal
body-force path.

## Migration

| Previous surface | Migration |
| --- | --- |
| `generate_shear_wave(...) -> Vec<ElasticWaveField>` | Treat each returned item as `ElasticDisplacementSnapshot`; displacement fields and `time` keep their names and values, while velocity fields are intentionally absent. |
| `reconstruct_elasticity(&[ElasticWaveField])` | Pass `&[ElasticDisplacementSnapshot]`. |
| A high-level caller reading `vx`, `vy`, or `vz` | Use the lower-level `ElasticWaveSolver::propagate_waves_with_body_force_only_override` full-history contract. |

No version bump is part of this development increment. Release versioning must
classify the changed high-level signatures as a major contract change.

## Alternatives rejected

**Leave the existing method and add a displacement-named sibling.** The documented
high-level default would continue retaining unobserved velocities, and callers
would need to know which of two overlapping APIs is canonical.

**Populate velocity arrays with zeros.** This preserves the old type but corrupts
its value semantics and reports fabricated solver state.

**Retain only scalar displacement magnitude.** Harmonic analysis can use a scalar
series, but elasticity reconstruction and other vector-displacement consumers
require component direction. Removing those components would be a different
domain contract rather than a storage correction.

**Store snapshots as one pre-zeroed four-dimensional array per component.** This
reduces allocation count but adds full-history zero fill before the solver
overwrites every element. The measured issue is retained payload; contiguous
history storage remains a later layout experiment with its own bandwidth and
consumer-access evidence.

## Consequences

The high-level retained array payload is halved and no velocity array is
allocated for a retained high-level snapshot. Each displacement snapshot still
performs three necessary array allocations and copies; the propagated current
field and integration scratch remain unchanged.

This is a public breaking change. All first-party callers migrate in the same
series, and cargo-semver-checks must report only the recorded signature changes.
Differential tests compare every displacement component and timestamp with the
full-history route, including a nondivisible final snapshot. An allocation
census verifies the derived snapshot count, zero vector reallocations, and the
three-array-per-snapshot projection contract.
