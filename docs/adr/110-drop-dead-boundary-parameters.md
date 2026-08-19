# 110. Delete the inert BoundaryParameters surface

- Status: Accepted
- Date: 2026-08-17
- Item: `backlog.md#kw-sim-101`

## Context

`SimulationBuilder::boundary(BoundaryParameters { .. })` accepts three settings —
per-face `boundary_types`, `pml_thickness`, and `pml_alpha`. It validates them,
stores them on `SimulationConfiguration`, and **nothing ever reads a field**.
Repository-wide, every reference is definition, re-export, the builder setter,
the struct field, one `validate()` call, and a test constructing the default:

```
kwavers-boundary/src/config.rs      definition + Default + validate
kwavers-boundary/src/lib.rs:81      pub use
kwavers-simulation/src/builder.rs   self.config.boundary = params
kwavers-simulation/src/configuration.rs  field, validate, test default
```

`SolverBoundaryKind` exists only to type `boundary_types`, and is likewise
referenced nowhere but its own definition and that re-export.

Boundaries are really configured through a different path. `PmlConfig`
(kwavers-simulation `configs.rs`) carries `size`, `size_xyz`, `inside`, and
`alpha_xyz`, and `dispatch/{fdtd,pstd}.rs` turns it into a `CPMLConfig` and calls
`enable_cpml`. That path is live, tested, and complete: `alpha_xyz: None` means
"use the CPML default", `alpha_is_zero()` selects a transparent boundary, and
`size_xyz` gives per-dimension thickness.

So `BoundaryParameters` is not an unfinished feature. It is a superseded one that
`PmlConfig` replaced without its predecessor being removed.

The cost of keeping it is not neutral. A caller who sets `pml_thickness: 30`
gets a silently ignored setting and a boundary that is not the one they asked
for — the failure mode is a wrong answer, not an error. This is the same defect
class as `CPMLConfig::target_reflection` (KW-BND-099), which reads like a control
and is only an estimator input; there the field had a real remaining use, so it
was documented and pinned by test. Here there is no remaining use at all.

## Decision

Delete `BoundaryParameters`, `SolverBoundaryKind`, the `pub use` in
`kwavers-boundary/src/lib.rs`, `SimulationBuilder::boundary`, and the
`SimulationConfiguration::boundary` field.

`PmlConfig` is the single boundary-configuration surface. Callers that were
passing `BoundaryParameters` were not getting the behaviour they configured, so
the migration is to express the intent through `PmlConfig`:

| removed | replacement |
|---|---|
| `pml_thickness: n` | `PmlConfig::with_size(n)` |
| `pml_alpha: a` | `PmlConfig::with_alpha(a)` |
| `boundary_types: [PML; 6]` | the default; CPML is applied when a PML is configured |
| `boundary_types` with a non-PML face | unsupported before and after — it was never applied |

## Alternatives rejected

**Wire it up.** This would mean deciding what per-face boundary kinds mean for a
`CPMLConfig` that only carries per-dimension thickness — new capability, designed
backwards from a struct nobody chose. `PmlConfig` already covers the settings
that have consumers, and duplicating them behind a second surface is the
consolidation defect.

**Document it as inert.** What was done for `target_reflection`, and right there
because the field still feeds `theoretical_reflection`. Nothing here has a use to
preserve, so documenting would leave a knob whose only purpose is to be ignored.

**Deprecate and keep.** A `#[deprecated]` re-export is the compatibility shim the
integrity rules exclude: the callers are in this workspace and are updated in the
same change.

## Consequences

Breaking for any external caller of `SimulationBuilder::boundary` or the two
types — hence `[major]`. No in-repo call sites construct `BoundaryParameters`
outside the one test, so nothing loses behaviour: the settings were already not
being applied. The CHANGELOG records the removal with the mapping above.
