# 130. Exterior coupling carries its body mask

Status: Accepted

Board item: [KW-SOURCE-DOMAIN-OPTIONAL-MASK-2026-09-08](../../backlog.md#kw-source-domain-optional-mask-2026-09-08)

## Context

`SourceDomain` is a fieldless two-variant enum on `Nonlinear3dAperture`, and the
CT-derived body mask travels beside it as a separate `Option<&[bool]>` through
`ForwardInput`, `build_source_plan`, and the stencil functions. The two are
mutually dependent: exterior coupling deposits through the mask, a
tissue-boundary source has none. Nothing in the types says so, so the
combination is checked at runtime and panics twice:

- `forward/source/plan.rs:34` — `expect("exterior coupling sources require the
  CT-derived body mask")`
- `forward/source/stencil.rs:67` — `expect("exterior coupling source stencil
  requires body mask")`

The second is not recorded on the board item, which names only the first.

Two facts constrain the fix. Every production construction site passes
`Some(...)` — `optimization/objective.rs`, `steering.rs`,
`westervelt/calibration.rs`, and `westervelt/fwi.rs` (twice) — so the `Option`
models test convenience rather than a domain state, and the panic guards a
combination production never produces. And `build_aperture` already holds
`volume.body_mask` and already returns `Result`, so the aperture's construction
site is where the domain and the mask are both in scope.

## Decision

Give the variant its data, owned:

```rust
pub(crate) enum SourceDomain {
    TissueBoundary,
    ExteriorCoupling { body_mask: Vec<bool> },
}
```

`build_aperture` fills it from `volume.body_mask` in the same match that already
selects the variant from `volume.anatomy`. `ForwardInput::source_body_mask`
disappears; `build_source_plan` and `finite_source_stencil` match on the domain
and the exterior arm receives `&[bool]`, not `Option<&[bool]>`. Both `expect`s
go, and no runtime check replaces them — the invalid combination stops being
representable.

## Alternatives rejected

**A borrowed mask in the variant** (`ExteriorCoupling { body_mask: &'m [bool] }`)
avoids the allocation but adds a lifetime parameter to `Nonlinear3dAperture`,
which is `Clone + Debug`, stored behind `&'a` in `ForwardInput`, and named in 15
files. The aperture is control-plane configuration, and the lifetime-placement
rule puts borrows on the data plane: a struct-stored borrow propagates its
lifetime into every containing type, which is a cost worth paying for zero-copy
throughput and not for one allocation per aperture.

**A validating boundary** that converts `(SourceDomain, Option<&[bool]>)` into a
total `SourceGeometry`, returning a typed error, keeps the aperture owned but
has to run somewhere. The `ForwardInput` construction sites sit inside
infallible functions — `calibration_measurement -> CalibrationMeasurement`,
`calibrated_source_scale -> f64`, and the FWI internals — so the error would
ripple through several layers that have no other reason to be fallible. It also
leaves the discriminant in two places, free to disagree.

**A `SourceGeometry` field on `ForwardInput`, built by callers**, needs no
fallibility and no lifetime on the aperture, but keeps `aperture.source_domain`
as a second discriminant that a caller can contradict silently — trading one
unrepresentable-state defect for another.

## Consequences

The aperture owns `n³` bools for the two exterior anatomies (Liver, Kidney);
Brain carries none. `Nonlinear3dAperture: Clone` therefore clones the mask, so
cloning an aperture stops being free — it is built once per run and passed by
reference, so no current call site pays this per step.

`adjoint.rs` already declares the mask non-optional (`source_body_mask: &'a
[bool]`) and wraps it in `Some` on the way down; that wrapping disappears.

## Verification

The existing suites are the oracle: 59 `nonlinear3d` tests in `kwavers-therapy`
and the `forward/source` plan tests, whose assertions are unchanged by this
refactor — only their construction syntax moves. The `expect` sites are gone,
which `git grep` confirms, and the exterior stencil path is exercised by
`forward/source/tests.rs`'s `ExteriorCoupling` fixture.
