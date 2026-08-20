# 113. Finite-aperture diffraction in RF synthesis: an injected SIR seam

- Status: Accepted
- Date: 2026-08-19
- Item: `backlog.md#cov-4`

## Context

COV-4 asks for "discrete point-scatterer + spatial-impulse-response RF
synthesis" and calls it the largest remaining gap. Most of it is already built,
and both halves are tested:

- `kwavers-phantom::scatterers` — `PointScatterer`, `ScattererCloud`,
  `synthesize_rf`, `synthesize_rf_with_transmit`, `TransmitWavefront`, optional
  power-law attenuation.
- `kwavers-physics::analytical::transducer::spatial_impulse_response` —
  `CircularPistonSir` and `RectangularPistonSir`, the Tupholme–Stepanishen
  closed forms.

What is missing is the coupling, and the scatterers module already says so:

> It does **not** model finite-aperture diffraction — the Tupholme–Stepanishen
> spatial impulse response that Field II convolves in for extended elements.

`synthesize_rf` uses the monostatic point-element model
`RF_e(t) = Σ_s (a_s/r²)·pulse(t − 2r/c)`, exact for point elements and
far-field scatterers. Field II's refinement replaces the bare `pulse` with
`pulse ⊛ h_tx ⊛ h_rx`, the round-trip SIR for that element–scatterer geometry.

Three facts shape the decision.

**The crates do not know about each other.** `kwavers-phantom` depends on
`kwavers-core`, `kwavers-grid`, and `kwavers-medium` — not on `kwavers-physics`.
`kwavers-physics` does not reference phantom either. Nothing but the `kwavers`
facade depends on phantom, while six crates depend on physics. So the coupling
cannot be written today without deciding how the two meet.

**The element description cannot express an aperture.** `synthesize_rf` takes
`element_positions: &[[f64; 3]]`. An SIR needs the aperture's size *and* the
field point in the element's own frame — `(r, z)` for a circular piston. Points
alone cannot carry that, and for a curved array the orientation differs per
element (`ConvexArrayGeometry::element_normal`, ADR 112).

**One aperture already has the round trip, with an exact oracle.**
`CircularPistonSir::round_trip_response(r, z, dt, n)` returns the discretely
auto-convolved two-way kernel, and its normalization is closed-form:
`Σ_k out[k]·dt = (√(z²+a²) − z)²` on axis. `RectangularPistonSir` exposes
`evaluate` and the arrival times but no round-trip helper.

## Decision

**Inject the SIR through a seam that `kwavers-phantom` defines, rather than
adding a phantom → physics dependency.** Phantom declares what it needs — a
kernel for an element–scatterer geometry — and the caller supplies an
implementation. `kwavers-physics` keeps the closed forms; phantom keeps its
current three dependencies; the facade or a consumer wires them.

The seam takes the element's aperture frame, not just its position: a new
element description carrying position and outward normal, so `(r, z)` is
derivable per element. This is the same shape ADR 112 settled for the convex
array, and `ConvexArrayGeometry` already produces exactly those two vectors.

**Scope this increment to the circular piston.** It is the aperture with a
round-trip helper and an exact normalization identity to verify against.
Rectangular follows once its round trip exists, and the seam is what makes that
a later addition rather than a rewrite.

**Verification is the normalization identity, not a stored trace.** For an
on-axis field point the two-way kernel integrates to `(√(z²+a²) − z)²`
analytically, so the coupled path can be checked against a closed form rather
than a previous run. The point-element limit is the second check: as the
aperture radius goes to zero the SIR-coupled RF must converge on the existing
`synthesize_rf` output, so the refinement provably reduces to what it refines.

## Alternatives rejected

**Add a `kwavers-phantom → kwavers-physics` dependency.** It creates no cycle
and would work. It is rejected because it inverts the useful direction: phantom
is a leaf that only the facade consumes, and giving it an edge to a crate six
others depend on couples a tissue model to an analytical-physics crate for one
kernel. The seam gets the same capability without the edge.

**Put the coupled synthesis in the `kwavers` facade**, the only crate that
already depends on both. The facade is meant to be thin, and this is physics.

**Move the SIR down into a crate phantom already depends on.** Relocating
analytical transducer physics into `core`, `grid`, or `medium` to satisfy one
consumer misplaces it; six crates consume it where it is.

**Extend `RfSynthesisConfig` with a single aperture for all elements.**
Simplest, and wrong for the curved arrays ADR 112 just wired: element normals
differ across a convex array, so a per-array aperture cannot express the
geometry the SIR needs.

## Consequences

`synthesize_rf` keeps its current signature and meaning — the point-element
model stays the default and stays exact in its own limit. The SIR-coupled path
is an additional entry point taking aperture-aware elements plus a kernel
provider.

The cost is real and belongs to the caller: a kernel per element–scatterer pair
is far more work than one scaled pulse, and the kernel depends only on `(r, z)`,
so the seam must let a provider cache or quantize rather than forcing a fresh
evaluation per pair. That is a reason the provider is injected rather than
called directly.
