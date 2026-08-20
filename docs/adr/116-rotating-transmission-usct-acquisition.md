# 116. Rotating the apparatus, not the model, for transmission USCT

- Status: Accepted
- Date: 2026-08-20
- Item: `backlog.md#fwi-024-d` (atlas `backlog.md#atlas-usct-fwi-024`)
- Follows: [115](115-fwi-transmission-acquisition-seam.md)

## Context

ADR 115 gave frequency-domain FWI an acquisition seam, so a rotating
transmission-USCT acquisition is now expressible. It did not settle *how* the
rotation is represented, and there are two routes with materially different
consequences for the gradient.

**Rotate the apparatus.** One fixed grid; the acquisition reports rotated
element positions per view. No resampling anywhere. This is what
`TransmissionAcquisition` already expresses — a rotating implementor is a few
dozen lines.

**Rotate the model.** Resample the slowness volume into each view's frame,
simulate on an axis-aligned grid, resample the gradient back. This is what the
board item originally assumed ("per-view interpolation between a fixed
reconstruction grid and view-aligned simulation grids").

The obstacle that makes this a decision rather than an obvious choice: rotating
the apparatus puts elements off the grid nodes, and the PSTD-discretized paths
currently refuse that.

## What refuses off-node elements, and what does not

The split follows the discretization, not the operator family:

| path | projection | off-node |
| --- | --- | --- |
| single-scatter Born | continuous Green's functions, no grid projection | accepted |
| CBS `DenseFreeSpace`, `SpectralPeriodic` | `source_density_from_bli` / `sample_field_with_bli` | accepted |
| CBS `SpectralPstdPeriodic` | `sample_array_on_grid` → `exact_grid_index` | **refused** |
| PSTD finite-window | `source_spectrum_on_grid`, `receiver_indices_on_grid` → `exact_grid_index` | **refused** |

`exact_axis_index` rejects any coordinate more than `1e-9` off a node. A linear
array rotated by anything but a multiple of 90° lands off-node on essentially
every element, so both PSTD paths would reject every view. They fail loudly
rather than silently mis-placing elements, which is the good version of this
problem, but it is a hard refusal.

That the refusal tracks the PSTD discretization is not incidental: those paths
inject and sample on the spectral grid, where a node index is the natural
address. The band-limited stencil is the standard way to give such a scheme a
continuous address, which is why the sibling CBS variants already use it.

## Decision

**Rotate the apparatus, and extend the PSTD paths to band-limited projection**
so every operator accepts off-node elements.

The deciding argument is where the interpolation error lands. Rotating the model
resamples the slowness volume forward and the gradient back, once per view, on
every iteration — so interpolation error sits *inside the quantity the inversion
descends on*. That error is not noise: it correlates with view angle, so it
biases the reconstruction in exactly the angular pattern a USCT sweep is trying
to resolve, and repeated resampling compounds it across iterations. Rotating the
apparatus keeps model and gradient on one grid for the entire inversion; the
only interpolation is in projecting sources and receivers, which is where this
codebase already puts it for the non-PSTD operators.

It also matches the physics. The stage rotates the transducers; the breast does
not rotate. A formulation whose state variable moves when the apparatus moves is
a computational convenience, and it introduces an artifact the physical
experiment does not have.

### The extension is strictly additive, not a change of behaviour

`bli_weights` short-circuits on-node axes: `on_grid_axes` marks an axis whose
offset is below `spacing × 1e-3`, and the stencil loop then keeps only the
zero-offset term on that axis. The kernel is `sinc(π·Δ/h)` per axis, so an
on-node point yields a single contribution of weight `sinc(0) = 1`.

Every geometry the finite-window path accepts today is on-node within `1e-9`
relative — that is precisely what `exact_axis_index` enforces. At that offset
the sinc weight is `1 − (π·10⁻⁹)²/6 ≈ 1 − 1.6·10⁻¹⁸`, which is `1.0` exactly in
`f64`. So for every currently-supported acquisition, BLI projection reduces to
the existing single-node injection **bitwise**, and the change only extends the
accepted domain. That is the oracle: existing finite-window results must not
move at all, and a test pins it rather than assuming it.

### Adjoint consistency comes from reusing one weight function

The risk in BLI projection is a forward that spreads with one stencil and an
adjoint that gathers with another; the gradient is then inconsistent with the
forward operator, and finite-difference gradient checks fail in a way that looks
like a physics bug.

This crate already avoids that structurally: `sample_field_with_bli` and
`receiver_adjoint_from_bli` both call `nonempty_bli_weights` and differ only in
gather versus scatter. The PSTD extension reuses the same function rather than
deriving a second stencil, so the adjoint is the transpose by construction.

The check already exists and already covers both paths being changed.
`tests/gradient_fd.rs` compares the analytic adjoint gradient against a central
difference at 5·10⁻⁴ relative, with a case per operator — including
`pstd_spectral_cbs_adjoint_gradient_matches_finite_difference` and
`pstd_finite_window_born_adjoint_gradient_matches_finite_difference`. A stencil
mismatch introduced by this extension fails there, on the exact two operators it
touches, without a new test having to be designed for the purpose.

## Consequences

Increment 2 becomes two pieces, in order:

1. Extend the four on-node projections — `source_spectrum_on_grid` and
   `receiver_indices_on_grid` in the finite-window path, `sample_array_on_grid`
   and its adjoint in the CBS PSTD variant — to BLI, reusing
   `nonempty_bli_weights`. Oracle: on-node geometries
   produce bitwise-identical results; finite-difference gradient checks still
   pass with the off-node path active.
2. A `RotatingLinearPair` acquisition — two opposed linear arrays swept through
   views — implementing `TransmissionAcquisition`. Oracle: a 360° sweep recovers
   a known sound-speed phantom within a derived tolerance, and a view at 0°
   reproduces the un-rotated acquisition exactly.

Gradient accumulation across views needs no special machinery: views are
transmits, and the existing gradient already accumulates over transmits.

### What this rejects

**Rotate the model per view.** Rejected for putting angle-correlated
interpolation error inside the gradient, as argued above. It remains the only
route if the BLI extension turns out to break adjoint consistency in the PSTD
path — that is the falsifying evidence, and it is checkable by the existing
finite-difference gradient test before any of increment 2's second piece is
written.

**Restrict the sweep to multiples of 90°.** Keeps every element on-node and
needs no change at all, but a four-view sweep does not reconstruct anything
useful; this trades the entire capability for the convenience.

**Leave the PSTD paths refusing rotated acquisitions.** Born and the non-PSTD
CBS variants would carry the rotating acquisition and PSTD would not. A seam whose
implementors are silently valid for some operators and rejected by others is the
kind of partial contract that surfaces as a confusing runtime error long after
the configuration was chosen.
