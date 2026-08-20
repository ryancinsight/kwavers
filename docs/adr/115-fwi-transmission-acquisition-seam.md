# 115. A transmission-acquisition seam for frequency-domain FWI

- Status: Accepted
- Date: 2026-08-19
- Item: `backlog.md#fwi-024-d` (atlas `backlog.md#atlas-usct-fwi-024`)
- Revised: 2026-08-19 — dispatch corrected from generic to `&dyn` (see "Dispatch"); numbered 115 after 113 and 114 were taken

## Context

Frequency-domain FWI takes `&MultiRowRingArray` in every entry point. That type
is not a parameter of the algorithm — it is welded into it. Thirty-six
references across all six modules of `inverse/fwi/frequency_domain`
(`forward`, `gradient`, `gauss_newton`, `operator`, `finite_window`,
`inversion`) reach through it, and what they consume is small and specific:

| call | uses | meaning to the algorithm |
| --- | --- | --- |
| `circumferential_elements()` | 5 | how many transmit events there are |
| `cylindrical_source(t)` | 7 | which elements fire on transmit `t` |
| `elements()` | 2 | where the receivers are |
| `element_count()` | 10 | how many receivers, i.e. the row width |

Nothing else about a ring is used. The physics loop in `predict_born_rows` is
already generic in everything but these four questions: it walks transmits,
builds an incident field from that transmit's sources, and evaluates a Green's
function at each receiver.

FWI-024-D asks for a different acquisition — two opposed linear arrays on a
rotation stage, swept through views — and it cannot be expressed through this
type at all. `cylindrical_source` indexes `elements[row * circumferential +
angular]`, a ring's own layout; there is no arrangement of a rotating linear
pair that answers it correctly.

## The constraint that decides the shape

A rotating stage moves the **receivers**, not only the sources. At view `v` both
arrays sit at a different angle, so every receiver coordinate differs from view
to view.

The current API cannot express that: `elements()` returns one fixed receiver set
for the whole acquisition, and `forward.rs` hoists it outside nothing — it
re-reads it inside the transmit loop, but always the same slice. A ring array
happens not to care, because rotational symmetry means the receiver set is
identical for every transmit. That coincidence is the only reason a fixed
receiver set has worked so far, and it is exactly what a rotation stage breaks.

So the seam must make receivers a function of the transmit, not a property of
the acquisition. The receiver *count* stays constant — the same two arrays are
present at every view — which preserves the `[transmissions, receivers]` matrix
shape that `Array2` and every observation comparison depend on.

## Decision

Introduce a `TransmissionAcquisition` trait in
`inverse/fwi/frequency_domain/acquisition.rs`, and take
`&dyn TransmissionAcquisition` wherever the six modules currently take
`&MultiRowRingArray` (see Dispatch below for why `dyn`):

```rust
pub trait TransmissionAcquisition {
    /// Number of transmit events in the acquisition.
    fn transmission_count(&self) -> usize;

    /// Receivers recorded per transmit. Constant across transmits: it is the
    /// row width of every observation matrix, and a ragged acquisition would
    /// make observed and predicted rows incomparable.
    fn receiver_count(&self) -> usize;

    /// Elements firing on `transmit`.
    fn sources(&self, transmit: usize) -> &[ElementPosition];

    /// Receiver positions for `transmit`.
    ///
    /// Indexed by transmit because a rotating stage moves the receivers with
    /// the sources. An acquisition whose receivers are fixed returns the same
    /// slice for every index.
    fn receivers(&self, transmit: usize) -> &[ElementPosition];
}
```

Borrowed slices rather than returned `Vec`s. `cylindrical_source` allocates a
fresh `Vec` on each of its seven call sites today, inside the transmit loop of a
solver that runs per frequency per iteration; a rotating acquisition can
precompute every view at construction, and the ring array can precompute its
per-transmit columns once. The seam should not institutionalise an allocation
the algorithm never needed.

`MultiRowRingArray` implements the trait, keeping its inherent methods, so no
caller outside FWI changes.

### Dispatch: `&dyn`, because the operator seam is already `dyn`

The acquisition is passed through `HelmholtzForwardOperator::predict_receiver_rows`,
and `Config` stores that operator as `Arc<dyn HelmholtzForwardOperator>`. A
generic method on a `dyn`-dispatched trait is not dyn-compatible, so
`predict_receiver_rows` cannot take `<A: TransmissionAcquisition>` without
either erasing the operator selection Config exists to carry, or making Config
itself generic in the acquisition — which would propagate a type parameter
through every FWI entry point, the Python bindings included, to remove one
pointer indirection per transmit.

So `TransmissionAcquisition` is designed dyn-compatible — no generic methods, no
`Self` returns, no associated types — and is passed as `&dyn TransmissionAcquisition`.

The cost is bounded by where the calls sit. `sources(t)` and `receivers(t)` are
called once per transmit, outside the receiver loop and far outside the voxel
loop underneath it; `receiver_count()` is called once per row allocation. The
per-element work — the Green's-function evaluation over every voxel for every
receiver — involves no trait call at all. A virtual call per transmit against a
volume integral per transmit is not a measurable ratio.

This corrects the first draft of this ADR, which specified `<A: TransmissionAcquisition>`
generically and argued monomorphisation. That argument was made without checking
how the operator is stored, and it is wrong for this seam: the free functions in
`forward` and `gradient` could take a generic parameter, but the trait method
they are reached through cannot, so a generic seam would have to be erased at
that boundary anyway. Choosing `&dyn` once, at the trait the design already
commits to, is the honest version of the same decision.

## Consequences

`predict_born_rows` becomes:

```rust
let mut output = Array2::zeros([transmissions, acquisition.receiver_count()]);
for transmit in 0..transmissions {
    let sources = acquisition.sources(transmit);
    let incident = incident_field(sources, &centers, reference_wavenumber, min_distance);
    for (receiver_index, &receiver) in acquisition.receivers(transmit).iter().enumerate() {
```

— a mechanical substitution, which is the point: the physics does not move, and
a diff that only renames the accessor is a diff whose correctness is checkable
by reading it.

`transmissions` stops being derived from `circumferential_elements()` at the
call site and becomes `transmission_count()`, removing the one place a caller
had to know a ring-specific fact to size the loop.

### What this does not do

It does not add the rotating acquisition. That is FWI-024-D's second increment,
and it needs its own decisions — per-view interpolation between a fixed
reconstruction grid and view-aligned simulation grids, and gradient accumulation
across views — none of which this trait constrains. This ADR only removes the
obstacle that made those undiscussable.

It does not touch time-domain FWI or the transducer crate's own array types.

## Verification

The seam is behaviour-preserving for the existing acquisition, so the oracle is
equality, not tolerance: `MultiRowRingArray` driven through
`TransmissionAcquisition` must produce **bitwise identical** predicted rows to
the current concrete path, on the same phantom, at every frequency. Evaluation
order is unchanged, so anything short of bitwise equality means the substitution
was not mechanical after all.

A second acquisition implementing the trait — a fixed opposed pair, without
rotation — is the cheapest evidence that the seam is genuinely general rather
than a rename of the ring's API. It is small enough to write in the same change
and would catch a trait whose contract still assumes rotational symmetry.

## Alternatives rejected

**Add rotation to `MultiRowRingArray`.** A ring that can also be a rotating
linear pair is two acquisitions in one type, selected by a flag — boolean
blindness over a domain distinction, and every FWI module would then carry the
branch.

**Take `&[ElementPosition]` plus a transmit-to-source index map.** This is the
seam with its structure flattened into two parallel arrays the caller must keep
consistent, and it still cannot express per-transmit receivers without a second
map. The trait states the same information with the invariant enforced.

**Defer the seam and write the rotating acquisition against a copy of the FWI
modules.** A second copy of the solver per acquisition geometry is the cloned-
algorithm defect this repository's standards prohibit outright, and it would
double the surface every subsequent FWI change has to touch.
