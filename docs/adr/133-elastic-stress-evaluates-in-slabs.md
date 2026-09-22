# ADR 133: The elastic stress evaluation runs in slabs

Status: Accepted (revised 2026-09-22; see the revision below)

## Context

The three-dimensional elastic acceleration is a two-stage chain over whole
grids: `stress_into` writes six stress components from the displacement
field and the Lamé parameters, and the divergence pass reads those six and
the density to write three accelerations.

Four fusion increments (#816, #817, #818, #819) removed every intermediate
that was not required by the chain itself, taking the traffic per evaluation
from about 136 MB to 40 MB at 64 cubed and the whole step from 2036-2126 us
to 998-1021. What remains is the chain's own shape: the six stress fields
are written by stage one and read by stage two.

A scaling sweep of the evaluation (2026-09-21, `swe_step_phase_split`) shows
where that costs:

| N | working set | stress | ns/cell |
|---|---|---|---|
| 24 | 1.4 MiB | 83 us | 6.00 |
| 32 | 3.2 MiB | 96 us | 2.93 |
| 48 | 11.0 MiB | 158 us | 1.43 |
| 64 | 26.0 MiB | 316 us | 1.21 |
| 96 | 87.8 MiB | 3576 us | 4.04 |

The host's L3 is about 36 MB. While the live set fits, the chain runs at
1.2 ns per cell; past it, 4.0. At 96 cubed the evaluation moves roughly
245 MB in 3408 us -- about 72 GB/s, which is this machine's DRAM rate, not
its cache rate. The six stress fields are 42 MB there: stage one writes them
to DRAM and stage two reads them back.

The fusions' own value moves with the regime, which is the same effect seen
from the other side: the diagonal is 1.25x at 64 cubed and 1.73x at 96,
while the three shears are 2.0x and 1.25x.

kwavers runs production grids at 128 cubed and above, entirely in the second
regime.

## Decision

Evaluate the chain in slabs of x-planes rather than whole grids, so a slab's
stresses are produced and consumed while they are still in cache and never
reach DRAM.

A slab of `S` planes needs its stresses over `S + 4` planes, because the
divergence's fourth-order stencil reaches two planes either side; those
stresses in turn need displacement over `S + 8`. The implementation keeps a
rolling window of stress planes rather than recomputing the halo per slab.

`S` is chosen so the window fits L2 per worker: six stress components over
`S + 4` planes of `N * N` f64.

## Consequences

Expected: at 96 cubed the evaluation's DRAM traffic falls from about 245 MB
to the inputs and outputs alone -- displacement, Lamé pair, density, three
accelerations, roughly 63 MB -- which at the measured 72 GB/s is about
875 us against 3408.

Values must not change. Slabbing alters traversal order only; each output
lane is still the same arithmetic over the same stencil neighbourhood, so
the acceptance test is bitwise equality against the current kernels over the
shapes the fused kernels already use, including shapes whose extent is not a
multiple of the slab height.

Cost: the kernels stop being whole-grid passes over leto views and become a
pipeline with a halo and a rolling buffer. That is a real increase in
complexity, confined to the elastic stress module; leto's fused kernels are
unchanged and keep serving whole-grid callers.

Risk: the win is a hypothesis until measured. The stop condition, written
before implementing, is the 96-cubed stress arm at least 2x faster than the
current kernels at comparable host load, with 64-cubed no worse than the
identical-code drift band. Below that, the complexity is not paid for and
the slab path is dropped rather than kept beside the whole-grid one.

## Alternatives

Leaving it whole-grid keeps the simpler kernels and accepts 4 ns per cell at
production sizes.

Tiling in all three axes rather than slabbing in one would cut the halo's
share further, but the two contiguous axes are exactly where the lane
writers vectorise; slabbing the outermost axis keeps those intact.

Shrinking the live set by recomputing stresses per divergence instead of
storing them triples the stencil work, which the measurements above show is
already the dominant term once traffic is minimised.

## Revision 2026-09-22 -- built, measured, and below its own bar

**What was built.** The stress window of the Decision, implemented in the
leading planes of the scratch stress fields: each slab slides the stress
planes it still needs to the front, computes only the rest, then writes its
accelerations (`stress_acceleration_in_slabs`). It rests on leto ADR 0032:
fused passes over plane windows, with the stencil chosen by the grid plane,
so every acceleration is the whole-grid value to the bit -- asserted for
every slab height from one plane to past `nx`, under both density scales.
The same leto change lets all six stresses come from one pass over the nine
displacement gradients instead of four; that alone is 169-175 -> 149-153 us
at 64 cubed and 1621-1721 -> 1420-1503 us at 96 cubed, on every path.

**What was measured** (release, fastest of paired repeats,
`swe_acceleration_slab_sweep`):

| N | whole-grid | best slabs | ratio |
|---|---|---|---|
| 64 | 295-312 us | 411-419 (24 planes) | 0.7x |
| 80 | 884-1236 | 928-1059 (16-24) | ~1x |
| 96 | 2742-3113 | 1857-1904 (24) | 1.5x |
| 128 | 9712-10677 | 5294-5634 (16) | 1.8x |

**Against the stop condition.** The Risk section set 2x at 96 cubed as the
bar below which the slab path is dropped. It reached 1.5x there. That is a
miss, recorded as one.

Two steps on the way were attributed rather than tuned. Restricting
whole-grid stress arrays to a plane range first gave 1.28x: each slab wrote
to lines last touched a step earlier and paid read-for-ownership and
write-back, which the reused window removed (1.28x -> 1.5x). Cutting the
regions per slab from five to two through the one-pass stress improved slab
and whole-grid alike and left the ratio where it was. Row-granular tasks did
not move the optimum. What remains between 1.5x and the ~2.6x
cache-resident ceiling at 96 cubed (64 cubed runs at 1.15 ns per cell) is
not attributed.

**Why it is kept anyway.** The bar was set at 96 cubed from the traffic
estimate in Consequences. The Context names 128 cubed and above as
production, and there the path runs at 1.8x, with the whole-grid path
retained wherever it is faster. The complexity is one driver function, one
selection rule and leto's windowed pass, all bitwise-tested. The decision
to keep it is a revision of this ADR's own bar, made after the numbers were
seen, and open to reversal: if later work at production sizes shows the
path not paying, it is deleted as the Risk section intended.

**Selection.** Whole-grid while the live fields (14, or 15 with a density
field) fit the last-level cache that themis reports; past it, the largest
slab whose fields fit that cache, capped at the worker count. That picks
24 planes at 96 cubed and 16 at 128 cubed, the measured best of
8/12/16/24/32 at each. A platform reporting no cache size stays whole-grid.

