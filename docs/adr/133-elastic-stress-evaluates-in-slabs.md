# 133. The elastic stress evaluation runs in slabs

Status: Proposed

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
