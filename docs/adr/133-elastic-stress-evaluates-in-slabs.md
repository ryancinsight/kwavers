# ADR 133: The elastic acceleration evaluates in slabs past the cache

Status: Accepted (revised 2026-09-22; see the revision notes at the end)

## Context

The three-dimensional elastic acceleration is a two-stage chain: six stress
components from the displacement gradients and the Lamé parameters, then
their divergence, scaled by the density, into three accelerations.

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

While the live set is cache-resident the chain runs at 1.2 ns per cell;
past it, 4.0. At 96 cubed the evaluation moves roughly 245 MB in 3408 us --
about 72 GB/s, this machine's DRAM rate. The six stress fields are 42 MB
there: stage one writes them to DRAM and stage two reads them back. kwavers
runs production grids at 128 cubed and above, entirely in that regime.

## Decision

1. **Slabs through a stress window.** Past the caches the chain runs a slab
   of x-planes at a time. A slab of `S` planes needs its stresses over
   `S + 4` planes, since the divergence's fourth-order stencil reaches two
   planes either side. Those planes live in a window -- the leading planes of
   the scratch stress fields, reused for every slab -- so they are written
   and read back while resident. Each slab slides the planes it shares with
   the previous one to the front and computes only the rest, so no stress
   plane is computed twice (`stress_acceleration_in_slabs`). It rests on
   leto ADR 0032: fused passes over plane windows that take the stencil the
   grid plane gives, so every acceleration is the whole-grid value to the
   bit.
2. **Selection** (`slab_height`). Whole-grid while the live fields -- 14,
   or 15 with a density field -- fit every cache level past the first,
   private and shared, summed (`cache_capacity_bytes`). Past that, the
   largest slab whose window and the planes around it fit the shared last
   level (`last_level_cache_bytes`), less the four planes the stencil
   reaches, capped at the worker count, since each pass hands one plane to
   a task. A platform reporting no caches stays whole-grid, as does scratch
   whose stress fields are not C-contiguous, since the window slides its
   planes as contiguous runs.
3. **One stress pass.** All six stresses come from one pass over the nine
   displacement gradients, where four passes -- the diagonal, then a shear
   at a time -- read the displacement nine times.

## Consequences

Measured (release, fastest of paired repeats, `swe_acceleration_slab_sweep`,
a 285K with 40 MiB of second-level caches, a 36 MiB last level and 24
workers):

| N | live set | whole-grid | best slabs | ratio |
|---|---|---|---|---|
| 64 | 29 MB | 295-312 us | 411-419 (24 planes) | 0.7x |
| 72 | 42 MB | 415-436 | 644-948 | < 1 |
| 76 | 49 MB | 657-701 | 830-923 | < 1 |
| 80 | 57 MB | 884-1236 | 928-1059 (16-32) | ~1x |
| 96 | 99 MB | 2742-3113 | 1857-1904 (24) | 1.5x |
| 128 | 235 MB | 9712-10677 | 5294-5634 (16) | 1.8x |

At 72 and 76 cubed whole-grid won 29 of the 30 paired comparisons across
three runs (on a host at 35-60% load from other processes, which the
pairing absorbs and the absolute times do not), although the live set is
past the last level: the private caches hold the rest. The selection
therefore compares the live set with both levels together, which routes 72
to 80 cubed whole-grid and 96 cubed and above to slabs. The slab height the rule gives, 24 planes at 96 cubed
and 16 at 128, is the measured best of 8, 12, 16, 24 and 32 at each; which
of the rule's two limits binds is what moves the optimum between them.

The one stress pass is 169-175 -> 149-153 us at 64 cubed and 1621-1721 ->
1420-1503 us at 96 cubed, on every path.

Values do not change. Slabbing alters which planes are in flight, not the
arithmetic: every acceleration is asserted equal to the whole-grid value
for every slab height from one plane to past the plane count, under both
density scales, on grids whose plane count no slab height divides.

After a slabbed evaluation the scratch stress fields hold the last slab's
window, not the grid's stress; `ElasticStepScratch` documents it.

Cost: one driver function, one selection rule and leto's windowed pass,
confined to the elastic stress module (`stress/slabs.rs`).

## Alternatives

- **Whole-grid only** keeps the simpler kernel and accepts 4 ns per cell at
  production sizes.
- **Plane ranges over whole-grid stress arrays**, without a window. Measured
  first: 1.28x at 96 cubed. Every line a slab wrote was last touched a step
  earlier and paid a read for ownership and a write-back.
- **Tiling all three axes.** The two contiguous axes are where the lane
  writers vectorise; slabbing the outermost keeps them intact.
- **Recomputing stresses per divergence** instead of storing them triples
  the stencil work, which is the dominant term once traffic is minimised.
- **Selecting against the last level alone.** Routes 70 to 79 cubed to
  slabs, where whole-grid is measured faster.

## Revision notes

**2026-09-22 -- built, measured, below its own bar.** The Decision first
chose the slab height to fit L2 per worker and set a stop condition: 96
cubed at least 2x faster, or the slab path is dropped. Built through the
reused window it reached 1.5x there, a miss recorded as one. It is kept on
the production sizes the Context names: 1.8x at 128 cubed, with whole-grid
wherever it is faster. What separates 1.5x from the ~2.6x cache-resident
ceiling at 96 cubed is unattributed. The slab height became the last-level
rule above, the measured best at 96 and 128 cubed.

**2026-09-22 -- selection against both cache levels.** Review found the
last-level threshold sent 70 to 79 cubed to slabs without measurement. The
sweep then measured whole-grid ahead at 72 and 76 cubed and level at 80, so
the threshold compares the live set with the private and shared levels
together. On a hierarchy whose last level duplicates the private levels
that sum overstates the capacity and keeps a band of sizes whole-grid where
slabs would win -- slower, never different in value.
