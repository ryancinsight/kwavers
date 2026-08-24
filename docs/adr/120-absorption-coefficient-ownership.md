# ADR 120 — Who owns the power-law absorption coefficient

- **Status:** Accepted
- **Board item:** `KW-ABSORPTION-CONFIG-PRECEDENCE`
- **Class:** [major]
- **Date:** 2026-08-21 (accepted and implemented 2026-08-22)

## Context

Three places in the pseudospectral path decide what absorption coefficient a
simulation runs at, and two of them disagree.

**The callers resolve explicit-wins.** `kwavers-python`'s
`simulation_py/solvers/pstd.rs:145` and `kwavers-simulation`'s
`dispatch/pstd.rs:256` both compute an `effective_alpha_db`: the user's
coefficient when it is positive, otherwise `medium.alpha_coefficient(0, 0, 0,
grid)`. They then place that resolved value in
`PSTDConfig::absorption_mode`, which reads as an instruction to the solver.

**The solver resolves medium-wins.** `initialize_absorption_operators`
(`pstd/physics/absorption/init.rs:185`) reads the medium's coefficient per voxel
and uses the config's value only where the medium reports exactly zero.

**The default makes the disagreement bite.** `HomogeneousMedium::new` seeds
`absorption_alpha` with `WATER_ABSORPTION_ALPHA_0`, which is never zero. So on
the most common medium the config coefficient is unreachable: a caller asking for
`40 dB/(MHz^1.5 cm)` gets water's coefficient, and the result is
indistinguishable from a lossless run at the durations the k-Wave reference cases
use. Substituting `4000` changes the output by nothing to six significant
figures.

This is user-visible. A Python caller passing `alpha_coeff_db = 0.75` on a
homogeneous medium has their value resolved correctly by the binding, written
into the config, and then silently discarded by the solver.

### How it surfaced

The k-Wave absorbing differential case `ivp_absorbing_2d` (ADR 119) measured
`r = 0.836` against the reference. Setting the coefficient through
`HomogeneousMedium::set_acoustic_properties` instead gives `r = 0.999999924` at
relative L2 `8.10e-3`. The absorption physics is correct; only its configuration
route is not.

The existing absorption unit tests could not catch this. They construct a solver
and call `apply_absorption_to_pressure` on a config-derived kernel directly, so
they never exercise the precedence, and none of them compares an absorbing run to
a lossless one.

### Why the medium-wins rule exists

It is load-bearing for heterogeneous media. Absorption genuinely varies per voxel
in tissue, and `init`'s per-voxel read is the only place that variation survives:
both callers flatten the medium to a **single sample at the origin** before
building the config, so a heterogeneous medium's spatial structure reaches the
solver exclusively through the path that overrides the config.

The two rules are therefore each correct for one medium class and wrong for the
other, and nothing in the types distinguishes the classes.

### The exponent has the same defect, and that changes the fix

Implementation surfaced a second half the filing missed. `alpha_power` was
resolved by the identical pattern, against its own sentinel: the medium's value
won unless it was `0.0` or `1.0`. `HomogeneousMedium::new` seeds `1.05`, which
is neither, so a caller asking for `y = 1.5` got `1.05`.

That is not merely the same bug twice. Absorption is `alpha(f) = alpha_0 f^y`,
and the old code took `alpha_0` from one owner and `y` from another whenever
they disagreed — evaluating neither party's power law. Fixing only the
coefficient left the absorbing reference case at `r = 0.838`, indistinguishable
from the original defect, because the exponent was still the medium's.

The two parameters are therefore resolved **together, from one owner**. A caller
requesting absorption specifies the whole power law, not half of it.

## Decision

Make the distinction the code is missing explicit: separate *"the caller
requested this coefficient"* from *"the medium happens to report this
coefficient"*, which today are the same `f64`.

**Recommended option — an explicit override, per-voxel medium otherwise.**

- `AbsorptionMode::PowerLaw` carries `alpha_coeff: Option<f64>`. `Some` is an
  explicit request and applies uniformly, and its `alpha_power` is authoritative
  with it; `None` means the medium owns both parameters and `init` reads them
  per voxel, preserving heterogeneity.
- Both callers stop pre-resolving. They pass `Some(user_value)` when the user
  supplied one and `None` otherwise, deleting the duplicated resolution at each
  site. Resolution then happens once, in `init`, which is the only place that can
  see whether the medium varies.
- `HomogeneousMedium::new`'s water default stays — it is a reasonable default for
  an unconfigured medium — but it stops outranking an explicit request, because
  `Some`/`None` now distinguishes the two cases that `> 0.0` could not.

This is [major]: `AbsorptionMode` is public and the variant's shape changes. In-repo
callers migrate in the same change; there is no compatibility shim.

## Alternatives rejected

- **Config always wins.** Simplest, and it matches what the callers already
  believe. Rejected: it silently flattens heterogeneous absorption to the origin
  sample, turning a correctness bug for homogeneous media into a correctness bug
  for tissue.
- **Medium always wins; delete `alpha_coeff` from the variant.** Honest about
  today's behaviour and matches k-Wave, which puts absorption on the medium.
  Rejected on its own, because it leaves callers no way to request a coefficient
  without mutating the medium, and `HomogeneousMedium::new` gives no constructor
  argument for one — the caller would have to construct then mutate, which is the
  temporal coupling the standards forbid. Worth revisiting together with a
  medium constructor that takes absorption.
- **Sentinel zero means "unset".** What the code does today. Rejected: it cannot
  express "deliberately lossless" distinctly from "unset", which is exactly the
  ambiguity that produced this defect.

## Verification

The absorbing k-Wave reference case (ADR 119) is the regression test. It now
carries the coefficient **only** in `PSTDConfig::absorption_mode` and leaves the
medium at the water value `HomogeneousMedium::new` seeds, so the case fails if
the precedence regresses rather than quietly measuring the wrong medium. Before
this change the test had to set the coefficient on the medium to work around the
defect.

Agreement is `8.095065e-3` relative L2 at `r = 0.999999924` — the same numbers
the workaround produced, now reached through the documented API. That equality is
the point: the physics was always right, only its configuration route was not.

`cargo nextest run -p kwavers-solver -p kwavers -p kwavers-physics
-p kwavers-simulation` is `3356 passed`.

One existing test moved from an explicit request to `None` rather than being
migrated mechanically: `stratified_exponent_matches_per_tissue_uniform_operator`
builds a medium with a spatially varying exponent and asserts the stratified
operator engages. Under the new rule an explicit request applies uniformly, so
the per-voxel exponent would never be read and no strata would form. Its intent
was always medium-owned absorption, and it now says so. The suite caught this;
a mechanical `f64` to `Some(f64)` migration would have silently inverted it.

## Consequences

- A caller's coefficient reaches the solver, and a heterogeneous medium keeps its
  spatial variation. Neither is true today.
- Any absorption result produced by configuring `PSTDConfig` alone was computed at
  water's coefficient. That includes anything routed through the Python binding
  with an explicit `alpha_coeff_db` on a homogeneous medium.
- The k-Wave absorbing reference case becomes the regression oracle: it fails if
  the coefficient stops reaching the solver, which is how this was found.
- A test must assert the precedence directly — an absorbing run configured only
  through `PSTDConfig` must differ from a lossless one — because the fractional
  Laplacian machinery is well covered while its wiring was not.

## Not decided here

Whether `HomogeneousMedium` should take absorption as a constructor argument
rather than requiring `set_acoustic_properties` after construction. It is the
same ambiguity one level down, and it is a separate item.
