# ADR 121 — Plugins receive the sources they are given

- **Status:** Accepted
- **Board item:** `KW-PSTD-PLUGIN-SOURCES-DROPPED`
- **Class:** [major]
- **Date:** 2026-08-22

## Context

`PluginManager::execute` takes a source list and threads it to every plugin
through `PluginContext::sources`. Two of the three solver plugins never read it.

`PSTDPlugin::initialize` constructed its `PSTDSolver` with
`GridSource::default()` — an empty source — and `PSTDPlugin::update` bound the
context parameter as `_context`. `FdtdPlugin` did the same, its construction site
carrying the comment "no active sources unless configured elsewhere". Only
`HybridPlugin` consumed `context.sources`.

The consequence is silent. A caller who builds a source, hands it to
`PluginManager::execute`, and drives the pseudospectral or finite-difference
solver gets a simulation that runs to completion, returns `Ok`, and was never
driven. Nothing fails, so nothing says so. That is the same shape as an
input-insensitive implementation: the output does not depend on an input the
signature accepts.

None of the machinery was missing. `PSTDSolver::add_source_arc` builds the mask,
determines the injection mode, computes the spectral gradient masks for velocity
sources, and pushes onto `dynamic_sources`, which the stepper consumes every
step. `SourceHandler::add_source` performs the equivalent conversion for the
`GridSource` route, collecting mask indices in Fortran order to match k-Wave's
own ordering. The only missing link was the plugin calling either one.

This surfaced while adding the driven differential case for ADR 119: the case
could not be expressed through the plugin path at all, and had to go through
`PSTDSolver` directly.

## Decision

Plugins register the sources they are handed, on their first `update`.

Registration cannot happen in `initialize`, which is not given the sources —
they arrive through `PluginContext`, which only `update` receives. Doing it once
and recording that keeps the solver's source list from growing by one copy per
time step.

`PluginContext::sources` changes from `&[Box<dyn Source>]` to
`&[Arc<dyn Source>]`. The solver stores its sources for the run's duration
because the stepper queries `amplitude(t)` every step, so it needs ownership
that outlives the context borrow; `Arc` gives the plugin a share without taking
the caller's. `PSTDSolver::add_source` already converted `Box` to `Arc`
internally, so `Arc` was the internal form regardless — this makes the boundary
agree with it, and lets one source be shared across plugins.

This is [major]: `PluginContext` and both `PluginManager::execute` signatures are
public. In-repo callers migrate in the same change.

## Verification

The driven k-Wave reference case (ADR 119) validates both routes against one
stored field:

| Route | Relative L2 vs k-Wave | Pearson `r` |
| --- | --- | --- |
| `PSTDSolver` with a `GridSource` | 2.58e-3 | 0.999996674 |
| `PluginManager` with a `dyn Source` | 2.58e-3 | 0.999996674 |

The two routes agree with **each other** to `4.0e-13` relative L2 — the
accumulated difference of a few reordered floating-point operations over 151
steps. They are the same computation reached two ways, and the test's bound of
`1e-10` says so at seven orders below the `2.58e-3` at which either differs from
the reference, so it cannot be satisfied by two routes that merely both sit near
k-Wave.

`the_plugin_path_drives_the_solver_it_was_given_sources_for` asserts three
things rather than one, because a parity bound alone would not have caught the
original defect had the field been attenuated rather than absent: agreement with
k-Wave at the driven case's bound, agreement with the solver route to round-off,
and a field peak at least half the reference's — which a run whose sources were
discarded cannot produce.

## Consequences

- A caller who passes sources to `PluginManager::execute` gets a driven
  simulation from `PSTDPlugin` and `FdtdPlugin`, which is what the signature
  always claimed.
- The FDTD plugin is fixed in the same change because it carried the identical
  defect; leaving a known instance of a class unfixed beside a fixed one is how
  the class survives. Its route is not validated against k-Wave here — the
  finite-difference scheme sits at its own dispersion error rather than at the
  parity bound (ADR 119) — so its guarantee is the shared registration path, not
  a measurement.
- Anything previously driven through the plugin path produced an undriven
  result. There is no way to distinguish such a run from a correct one after the
  fact except by its output being physically empty.

## Alternatives rejected

- **Remove the sources parameter so misuse is a compile error.** Honest about the
  old behaviour and much smaller. Rejected: sources belong in a multi-physics
  composition API, the solver machinery to honour them already existed and was
  correct, and removing the parameter would have deleted a capability to avoid
  connecting two things that were one call apart.
- **Keep `Box` and register by reference.** Rejected: the stepper queries
  `amplitude(t)` every step, so the solver must hold the source beyond the
  context borrow. A by-reference registration would need the source to outlive a
  borrow it cannot.
- **Register in `initialize` by changing the `Plugin` trait.** Rejected: it
  widens the trait every implementor must satisfy in order to move information
  that already reaches `update`.
