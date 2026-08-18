# 111. Retire the `KzkSolverPlugin` surface onto the complex-field KZK adapter

- Status: Accepted (retroactive)
- Date: 2026-08-18
- Item: `backlog.md#atlas-kwavers-kzk-tests-082`
- Records: commit `950fbc588`, which made the change without a decision record

## Context

`crate::forward::nonlinear::kzk_solver_plugin` exposed `KzkSolverPlugin`, the
`Plugin` implementation the `PhysicsCatalog` handed out for
`NonlinearEquation::KZK` — the therapy planning path. It carried three physics
defects, all live on that path:

1. **Spectral/real-space index conflation.** Its diffraction operator indexed
   transverse wavenumbers by the *spatial* grid index
   (`kx = 2πi/(nx·dx)` for cell `i`), so each cell was multiplied by a factor
   belonging to a different mode.
2. **Real `cos()` in place of the complex propagator.** The parabolic
   propagator is `exp(−i k_T²Δz/(2k₀))`; the plugin applied `cos(k_T²/(2k₀))`,
   discarding both the phase accumulation and the `Δz` scaling.
3. **Dimensionally wrong absorption.** `Plugin::update` raised a per-`Δz`
   attenuation factor to the power of a *time* step
   (`absorption.powf(dt/2)` with `dt ≈ 1e−7`), which is `≈ 1` — the plugin
   attenuated essentially nothing.

A correct complex-field solver — `KZKSolver` in `forward::nonlinear::kzk` —
already existed, validated by the module's own absorption, diffraction, and
harmonic-generation tests. Commit `950fbc588` deleted the buggy module and
introduced `KzkPlugin`, a thin adapter wrapping `KZKSolver`.

That commit did not record the decision, did not carry a CHANGELOG entry, and
did not run `cargo semver-checks` — and the tests it shipped with the new
adapter asserted only finiteness and positivity. Re-running the retired plugin
under those tests confirms they were vacuous: all three passed against the
buggy implementation (see Consequences).

## Decision

Retire the `kzk_solver_plugin` module and the `KzkSolverPlugin` type outright.
`forward::nonlinear::kzk::KzkPlugin` is the single `Plugin` implementation for
`NonlinearEquation::KZK`.

The adapter owns no physics. Its contract is exactly three things, and each is
now pinned by a test that fails against the retired implementation:

| adapter responsibility | oracle |
|---|---|
| axis remap therapy `(x, y, z)` → KZK `(z, x, y)`, cached per-plane readout | bit-for-bit differential equality against a reference `KZKSolver` built from the same config and source |
| absorption is carried through unchanged | Beer–Lambert `p(z₂)/p(z₁) = exp(−α·Δz)`, asserted to a derived round-off bound |
| diffraction is carried through unchanged | a transversely uniform field stays uniform, because `H(k_T = 0) = 1` |

The peak-normalising source extraction destroys absolute scale, so the
absorption oracle is stated as an axial *ratio*, which the normalisation leaves
invariant.

## Alternatives rejected

**Fix `KzkSolverPlugin` in place.** Two of the three defects are in operators
that `KZKSolver` already implements correctly and already tests. Repairing the
second copy would have entrenched the duplication rather than removing it.

**Keep the module as a deprecated re-export.** A `#[deprecated]` alias to
`KzkPlugin` is the compatibility shim the integrity rules exclude, and here it
would be worse than usual: the two types have different constructors and
different update semantics, so the alias would not have been a drop-in.

**Keep the existence-only tests and add value-semantic ones beside them.** A
test that cannot fail is not weak coverage, it is a false signal on the board;
the correct action is replacement, not accumulation.

## Consequences

**Breaking**, hence `[major]`. `crate::forward::nonlinear::kzk_solver_plugin`
and `KzkSolverPlugin` are gone from the public surface of `kwavers-solver`.
External callers migrate to `crate::forward::nonlinear::kzk::KzkPlugin`, or —
preferably — obtain the plugin from `PhysicsCatalog` and never name the type.
The two are not signature-compatible: `KzkSolverPlugin` exposed
`initialize_operators`, `propagate_volume`, `solve`, `shock_formation_distance`,
and `apply_retarded_time` as inherent methods; `KzkPlugin` exposes only the
`Plugin` trait. Callers needing the solver directly use
`forward::nonlinear::kzk::KZKSolver`.

No in-repo caller survives: `plugin/catalog.rs` and the therapy
`orchestrator/execution.rs` were rewired in the same commit.

**Behaviour change on the therapy path.** Results are not comparable across the
retirement. Under the retired plugin a plane wave in a 0.5 dB/(cm·MHz) absorber
lost 6e−11 of its amplitude over 24 mm where Beer–Lambert requires 13%; the new
path reproduces the analytical value to 1e−11 relative. Any stored therapy
result produced through `NonlinearEquation::KZK` before this change understates
attenuation and misplaces diffractive spreading, and must be regenerated.

**Verification of the retirement itself.** Each new oracle was falsified against
the resurrected implementation (restored from `950fbc588^`, wired back into the
catalog):

| test | against `KzkPlugin` | against resurrected `KzkSolverPlugin` |
|---|---|---|
| `plane_wave_decays_at_the_beer_lambert_rate` | pass | fail: measured ratio 0.999999999938 vs required 0.870964 (14.8% error) |
| `plane_wave_stays_transversely_uniform` | pass | fail: transverse spread 1.0 at the first plane |
| `adapter_reproduces_the_reference_solver_under_the_axis_remap` | pass | fail: 1.30e−5 Pa written vs 8.14e−5 Pa reference at the first cell |

The three superseded tests (`plane_wave_absorption_oracle`,
`plugin_evolves_real_field`, `focused_beam_amplitude`) all **passed** against
the resurrected buggy plugin, which is the evidence that they never tested it.
