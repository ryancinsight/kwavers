# ADR 119 — In-repository k-Wave differential oracle

- **Status:** Accepted
- **Board item:** `KW-GAP-2026-08-20-KWAVEPARITY`
- **Class:** [major] [arch]
- **Date:** 2026-08-21

## Context

The README stated pseudospectral parity with k-Wave at Pearson `r >= 0.9999`
and located the harness that produced it at
`external/k-wave-julia/benchmarks/kwavers`. `external/` is gitignored
(`.gitignore:10`, `:96`) and tracked by zero files, so the headline validation
claim was not reproducible from a clean clone and no gate enforced it.

The gap was not a shortage of comparison code. `crates/kwavers-solver/src/validation/kwave_comparison/`
compares against analytical solutions only, and the Python side carries a large
cached-parity suite under `crates/kwavers-python/tests/`. What was missing was
reference *data*: no `.npz` artifact was tracked, so the Python suite could only
run against a locally generated cache, and the Rust-to-pytest bridge hard-sets
`KWAVERS_SKIP_KWAVE=1` (`crates/kwavers/tests/validation/python/mod.rs:155`), so
`cargo test` never exercised parity at all.

## Decision

Commit a small set of k-Wave reference fields with a provenance manifest, and
consume them from a Rust differential test that runs in the default gate.

- `scripts/generate_kwave_reference.py` drives `k-wave-python` over the
  reference `kspaceFirstOrder-OMP` binary and writes one NPZ archive per case
  plus `manifest.json`. Regeneration is the only step that needs the external
  reference solver.
- `crates/kwavers/tests/reference/kwave/` holds the artifacts: two NPZ archives
  and the manifest, 156 KB in total. `.gitignore`'s blanket `*.npz` rule is
  negated for exactly this directory, because these files are gate inputs rather
  than run outputs.
- `crates/kwavers/tests/kwave_reference_parity.rs` runs the kwavers k-space
  pseudospectral solver on the manifest's discretization and asserts agreement.

Vendoring the k-Wave source tree and executing MATLAB in CI both remain
non-goals. The reference solver stays external; only its output is committed.

## Case design

Both cases are lossless homogeneous-water initial-value problems with an
isotropic Gaussian initial pressure at `sigma = 3 dx`, on a `0.1 mm` grid at
`c0 = 1500 m/s`, `rho0 = 1000 kg/m^3`, Courant number `0.15`.

| Case | Grid | Steps | `dt` | Window radius | Absorption | Archive |
| --- | --- | --- | --- | --- | --- | --- |
| `ivp_homogeneous_2d` | 64 x 64 | 100 | 10 ns | 24 cells | lossless | 23 KB |
| `ivp_homogeneous_3d` | 32 x 32 x 32 | 50 | 10 ns | 12 cells | lossless | 128 KB |
| `ivp_absorbing_2d` | 64 x 64 | 100 | 10 ns | 24 cells | `40 dB/(MHz^1.5 cm)`, `y = 1.5` | 22 KB |

Four choices make the comparison an oracle rather than a coincidence.

1. **Identical discretization.** The manifest records the `dt` and step count
   k-Wave used, and the Rust run installs both. The seed is recomputed from the
   same closed form and checked against the stored `p0` to `1e-12`, so the two
   codes are handed the same problem rather than a similar one.
2. **Boundary-free window.** The comparison is restricted to a centred window
   the wavefront has entered but the absorbing layer has not reached. The kwavers
   run therefore uses a transparent boundary and no internal PML: nothing needs
   to absorb because nothing has arrived. A divergence inside the window is a
   divergence of the propagation scheme.
3. **No one-sided filtering.** Reference-side `smooth_p0` and kwavers-side
   anti-aliasing are both off. Each is applied by one code and not the other, and
   leaving either on would compare two different initial-value problems. The seed
   is band-limited by construction, so neither filter has work to do.
4. **Matching temporal correction.** Both codes run the Treeby-Cox (2010)
   `kappa = sinc(c dt |k| / 2)` k-space correction, which is what makes the two
   schemes the same scheme rather than two neighbours.

`k-wave-python` ships no one-dimensional solver, so the README's former 1-D row
has no reference here; the axisymmetric `kspaceFirstOrderAS` case is a separate
geometry and is left to a follow-up item.

## Measured result

Full-field agreement over the comparison window, at the committed revision:

| Case | Relative L2 | Relative L-infinity | Pearson `r` |
| --- | --- | --- | --- |
| `ivp_homogeneous_2d` | 5.50e-7 | 1.05e-6 | 1.000000000 |
| `ivp_homogeneous_3d` | 1.06e-4 | 2.20e-4 | 0.999999994 |
| `ivp_absorbing_2d` | 8.10e-3 | 4.99e-3 | 0.999999924 |

Both cases exceed the `r >= 0.9999` figure the README published, by four to five
orders of magnitude in the L2 norm.

The finite-difference solver is measured against the same reference as a
cross-scheme check, since the pseudospectral solver shares its k-space machinery
and cannot act as an independent oracle for it:

| Case | Relative L2 | Relative L-infinity | Pearson `r` |
| --- | --- | --- | --- |
| `ivp_homogeneous_2d`, finite difference | 2.53e-2 | 3.12e-2 | 0.999647 |

That separation is the scheme's own dispersion error, not a defect. A
fourth-order staggered scheme's relative phase error is `(k dx)^4 / 30` per
wavelength travelled; being quartic in the wavenumber it is set by the highest
resolved content, and a `sigma = 3 dx` Gaussian has a spectral width of `1/3`
rad per cell, putting its three-sigma edge near `k dx ~ 1` and the predicted
error at `1/30 ~ 3e-2`. The measurement is `2.5e-2`. Its gate is `5e-2` on
relative L2 and `0.99` on correlation — a dispersion error shifts phase, it does
not decorrelate, so falling below the correlation floor would mean the scheme is
wrong rather than merely dispersive.

The four tests execute in 2.35 s, inside the standard nextest budget.

## The absorbing case

`ivp_absorbing_2d` is the lossless two-dimensional case with power-law
absorption switched on and nothing else changed, so the pair isolates the
absorption model.

`alpha_coeff` is set to `40 dB/(MHz^1.5 cm)`, far above tissue's 0.5 to 1.5. The
seed's dominant content sits near 0.8 MHz and the wave travels 1.5 mm, so a
tissue coefficient attenuates by under one percent -- inside the comparison's
noise, and unable to distinguish a correct absorption model from none at all. At
40 the same path attenuates by roughly a third. Both codes receive the identical
coefficient, so the value's realism has no bearing on what the comparison
establishes. The test asserts that separation explicitly against the lossless
reference, because agreement alone would not prove the model ran.

Its bound is `2e-2` rather than the lossless `1e-3`: the residual is not the
storage-precision floor but the two codes' differing discretizations of the
fractional-Laplacian operator and its Kramers-Kronig dispersion partner
`eta tan(pi y / 2)`, which does not shrink with resolution. `2e-2` clears the
measured `8.1e-3` while staying far below the `0.32` attenuation under test, so a
mismodelled absorption cannot meet it.

### A defect this case exposed

The first attempt measured `r = 0.836` and, on inspection, produced a field
bit-identical to the lossless run and completely insensitive to `alpha_coeff` --
`4000` and `40` gave the same output to six digits.

`initialize_absorption_operators` prefers the *medium's* absorption coefficient
over the one in `PSTDConfig::absorption_mode`, falling back to the config only
when the medium reports exactly zero. `HomogeneousMedium::new` seeds
`absorption_alpha` with water's coefficient, which is never zero, so the config
coefficient is unreachable through that medium and the solver silently ran at
water's absorption instead of the requested value. Setting the coefficient
through `HomogeneousMedium::set_acoustic_properties` produces the agreement
tabulated above.

The precedence is real behaviour and may be the intended design -- k-Wave also
puts absorption on the medium -- but it is silent and undocumented, and it makes
a public configuration field inert for the most common medium type. Filed as
`KW-ABSORPTION-CONFIG-PRECEDENCE`; the reference test documents the working route
at its medium constructor rather than working around it silently.

## Tolerance derivation

The two codes integrate the same system with the same scheme over the same grid,
time step, and step count, so the residual is not a truncation error that shrinks
with resolution. Two sources set its floor:

- The reference is produced and stored in single precision
  (`data_cast="single"`), about `1.2e-7` relative per value, accumulating to
  roughly `1e-6` over the case's step count. This dominates the two-dimensional
  case, which measures `5e-7`.
- The three-dimensional window sits four cells from the domain edge, where the
  reference absorbs into a PML and kwavers is periodic. The pseudospectral
  operator has global support, so a small fraction of the outgoing field
  re-enters the window. This dominates the three-dimensional case at `1e-4`.

The gate is set at `1e-3` relative L2 and `5e-3` relative L-infinity: an order of
magnitude over the larger measured value, and an order inside the `1e-2` figure
the existing k-Wave comparison module states as its own acceptance threshold. A
tighter bound would be fitted to these two measurements rather than derived from
the sources above.

## Discrimination guard

`parity_degrades_when_the_step_count_is_wrong` requires that running one step too
few or too many costs at least two orders of magnitude of relative L2 error. A
window loose enough to accept a wrong final time is not measuring propagation, and
this is the assertion that makes the parity bounds falsifiable rather than
decorative.

The guard also pins the step-count convention that the first draft of this work
got wrong. `kgrid.Nt` is k-Wave's count of time *points*, spanning
`t_array = (0 : Nt - 1) * dt`, so the field it returns has advanced `Nt - 1`
propagation intervals. Recording `Nt` instead over-propagates by one step and
degrades agreement from `5e-7` to `5e-2` — a `r = 0.998` result that looks like a
solver divergence and is not one. The generator records `Nt - 1`, with the reason
at the assignment site.

## Consequences

- The README's parity claim now cites an in-repository test and its measured
  result. The gitignored `external/` path is no longer load-bearing.
- Regenerating or extending the reference set requires `k-wave-python` and its
  OMP binary on the generating machine. Running the gate does not.
- `KWAVERS_SKIP_KWAVE=1` in the Rust-to-pytest bridge is unchanged and still
  gates the Python cached-parity suite, which depends on the external solver.
  That suite's reproducibility is a separate item; this ADR closes the Rust-side
  gap only.
- The reference set covers linear propagation in two and three dimensions,
  lossless and with power-law absorption. Nonlinearity, heterogeneous media,
  elastic propagation, and source-driven (as opposed to initial-value) problems
  have no committed reference yet, and each is a follow-up case rather than a
  claim this ADR supports.

## Alternatives rejected

- **Vendor the k-Wave source and run it in CI.** Rejected: it imports a MATLAB or
  large C++ build dependency into every gate to reproduce output that does not
  change. Committing the output is the same evidence at a fraction of the cost.
- **Keep the comparison in the Python suite only.** Rejected: the Rust solver is
  the artifact under test, and routing its only differential oracle through a
  binding layer and an environment flag is what allowed the claim to go unchecked.
- **Store a sensor time series instead of the final field.** Rejected: a
  full-field comparison discriminates strictly more, and at these grid sizes it is
  still a small artifact.
