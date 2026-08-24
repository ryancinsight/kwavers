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

## Reproducibility of the reference set

Regenerating a case produces a byte-identical archive: re-running the generator
for `ivp_homogeneous_2d` and `ivp_absorbing_2d` yields the same SHA-256 as the
committed manifest records and bitwise-equal arrays. The reference solver is
deterministic for these inputs, so the manifest's hashes are a working
verification mechanism rather than decoration -- a regenerate-and-diff check can
tell a stale artifact from a current one, and a hash mismatch means the inputs
or the reference solver changed, not that the run drifted.

This was measured on the generating platform, which the manifest records. It is
not a claim about bitwise agreement across platforms or across `k-wave-python`
versions; a hash mismatch after either changes is expected and is what the
recorded provenance exists to explain.

## Case design

Both cases are lossless homogeneous-water initial-value problems with an
isotropic Gaussian initial pressure at `sigma = 3 dx`, on a `0.1 mm` grid at
`c0 = 1500 m/s`, `rho0 = 1000 kg/m^3`, Courant number `0.15`.

| Case | Grid | Steps | `dt` | Window radius | Absorption | Archive |
| --- | --- | --- | --- | --- | --- | --- |
| `ivp_homogeneous_2d` | 64 x 64 | 100 | 10 ns | 24 cells | lossless | 23 KB |
| `ivp_homogeneous_3d` | 32 x 32 x 32 | 50 | 10 ns | 12 cells | lossless | 128 KB |
| `ivp_absorbing_2d` | 64 x 64 | 100 | 10 ns | 24 cells | `40 dB/(MHz^1.5 cm)`, `y = 1.5` | 22 KB |
| `ivp_layered_2d` | 80 x 64 | 120 | 8.33 ns | 24 cells | lossless, layered medium | 34 KB |
| `src_tone_burst_2d` | 96 x 80 | 151 | 10 ns | 30 cells | lossless, driven point source | 36 KB |
| `src_nonlinear_2d` | 96 x 80 | 151 | 10 ns | 30 cells | driven, `B/A = 20` at 5 MPa | 36 KB |

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
| `ivp_layered_2d` | 7.97e-3 | 1.54e-2 | 0.999963422 |
| `src_tone_burst_2d` | 2.58e-3 | 2.20e-3 | 0.999996674 |
| `src_nonlinear_2d` | 3.30e-3 | 2.80e-3 | 0.999994550 |

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

The nine tests execute in 3.10 s, inside the standard nextest budget.

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

## The layered case, and the axis conventions it exposed

`ivp_layered_2d` steps sound speed from 1500 to 1800 m/s and density from 1000
to 1200 kg/m^3 across a two-cell hyperbolic-tangent interface eight cells right
of the seed. The wavefront reaches it around step 53 of 120, so the recorded
field carries a transmitted wave, a reflection back through the seed, and a
refracted front. The test asserts agreement with the reference and separation
from a uniform-medium run at the same discretization, which measures `0.27`.

The step is smoothed rather than sharp because both codes are pseudospectral and
a discontinuity rings in both; smoothing keeps the medium as band-limited as the
seed, so the comparison measures heterogeneous propagation rather than two Gibbs
phenomena.

**Its grid is deliberately non-square, and that is load-bearing.** Two
orientation defects were live in this harness and invisible to every square,
symmetric case:

1. k-Wave returns the recorded field with its axes reversed relative to the grid
   it was given -- a `(Nx=64, Ny=48)` problem comes back shaped `(48, 64)`. On a
   square case `.reshape(shape)` is a no-op and stores a transposed field.
2. `np.savez` records an array's memory order in the NPY header, and a
   transposed array is Fortran-contiguous, so the stored buffer was column-major
   while the Rust reader indexed it row-major.

Both are now closed at their source: the generator reverses the axes explicitly
and asserts the result matches the case shape, then writes with
`np.ascontiguousarray`; the reader asserts the stored array is not Fortran-ordered
and places elements by index rather than copying into a backing slice whose order
is its own business. The non-square case makes each of these a shape error or an
assertion failure rather than a silent transpose.

The lossless and absorbing results are unchanged by the corrections, because an
isotropic seed in a uniform medium is symmetric under transpose to within the
`1.8e-7` that separates the two orientations -- which is exactly why a square
case could not detect the defect. Their stored archives were regenerated so the
committed data is correctly oriented regardless.

The lesson is recorded rather than merely fixed: **a differential oracle whose
cases are all symmetric cannot detect a transposition**, and every convention a
reference solver carries -- axis order, memory order, time-point counting --
needs at least one case that breaks the symmetry hiding it.

## The driven case, and the step count it corrected

`src_tone_burst_2d` injects a Gaussian-windowed 3 MHz burst at a single
off-centre cell of a 96 x 80 grid, starting from a zero field. Every other case
seeds an initial pressure and lets it evolve, which never reaches the source
injection path -- and that path is the one a real driven simulation runs on. It
carries four conventions the initial-value cases cannot check: where the mask's
cell sits, which signal column a step consumes, how the source term is scaled,
and whether the k-space source correction is applied to it. All four have to be
right at once for the case to pass, and it does, at `2.58e-3` and
`r = 0.999996674`.

It runs through `PSTDSolver` rather than `PluginManager`. The plugin path
constructs its solver with an empty `GridSource` and never forwards the sources
handed to `execute`, so a driven comparison cannot be expressed through it. That
is worth knowing independently of this ADR: a caller who passes sources to
`PluginManager::execute` and expects the pseudospectral solver to see them will
get a silently undriven simulation.

**The propagation-interval count differs by source type, and the difference is
one step.** An initial-value case has a meaningful state at `t = 0` -- the seed
itself -- so the first of k-Wave's `Nt` time points is the initial condition and
the returned field is `Nt - 1` intervals later. A driven case starts from a zero
field, which is not a state worth counting: k-Wave updates for every one of its
`Nt` points and the returned field is `Nt` intervals later.

Both were measured rather than assumed. Using `Nt - 1` on the driven case scores
`2.59e-1` instead of `2.58e-3` -- two orders worse, and the kind of result that
reads as a solver defect. The manifest therefore records the interval count for
the case at hand rather than one rule for both, and the step-count guard now
covers the driven case, where a single wrong step costs the same hundredfold it
costs the initial-value cases.

That is the third convention this series has had to pin down by measurement,
after time-point counting and axis order. The pattern is stable enough to state
plainly: **a reference solver's conventions are discovered by a case that breaks
the symmetry hiding them, never by reading them off the interface.**

## The nonlinear case, and the trade that sizes it

`src_nonlinear_2d` is the driven case with finite-amplitude propagation switched
on and nothing else changed. It matches at `3.30e-3`, barely above the driven
case's `2.58e-3`, which says the nonlinear equation of state is where the two
codes agree most closely rather than least.

Two parameters trade against each other, and the trade is the design.
Accumulated distortion scales with the propagation distance as a fraction of the
plane-wave shock distance `rho c^3 / (beta omega p0)`, so raising either `beta`
or `p0` increases it. But crossing the shock distance asks both codes to resolve
a discontinuity neither has shock capturing for, which would compare two Gibbs
phenomena rather than two nonlinear propagations.

At water's `B/A = 5` and 5 MPa the case measured a `1.5e-2` separation from a
linear run — only five times the parity bound, too close to distinguish a
correct nonlinear term from an absent one. `B/A = 20` at the same amplitude puts
the shock distance at about 3.3 mm against the 2.25 mm travelled, roughly
seventy percent of the way, and the separation at `4.58e-2`: fourteen times the
measured residual. That is as far into the nonlinear regime as the smooth-wave
assumption reaches.

Nonlinearity is the one physics dimension with no configuration route:
`PSTDConfig` carries only the boolean that switches the term on, so `B/A` travels
on the medium, which is also where k-Wave puts it. That is a coherent split —
unlike the absorption coefficient's, which ADR 120 had to correct — because there
is no second owner to disagree with.

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
- The reference set covers linear propagation in two and three dimensions:
  lossless, with power-law absorption, through a layered medium, driven by a
  time-varying point source, at finite amplitude, and from a distributed
  multi-cell source. Elastic propagation has no committed reference yet, nor
  does the axisymmetric geometry. Each is a follow-up case rather than a claim
  this ADR supports.
- The distributed case pins k-Wave's mask-cell ordering, which the single-cell
  cases cannot observe: with one masked cell there is exactly one signal row and
  any mapping between mask and signal produces the same field. k-Wave consumes a
  mask's cells by column-major linear index, first axis varying fastest. That was
  measured rather than assumed -- driving a four-cell mask one row at a time put
  each resulting field's `|p|` centroid within 0.13 cells of the cell that
  ordering predicts, against a 2-to-4 cell separation between the candidates, so
  the reference's own convention was established before any kwavers field was
  compared to it. The case then holds kwavers to it two ways: parity at
  `3.36e-3` relative L2, and a guard that reverses the signal rows and requires
  the field to move, which it does by `1.05` -- 313 times the parity residual,
  with correlation falling from `0.999994` to `0.446`. Without that guard the
  case would pass for a solver that drove the right cells with the wrong
  signals.

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
