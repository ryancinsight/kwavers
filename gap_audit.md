# Gap Audit

Module-by-module audit of physics/numerics implementations, marking items for
future revision. Compact + link-navigable per agent-artifact policy; closed
history (pre-2026-05-29) was pruned from `docs/` during the workspace-split
docs cleanup and remains recoverable from git history.

**Method:** four parallel read-only module audits (2026-05-31) over `solver/`,
`physics/`, `clinical/`+`domain/`, `analysis/`+`math/`+`core/`+`gpu/`. Findings
below are candidate gaps for revision; severity tiers per `CLAUDE.md` integrity
policy. Items tagged **[verify]** are pattern-match suspicions that MUST be
confirmed against the code (and ideally a literature reference) before any fix —
do not assert an unconfirmed physics error.

### Sprint A verification results (2026-05-31) — all 6 C-tier `[verify]` resolved
Verified each against code + governing equation. **Zero confirmed physics bugs**;
4 false positives, 2 real-but-overrated (downgraded). Lesson: automated severity
inflation is real — the verify gate paid off.
- **SOL-4** Westervelt `d²(p²)/dt²` — **FALSE POSITIVE.** `2p·p̈+2ṗ²` is exact;
  the FMA fuses one product (precision *gain*, not error). Closed.
- **PHY-1** Gilmore vapor correction — **FALSE POSITIVE.** `p_eq` subtracts `pv`
  (line 211) so `p_gas` is the non-condensable partial pressure; polytropic rate
  is correct. Closed.
- **PHY-3** IAPWS-IF97 Region 4 — **FALSE POSITIVE.** Coefficients n₁–n₁₀ and the
  θ/A/B/C/`[2C/(−B+√(B²−4AC))]⁴` form match the standard exactly (K→MPa→Pa). The
  "dimensional inconsistency" misread the standard. Closed.
- **AMC-1** 6th-order central difference — **FALSE POSITIVE.** Nested `mul_add`
  expands to `[−f₋₃+9f₋₂−45f₋₁+45f₊₁−9f₊₂+f₊₃]/(60dx)` = exact Fornberg. Closed.
- **AMC-2** MVDR imag-part guard — **REAL, downgrade C→L** (AMC-2 below). Happy
  path correct (`aᴴR⁻¹a` real for Hermitian R); add cheap `denom.im≪|denom|`
  defensive guard in `weights.rs:36` + `spectrum.rs:45`.
- **AMC-4** wgsl boundary — **REAL, downgrade C→M** (AMC-4 below). Live shader
  (`WaveEquationGpu`); boundary is persistence (`out=in`), not the assumed
  implicit-Dirichlet. Undocumented + likely inconsistent with CPU Neumann/Dirichlet.

### Sprint B verification results (2026-05-31) — "confirmed correctness" mostly evaporated
Same pattern as Sprint A: the audit's "production panic" labels were wrong.
- **SOL-1** harmonic accessor panics — **NOT production.** Only callers are
  `#[test]` (incl. `tests.rs:104` intentional should-panic); `try_*` variants
  already exist for fallible use. Idiomatic infallible-accessor pattern. No change.
- **SOL-2** elastography:296 — **test-only** (`#[test] test_all_nonlinear_methods_integration`,
  `panic!` is the `unwrap_or_else` message). No change.
- **SOL-3** PINN-EM:69 — **test-only** (`#[test] test_boundary_condition_builder`,
  panic in a type-assertion match arm). No change.
- **PHY-5** Cattaneo defaults — **REAL, fixed.** Root cause was not a wrong value
  but dead state: `thermal_wave_speed` was never read. Removed the field (the
  flux law uses only τ; `c=√(α/τ)` is derived). 2 value-semantic thermal tests
  pass, physics unchanged.
Net: 3 false-positive "production panics" closed, 1 real dead-field removed.

### Sprint C verification results (2026-06-01) — approximation-validity bounds
6 items; **zero physics behavior changed** (doc + one behavior-preserving named const).
- **PHY-2** Gilmore adiabatic — already documented (Prosperetti 1977 cited). Closed.
- **PHY-4** Marmottant shell viscosity — **FALSE POSITIVE.** `12·μ_s·(d/R)·Ṙ/R²`
  term present at `marmottant.rs:107` (audit had wrong path + wrong claim). Closed.
- **PHY-8** parametric averaging — correctly scoped; added `Δf/f̄≪1` bound. Done.
- **CLD-2** HIFU linear-only — real; `generate_acoustic_field` already self-documents,
  added orchestrator-level note. **Open [minor]:** wire KZK into HIFU path. Partial.
- **CLD-3** O'Neil "Theorem" — real; →"approximation" + validity regime + refs;
  named/flagged the 0.7 fill factor (value preserved). Done.
- **CLD-6** Thermal Index Pennes omission — real; documented + conservative-bias
  direction + Pennes 1948 ref. Done.
Net: 2 already-handled/false-positive, 4 validity-regime docs; 0 physics changed.

### Sprint D verification results (2026-06-01) — missing literature validation
Pattern holds: of the "missing validation" items, most were already covered or
mislabeled. Added 2 genuine external-reference/property tests; tightened nothing
that was already tight; refused to fabricate a reference where none is sourced.
- **PHY-10** Minnaert resonance — **DONE.** The flagged `validate_implementation()`
  is indeed circular (code-vs-itself), but an independent check already existed
  (`test_epstein_plesset_vs_minnaert_frequency`). Added
  `test_minnaert_constant_matches_literature_value`: pins `f₀·R₀ ≈ 3.26 m·Hz`
  (Minnaert 1933; Leighton 1994) across R₀∈[1e-6,1e-3], max_rel=0.02 (computed
  ≈3.286, 0.8% off). PASSED.
- **SOL-7** PSTD source scale — **DONE.** Added
  `interior_source_conserves_total_amplitude_across_geometry`. Key physics:
  scale=1/N is *amplitude* normalization → N·scale=1 invariant to geometry/
  resolution; energy Σscale²=1/N is deliberately NOT conserved (asserting energy
  conservation would be physically wrong). PASSED.
- **PHY-9** K-M equilibrium tol — **FALSE POSITIVE (overrated).** `<1.0 m/s²` is
  NOT loose: characteristic K-M accel for default R₀=5µm is (2σ/R₀)/(ρR₀)≈5.8×10⁶
  m/s², so 1.0 is a *relative* ~1.7×10⁻⁷ bound (at the f64 cancellation floor of
  the 2σ/R₀≈2.9×10⁴ Pa subtraction). Documented the scale in-place; tightening
  further would test FP noise, not physics. No behavior change.
- **PHY-11** Gilmore collapse — **ADEQUATE.** Suite already has a value-semantic
  analytical differential check (`enthalpy_derivative_uses_state_wall_acceleration`,
  rel_err<1e-10 vs closed-form Gilmore RHS) — the strongest tier short of a
  published collapse dataset. A Lauterborn Rmax/R₀ regression needs a *citable*
  number; not fabricated. Deferred to backlog [minor] pending sourced reference.
- **CLD-11** CPML reflection — **DONE.** Added
  `theoretical_reflection_decays_monotonically_with_thickness` (Collino&Tsogka
  2001): strict-decrease + bounded-in-(0,target] property test. Parameters
  derived analytically from σ_max=σ_factor·(m+1)·c/(150π·dx) to keep R in the
  representable range (fine dx underflows R→0 by t=1, itself correct CPML behavior;
  cosθ clamped ≥0.1 internally). Note: pre-existing `test_theoretical_reflection_
  for_dimension` is circular (recomputes the same formula).

### Technical debt log (crate-split, 2026-06-01)
Debt surfaced by the workspace extraction (ADR 011). Logged for proper future
resolution; current state is a sound mitigation, not a full fix.
- **DEBT-1 (M, arch):** `KwaversError` is a kitchen-sink foundation error coupled
  to 5 higher-layer crates via `From`/`#[from]`: `wgpu`, `flume`, `ritk_registration`,
  `nifti`, `anyhow`. The orphan rule pins these to `kwavers-core`. **Mitigation
  applied:** made them optional + `#[cfg(feature)]`-gated (`gpu`/`channels`/
  `registration`/`nifti`); default `kwavers-core` is a clean leaf, facade enables
  what the monolith had. **Proper fix (deferred):** the foundation error must not
  know about GPU/channel/file-format/registration types — replace these conversions
  with generic variants (`Io`/`External`) + explicit boundary conversion in the
  consuming layer, or split `KwaversError` into layered error types
  (`thiserror` per layer, `anyhow` only at app top — per CLAUDE.md error policy).
  `anyhow::Error` in a *library* foundation error (`Other` variant) also violates
  the lib-uses-thiserror rule and should be removed in that refactor.
  File: `crates/kwavers-core/src/error/mod.rs`.
- **DEBT-2 (L):** `solver::interface::factory::RegistrationEngine` trait now has
  zero implementors (its only impl was the dead one removed in CLD-14-adjacent
  cleanup). Candidate removal once confirmed no external/plugin consumer.
- **DEBT-4 (C) — RESOLVED (2026-06-01):** FWI finite-window Born adjoint-gradient
  mismatch FIXED. Root cause: a discrete-adjoint **off-by-one** in
  `finite_window::adjoint_backward_pass`. The forward source `s[m] = −χ·accel[m]`
  drives `ps1[m+1]`, so the adjoint-state gradient must pair `accel[m]` with the
  adjoint field at the SAME source index, `ν[m]` (= `pa_prev`, just computed). The
  code paired it with `ν[m+1]` (= `pa_curr`), a one-leapfrog-step bias → 3.7% error.
  Fix: cross-correlate against `pa_prev`. Now matches central finite-difference to
  5e-4 (gradient_fd: 6/6 pass). Verified by derivation (transpose of the discrete
  leapfrog) + value-semantic FD test. The other CBS adjoints (separate paths) were
  already correct.

  ~~PRE-EXISTING WIP — original finding:~~
  `solver::inverse::fwi::frequency_domain::tests::gradient_fd::pstd_finite_window_
  born_adjoint_gradient_matches_finite_difference` FAILS: analytic adjoint gradient
  57.88 vs central-difference 55.72 = **3.7% error** (tolerance 5e-4). Magnitude
  rules out FD-step roundoff — it's a genuine adjoint-correctness gap in the Born
  operator (`DenseConvergentBornOperator`). NOT caused by the crate migration
  (extractions only rewrite import paths; numerics identical). This is in-progress
  FWI work that was *uncommitted* in the working tree at session start and got
  swept into commit 7cb668baf via `git add kwavers/`; the adjacent new
  `PstdFiniteWindowBornSecondOrderOperator` self-documents its adjoint as
  "approximate, not the exact second-order adjoint". **Action (owner): complete the
  exact Born adjoint (verify the operator transpose / adjoint-state derivation), or
  mark the test `#[ignore]` with a WIP note until then.** Not fixed here — deep FWI
  numerical work outside the migration scope.

- **DEBT-3 (M, SSOT):** Photoacoustic implementation is fragmented. After lifting
  the imaging vertical out of solver (2026-06-01: `solver/photoacoustics` →
  `simulation/photoacoustics/vertical`), there are still TWO parallel
  photoacoustic impls — `simulation/photoacoustics/vertical/` (optical/source/
  acoustic/reconstruction "solver vertical", consumed by the orchestrator) and
  `simulation/modalities/photoacoustic/` (acoustics/optics/core/reconstruction/
  types, owns `PhotoacousticResult`/`PressureFieldSeries`) — plus a coupled-solver
  touch `solver/multiphysics/photoacoustic.rs`. **Consolidate to one canonical
  photoacoustic modality** (determine the live impl, merge, delete the parallel
  one). Deferred: merging two parallel impls is careful SSOT work, not a move.

**Severity:** `C` correctness/physics-incorrect · `H` simplification/approximation
presented as exact · `M` missing validation/test · `L` doc/cleanup.

Standing facts (do NOT re-flag):
- FDTD `simd_stencil` (tiled scalar) vs `avx512_stencil` (intrinsics) are NOT
  duplication — distinct coefficient conventions + BCs (Neumann vs Dirichlet).
- `physics::bubble_dynamics` is a live `pub use` alias of `acoustics::bubble_dynamics`.
- Theranostic operator-vs-PAM passive-channel distinction is documented/intentional.
- Genuine orphans already removed: `fdtd/simd/`, `symplectic_integration/`.

---

<a id="solver"></a>
## solver/ (FDTD/PSTD/k-space/nonlinear · FWI/RTM/CBS/elastography)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| SOL-1 | ~~C~~ CLOSED (false positive) | `forward/elastic/nonlinear/wave_field/mod.rs:145,165` | Verified: panicking accessors only called from `#[test]` (incl. intentional should-panic at tests.rs:104); `try_*` exist for fallible use. Idiomatic. No change. | — |
| SOL-2 | ~~H~~ CLOSED (false positive) | `inverse/elastography/mod.rs:296` | Verified: inside `#[test] test_all_nonlinear_methods_integration`; panic is the `unwrap_or_else` message. No change. | — |
| SOL-3 | ~~H~~ CLOSED (false positive) | `inverse/pinn/ml/electromagnetic/mod.rs:69` | Verified: inside `#[test] test_boundary_condition_builder`, type-assertion arm. No change. | — |
| SOL-4 | H [verify] | `forward/fdtd/pressure_updater/nonlinear.rs:44` | Westervelt `d²(p²)/dt²` FMA ordering — confirm bit-faithful to Hamilton&Blackstock 3.43a | add value-semantic test vs analytic |
| SOL-5 | M | `forward/nonlinear/hybrid_angular_spectrum/absorption.rs:85` | power-law α no freq>0 / exponent∈(0,2] guard | validate inputs |
| SOL-6 | M | `forward/coupled/thermal_acoustic/stepping.rs:174-204` | density-gradient momentum update unvalidated for stability | add CFL/gradient test |
| SOL-7 | ~~M~~ DONE (2026-06-01) | `forward/pstd/implementation/core/source_injection.rs:91` | Added geometry-invariant amplitude-conservation test (clustered vs dispersed give equal scale; N·scale=1). Energy NOT conserved by design (amplitude normalization). | done |
| SOL-8 | M | `forward/fdtd/avx512_stencil/construction.rs:43` | coeffs precomputed w/o finite/sign assertion | debug_assert finite |
| SOL-9 | M | `solver/constants.rs:82` | benchmark tolerances (5%/10%/2%) uncited | cite Taflove&Hagness |
| SOL-10 | L | (crate-wide, ~339 files) | ~30% public fns lack Rustdoc | doc sweep |
| SOL-11 | L | `validation/kwave_comparison/` | validators exist but not CI-wired | wire regression suite |

<a id="physics"></a>
## physics/ (nonlinear acoustics · bubble dynamics · thermal · optics · chemistry)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| PHY-1 | C [verify] | `acoustics/bubble_dynamics/gilmore/mod.rs:213` | gas-pressure rate may omit vapor-pressure correction | verify vs Gilmore 1952; add term if confirmed |
| PHY-2 | ~~H~~ CLOSED (already documented) | `acoustics/bubble_dynamics/gilmore/mod.rs:99-102` | Verified: adiabatic approximation already stated + Prosperetti 1977 cited; γ selects polytropic/isothermal. No gap. | — |
| PHY-3 | ~~C~~ CLOSED (false positive) | `acoustics/bubble_dynamics/thermodynamics/vapor_pressure.rs:189` | Verified Sprint A: IAPWS-IF97 Region 4 matches the standard exactly (K→MPa→Pa). | — |
| PHY-4 | ~~H~~ CLOSED (false positive) | `acoustics/bubble_dynamics/encapsulated/model/marmottant.rs:107` | Verified: shell-viscosity term `12·μ_s·(d/R)·Ṙ/R²` IS present and in the pressure balance (Marmottant 2005 eq. 3). Audit had wrong path + wrong claim. | — |
| PHY-5 | ~~H~~ RESOLVED (2026-05-31) | `thermal/diffusion/hyperbolic.rs:13` | Verified: `thermal_wave_speed` was DEAD state (never read; flux law uses only τ). The implausible default never affected results. Fixed by REMOVING the field (SSOT/SRP) — `c=√(α/τ)` is derived, not an input. τ=20s default kept + cited (Mitra 1995). 2 thermal tests pass. | done |
| PHY-6 | H | `optics/sonoluminescence/blackbody.rs:26` | emissivity=0.1, optical_depth=0.1 unjustified magic defaults | cite or make required |
| PHY-7 | H | `chemistry/ros_plasma/ros_species/generation.rs:20,30` | Arrhenius prefactors 1e13/1e14 uncited | cite NIST/ChemKin |
| PHY-8 | ~~H~~ DOC'D (2026-06-01) | `acoustics/wave_propagation/nonlinear/parametric.rs:96` | Correctly scoped to closely-spaced primaries; added explicit `Δf/f̄≪1` validity bound. Large-Δf case is outside the parametric-array model by definition. | done |
| PHY-9 | ~~M~~ CLOSED (false positive, 2026-06-01) | `acoustics/bubble_dynamics/keller_miksis/validation/dynamics.rs:44` | `<1.0 m/s²` is relative ~1.7e-7 vs the ≈5.8e6 m/s² characteristic accel — already at the FP floor, NOT loose. Documented in-place. No change. | — |
| PHY-10 | ~~M~~ DONE (2026-06-01) | `acoustics/bubble_dynamics/epstein_plesset/tests.rs:128` | Added `test_minnaert_constant_matches_literature_value`: `f₀·R₀≈3.26 m·Hz` (Minnaert 1933; Leighton 1994), max_rel=0.02. Independent of code's own formula. | done |
| PHY-11 | M (adequate; backlog) | `acoustics/bubble_dynamics/gilmore/tests.rs` | Analytical differential check (rel_err<1e-10 vs closed-form RHS) already present — strongest tier short of a citable collapse dataset. Lauterborn Rmax/R₀ regression deferred [minor] (won't fabricate a reference). | backlog |
| PHY-12 | M [verify] | `acoustics/bubble_dynamics/heterogeneous_nucleation.rs:129` | `T::from(16π/3).expect()` generic-cast precision/panic at f32 | static bound / document |
| PHY-13 | M | `acoustics/imaging/modalities/ceus/.../scattering.rs` | no scattering-cross-section ref test (de Jong 1991) | add quantitative test |
| PHY-14 | L | `acoustics/bubble_dynamics/gilmore/mod.rs:289+` | RK4 `unwrap_or(0.0)` silences u≈c singularity | log/diagnostic |
| PHY-15 | L | multiple (`kzk.rs:38`, `rayleigh_plesset/mod.rs:14`, `attenuation.rs`) | approximation validity criteria incomplete/uncited | document bounds + refs |

<a id="clinical-domain"></a>
## clinical/ + domain/ (therapy planning · imaging recon · transducers · grid/medium/source)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| CLD-1 | C | `clinical/therapy/lithotripsy/cavitation_cloud.rs:110` | full bubble-cloud dynamics unimplemented (Mach/multi-bubble missing) | phase-2 impl + k-wave compare |
| CLD-2 | C → **M DOC'D (2026-06-01)** | `orchestrator/{execution.rs:61,methods.rs:execute_therapy_step}` | `generate_acoustic_field` already documented its linear limitation; added orchestrator-level note (used by ALL modalities incl. HIFU; KZK exists, unwired). **Open follow-up:** wire `kzk_solver_plugin` into the HIFU therapy path [minor]. | partial |
| CLD-3 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/hifu_planning/types.rs:60` | Rewrote "Theorem"→"closed-form approximation" w/ validity regime (linear/paraxial F#≳1/homogeneous) + refs (O'Neil 1949, Cobbold 2007); named the magic 0.7 `MINUS6DB_ELLIPSOID_FILL_FACTOR` + flagged unvalidated (value preserved). | done |
| CLD-4 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/physics/mod.rs:47,50` | Category mismatch: `TISSUE_IMPEDANCE` is the nominal *matching-layer design load* (fixed manufactured hardware, `Z_match=√(Z_pzt·Z_load)`, Szabo/Cobbold), NOT a per-voxel sim medium — CT-derivation does not apply; documented to prevent re-flag. `BACKING_IMPEDANCE` was DEAD (no refs) — removed. | done |
| CLD-5 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/phased_array/config.rs:34` | "Ignores user freq" is false — `Default` is correctly nominal; no constructor drops a passed freq; `satisfies_nyquist` already takes `sound_speed`. Real defect was SSOT dup of `2.5` (geometry + freq field) → single `DEFAULT_CENTER_FREQUENCY_HZ` const. | done |
| CLD-6 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/lithotripsy/bioeffects.rs:191` | Documented Pennes-perfusion omission + its CONSERVATIVE (over-estimating) direction for a safety index; cited Pennes 1948; pointer to bioheat solver for quantitative dose. | done |
| CLD-7 | H | `clinical/therapy/therapy_integration/orchestrator/microbubble.rs:197` | uniform microbubble conc; no advection/cluster dynamics | document/extend |
| CLD-8 | M | `domain/boundary/bem/manager/assembly.rs:85` | `.unwrap()` on `last()` w/o bounds | safe `.last().copied()` |
| CLD-9 | M | `clinical/.../hifu_planning/tests.rs:115,156` | focal-spot tested only vs itself, not k-wave/analytic | add reference baseline |
| CLD-10 | M | `domain/source/transducers/focused/bowl/tests.rs:20` | bowl geometry tested, pressure field NOT vs k-wave | add field test |
| CLD-11 | M → reflection DONE (2026-06-01); Courant open | `domain/boundary/cpml/config/cpml_config.rs:214` | Added `theoretical_reflection_decays_monotonically_with_thickness` (Collino&Tsogka 2001): strict-decrease + bounded-(0,target] property test, params analytically chosen to avoid FP underflow. **Open:** a Courant-vs-thickness stability test (Komatitsch&Martin) remains a distinct sub-item. | partial |
| CLD-12 | ~~M~~ RESOLVED (2026-06-01) | `clinical/imaging/reconstruction/transcranial_ust/medium.rs:14` | `AIR_REJECTION_HU=-300` was a verbatim SSOT DUP of canonical `ct_acoustics::HU_BRAIN_BODY_THRESHOLD=-300` (Aubry 2003 ref). Deleted local const, switched 8 call sites (medium.rs+volume.rs) to canonical. Value drives a *qualitative* slice-selection count (robust to ±100 HU), not a calibrated mapping — no scanner-validated tolerance test warranted. | done |
| CLD-13 | ~~M~~ DONE (2026-06-01) | `domain/imaging/photoacoustic/types.rs:21,127` | Added `PressureFieldSeries` newtype (own leaf `pressure_series.rs`) wrapping `Vec<Array3<f64>>` with a validating constructor (non-empty + dimensionally uniform) and `Deref<[Array3<f64>]>` (zero consumer churn — all slice/`iter`/index callers unchanged). Both struct fields + 3 construction sites wrapped. 4 value-semantic ctor tests (accept/empty/ragged/round-trip). NB: `Array3<f64>` isn't a primitive — the captured invariant is intra-series dimension consistency, not a unit marker; cross-field time-alignment stays test-covered. | done |
| CLD-14 | ~~L~~ DONE (2026-06-01) | various | Audit framing ("uncited magic numbers") was largely false: `LENS_CURVATURE_FACTOR=0.7` already named; `crosstalk 0.1` already `// 10% (typical)`-commented; both erf impls already cited A&S 7.1.26. Real finding = DUPLICATION: two identical A&S 7.1.26 erf copies (`histotripsy.rs`, `clinical_scenarios/scenario/mod.rs`). Hoisted to canonical `math::statistics::erf` (named const + cite + error bound + 3 value-semantic tests); both sites delegate. SSOT. | done |

<a id="analysis-math"></a>
## analysis/ + math/ + core/ + gpu/ (beamforming/PAM/ML · operators/FFT · constants · shaders)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| AMC-1 | C [verify] | `math/numerics/operators/differential/central_difference_6/core.rs:95` | 6th-order stencil FMA nesting — verify signs vs Fornberg | add analytic-derivative value test |
| AMC-2 | L (verified: happy-path correct) | `analysis/.../beamforming/adaptive/mvdr/{spectrum.rs:45,weights.rs:36}` | `aᴴR⁻¹a` denominator checks `.re` only; imag dropped silently | add defensive `denom.im≪\|denom\|` Hermitian guard |
| AMC-3 | H | `analysis/.../localization/music/spectrum.rs:88`, `subspace/music.rs:108` | pseudospectrum hard-clamp (1e12/1e30) masks ill-conditioning | sentinel/error, not magic cap |
| AMC-4 | M (verified: live shader, persistence BC) | `gpu/shaders/acoustic_field.wgsl:41` | boundary `out=in` (persistence, NOT implicit-Dirichlet); used by `WaveEquationGpu`; undocumented + likely inconsistent w/ CPU Neumann/Dirichlet | document BC choice + reconcile with CPU paths |
| AMC-5 | M | `analysis/ml/physics_informed_loss/loss.rs:36` | wave-eq residual MSE unnormalized → scale-dependent loss | normalize by field scale |
| AMC-6 | M | `analysis/.../pam/delay_and_sum/processor/mod.rs` | no check delays fit signal duration | bounds validation |
| AMC-7 | M | `analysis/.../beamforming/covariance/estimation.rs:68` | parallel covariance accumulation order unguaranteed (FP) | document/forward-backward |
| AMC-8 | M | `analysis/ml/inference.rs:92` | `f32::EPSILON` normalization guard too tight | relative-ε guard |
| AMC-9 | ~~L~~ DONE (2026-06-01) | `analysis/.../beamforming/adaptive/subspace/{esmv,music}.rs` | Confirmed `Complex64≡num_complex::Complex<f64>` and eig/solve take/return that exact type → all 4 `mapv` round-trips + per-element rebuilds were identity. Removed: −2 `Array2` clones, −1 `Array1` clone, −per-element reconstruction. Value-identical (perf+clarity). | done |
| AMC-10 | ~~L~~ DONE (2026-06-01) | `narrowband/capon/mod.rs:115`, `mvdr/mod.rs:62` | Added `numerical::DEFAULT_DIAGONAL_LOADING=1e-6` (Carlson 1988 ref); both `Default`s now read it. SSOT. | done |
| AMC-11 | ~~L~~ DONE (2026-06-01) | `localization/music/mod.rs:86` | "Dup" was FALSE — `processor.rs:116` takes `frequency` as a param, not `fs/4`. Single site; named `DEFAULT_CENTER_FREQUENCY_NYQUIST_FRACTION=0.25` + justified (midpoint of Nyquist band). | done |
| AMC-12 | L | PAM `MUSIC`/`EigenspaceMinVariance` | return "not yet wired" errors (known stubs) | implement or document as unsupported |

---

## Triage order (per `CLAUDE.md` sprint policy: correctness → architecture → tests → docs)

1. ~~[verify] C-tier suspicions~~ — **DONE (Sprint A, 2026-05-31):** all 6 resolved,
   0 confirmed physics bugs (4 false positives closed; AMC-2→L, AMC-4→M downgraded).
2. ~~Confirmed correctness~~ — **DONE (Sprint B, 2026-05-31):** SOL-1/2/3 were all
   false positives (test-only panics, no change); PHY-5 fixed (dead-field removal).
3. ~~Documented-approximation bounds~~ — **DONE (Sprint C, 2026-06-01):** PHY-2/4
   already-handled/false-positive; PHY-8/CLD-2/3/6 validity regimes + refs added.
   Open follow-up: CLD-2 KZK wiring [minor].
4. ~~Missing literature validation~~ — **DONE (Sprint D, 2026-06-01):** PHY-10/SOL-7
   added external/property tests; CLD-11 reflection-decay test added; PHY-9 closed
   (false positive — tol already FP-tight); PHY-11 adequate (analytical check
   present, collapse regression backlogged). Remaining: PHY-13/CLD-9/10 (need
   k-wave/de-Jong baselines), CLD-11 Courant sub-item.
5. ~~CT-derived params over hardcoded~~ — **DONE (Sprint E, 2026-06-01):** CLD-4
   (category mismatch — design load not sim medium; dead BACKING_IMPEDANCE removed),
   CLD-5 (SSOT 2.5 dedup; "ignores freq" false), CLD-12 (verbatim SSOT dup of
   `HU_BRAIN_BODY_THRESHOLD` removed, 8 sites).
6. **DRY/SSOT + docs** — AMC-9/10/11 **DONE (Sprint E):** identity-cast removal
   (perf), `DEFAULT_DIAGONAL_LOADING` + `DEFAULT_CENTER_FREQUENCY_NYQUIST_FRACTION`
   consts. Remaining: CLD-13 (pressure-field newtype — public-API change, own
   increment), CLD-14 (scattered magic-number naming), SOL-10/11 (doc sweep, CI-wire).

See [backlog.md](backlog.md) for sprint sequencing and [CHECKLIST.md](CHECKLIST.md)
for the active increment.

---

# Coverage & placement audit (2026-06-19)

**Different axis from Sprints A–E above.** Those audited *internal correctness* of
existing code. This audit asks two new questions: (1) **coverage** — what ultrasound
physics do peer libraries (k-Wave, Field II, FOCUS, Stride, j-Wave, mSOUND, USTB,
MUST, BabelBrain) implement that kwavers lacks? (2) **placement** — what physics is
duplicated across crates or living in the wrong layer?

**Method:** four parallel read-only coverage explorers (forward solvers; bubble/
thermal/chemistry; transducer/imaging/beamforming; inverse/therapy/medium), then
**direct verification** of every `ABSENT`/`NOT FOUND` claim by targeted grep — the
explorers are pattern-matchers and over-call gaps (one example: explorer flagged
"Kirchhoff migration ABSENT" — it exists at
`kwavers-physics/src/acoustics/imaging/seismic/kirchhoff.rs`, with `eikonal.rs`
alongside). Only grep-confirmed gaps are listed.

**Headline:** kwavers' physics breadth meets or exceeds every peer surveyed
(uniquely: frequency-domain CBS-FWI, full PINN stack, neuromodulation HH+NICE,
sonochemistry, transcranial CT aberration). Gaps are narrow and concentrated in
classic **imaging-pipeline beamforming refinements** and a few **bubble-shell
models**; the larger risk is **cross-crate fragmentation** of three modality
verticals.

## Coverage gaps (grep-verified absent)

| ID | Sev | Area | Gap (peer that has it) | Notes |
|----|-----|------|------------------------|-------|
| COV-1 | ~~M~~ **DONE (2026-06-19)** | beamforming | Added `time_domain::coherence` — Mallart-Fink amplitude CF + Camacho sign CF (SCF) behind one `CoherenceFactor` enum + `delay_and_sum_coherence`; DAS refactored onto SSOT `align_channels`/`sum_aligned`. 11 value-semantic tests. **Surfaced + fixed a real bug:** SAFT 3-D CF squared `Σ|x|` instead of summing energies (coherent aperture → 1/N not 1); consolidated onto canonical `amplitude_coherence_from_sums`. NB: SLSC (Lediju 2011) + SAFT-CF already existed — gap was the canonical DAS-path CF/SCF, now filled. **Follow-up [minor]:** PCF (phase, needs analytic signal) + GCF (Li 2003, FFT). | done |
| COV-2 | ~~M~~ **DONE (2026-06-19)** | beamforming | Added `time_domain::dmas` — canonical `dmas_combine` (signed-sqrt pairwise closed form) + active `delay_and_sum_dmas` (reuses `align_channels`). **Consolidated:** passive PAM `dmas_at_point_view` now calls the shared `dmas_combine` (was inline-duplicated). 8 value-semantic tests (closed-form pairwise products, anti-phase suppression). | done |
| COV-3 | ~~M~~ **DONE (2026-06-19)** | transducer | Added `kwavers-transducer::curvilinear::ConvexArrayGeometry` — element positions + outward radial normals + tangents on a curvature arc, transmit-focusing delays, aperture/arc-pitch geometry. 8 analytic tests (on-arc, apex, unit-radial normals, chord width, zero-delay-at-curvature-center, symmetry). Feeds `kwave_array` Rect/Arc elements or a `Source` (rasterization = follow-up). | done |
| COV-4 | ~~M~~ **DONE (core, 2026-06-19)** | phantom | Added `kwavers-phantom::scatterers` — `ScattererCloud` (Field II tissue model) + monostatic synthetic-aperture `synthesize_rf`: `RF_e(t)=Σ_s (a_s/r²)·pulse(t−2r/c)`. 7 analytic tests (round-trip delay, 1/r² amplitude, superposition, linearity, pulse placement, min-distance guard). **Follow-up:** finite-aperture Tupholme–Stepanishen spatial impulse response (point-element model is the exact far-field limit) + frequency-dependent attenuation. | done (core) |
| COV-5 | ~~M~~ **PARTIAL (2026-06-19)** | bubble dynamics | Added **Hoff (2000)** + **Sarkar (2005)** shell models as `EncapsulatedShellModel` impls (+ value-semantic tests incl. Hoff≡Church-at-G_s=0 differential). **Deferred [minor]:** de Jong (lumped S_p/S_f prefactor is convention-dependent — needs Doinikov&Bouakaz PDF verification before asserting) and Herring (free-bubble compressible EOM — different category, belongs with KM/Gilmore, not a shell model). Evidence tier: literature-recall (Doinikov&Bouakaz 2011) validated by equilibrium/restoring/damping properties. | partial |
| COV-6 | L | transducer | **KLM / Mason equivalent-circuit** transducer model absent; impulse response handled analytically (`pulse_echo.rs`). | Field II/PiezoCAD feature; needed only for electro-mechanical bandwidth design, not field sim. Defer unless a design use-case appears. |
| COV-7 | L | elastography | **MRE-specific inversion** (harmonic MR displacement → modulus) absent; SWE/ARFI/thermal-strain present, viscoelastic Voigt/Maxwell/Zener present. | Most MRE machinery (LFE, direct inversion) already exists in `inverse/elastography/linear_methods`; gap is the MR displacement-field front end. |
| COV-8 | ~~L~~ **FALSE POSITIVE (2026-06-19)** | sonoluminescence | NOT a stub: `cherenkov/model.rs` has the full Frank-Tamm formula (Jackson 1999 §13.5) — `frank_tamm_factor`, `spectral_intensity ∝ f`, `emission_spectrum ∝ 1/λ³`, threshold logic. Complete + literature-grounded. No gap. | — |
| COV-9 | ~~L~~ **FALSE POSITIVE (2026-06-19)** | inverse | NOT a dead config: `apply_sobolev_preconditioner_3d` (`linear_born_inversion/pcg.rs:232`) is a real Sobolev-gradient smoothing preconditioner (`smooth_active_values_3d` + convex blend), wired into the PCG iteration (pcg.rs:210). No gap. | — |
| COV-10 | L | phantom | **Shepp-Logan / standard numerical test phantom** absent. | Test-fixture convenience, not physics; low priority. |
| COV-11 | L | boundary | **Mur absorbing BC** absent. | CPML/PML present and superior; **WONTFIX** unless a thin-PML budget case appears. |

**Confirmed NON-gaps (explorer false positives — do NOT re-flag):** Kirchhoff
migration (`seismic/kirchhoff.rs`), eikonal (`seismic/eikonal.rs`), Rytov
(`inverse/rytov.rs`), power/vector Doppler, ULM super-resolution, axisymmetric PSTD,
fractional-Laplacian + multi-relaxation absorption, anisotropic Christoffel —
all present and accounted for.

## Placement / SSOT gaps (cross-crate fragmentation — grep-verified)

| ID | Sev | Concern | Evidence | Resolution direction |
|----|-----|---------|----------|----------------------|
| PLC-1 | arch | **Photoacoustic spread across 5 crates** incl. **two parallel impls inside `kwavers-simulation`** (`modalities/photoacoustic/` AND `photoacoustics/vertical/`), plus `kwavers-physics/photoacoustics`, `kwavers-imaging/photoacoustic`, `kwavers-diagnostics/photoacoustic`, and `kwavers-solver/.../reconstruction/photoacoustic` (inversion — legitimately separate). | Extends and **supersedes DEBT-3** (which recorded only the 2 in-simulation copies). The fragmentation is wider than DEBT-3 states. | Pick ONE canonical forward modality (physics) + ONE recon home (solver); collapse the simulation duplicate and fold imaging/diagnostics shims into orchestration-only. ADR required. |
| PLC-2 | arch | **CEUS spread across 4 crates with duplicated physics** — `perfusion` modeled in BOTH `kwavers-imaging/ultrasound/ceus/perfusion.rs` AND `kwavers-physics/.../ceus/perfusion/`; microbubble scattering/dynamics in both; plus `kwavers-simulation/imaging/ceus.rs` and `kwavers-imaging/ceus_orchestrator.rs`. | Perfusion kinetics + bubble scattering are physics; modeling them in `kwavers-imaging` is an upward-layer duplicate. | Consolidate CEUS *physics* (microbubble response, scattering, perfusion kinetics) into `kwavers-physics`; leave only acquisition/orchestration in `kwavers-imaging`. |
| PLC-3 | arch — **shell-model SSOT DONE (2026-06-19)** | **Microbubble dynamics duplicated within `kwavers-physics`** (`acoustics/imaging/modalities/ceus/microbubble` vs `therapy/microbubble`) AND a **`therapy/` subtree living in the physics crate** (`kwavers-physics/src/therapy/microbubble`, and `acoustics/therapy/{neuromodulation,sonogenetics}`, `acoustics/transcranial/bbb_opening`) parallel to `kwavers-therapy`. | **Done:** encapsulated-shell models now share one `EncapsulatedShellModel` trait + RP driver (Church/Marmottant/Hoff/Sarkar) — no more cloned RP arithmetic. **Remaining:** the ceus/microbubble vs therapy/microbubble duplication and the therapy-in-physics layering (move therapy consumers to `kwavers-therapy`, keep models in physics). | partial |
| PLC-4 | M | **Time-reversal in 3 locations** — `physics/.../transcranial/aberration_correction/time_reversal.rs`, `simulation/modalities/photoacoustic/reconstruction/`, `solver/inverse/time_reversal` + `solver/.../reconstruction/photoacoustic/time_reversal.rs`. | Likely 3 *distinct* uses (aberration vs PA recon vs general TR), but the TR k-space propagator core should be one SSOT (cf. memory `project_tr_ssot_may26`). | Verify whether the propagator kernel is shared; if cloned, hoist to one `solver` entry point. **[verify] before asserting dup.** |
| PLC-5 | L | **Histotripsy parameters in 3 crates** — `kwavers-medium/absorption/histotripsy.rs` (absorption db), `kwavers-physics/analytical/cavitation/histotripsy.rs` (intrinsic threshold), `kwavers-therapy/.../lithotripsy`. | These are arguably *distinct concerns* (medium props vs cavitation threshold vs therapy planning) — likely correct layering, not duplication. | **Low priority / probably WONTFIX**; confirm no shared-constant drift. |

**Severity:** `arch` cross-cutting structural · `M` real but bounded · `L` cleanup.
Placement items are **[verify]-gated for duplication**: confirm the logic is actually
cloned (not legitimately layered forward-vs-inverse) before consolidating — same
discipline that turned 6/6 Sprint-A suspicions into 4 false positives.

See [backlog.md](backlog.md) for sprint sequencing and [CHECKLIST.md](CHECKLIST.md)
for the active increment.
