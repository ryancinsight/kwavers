# Implementation Backlog — Capabilities Documented in the Book but Not (Yet) in Code

This backlog tracks components the per-chapter book audits found **documented but not
implemented** in `kwavers`. The audit policy is: a documented capability must either be
implemented or the book must mark it as theory/not-implemented. The chapters have been made
honest (theory-only markers added); this backlog drives closing the gaps with real,
tested implementations.

Tags: change-class `[patch]/[minor]/[major]/[arch]`. Evidence tier required: property/
analytical test with value-semantic assertions (no `is_ok()`-only).

## Verification note (important)

Several earlier "NOT FOUND" results were **name-only false negatives** — the capability
exists under a different type name. Always re-verify by *algorithm* (grep the math, not the
claimed struct name) before implementing. Confirmed corrections below.

## Done

- ✅ **Acoustic transmission coefficient — `1−|R|` corrected to pressure `T=2Z_t/(Z_i+Z_t)`** —
  `[patch]` (2026-06-10, audit). `AcousticMaterialProperties::transmission_coefficient` returned
  `1 − |R|` — **neither** the pressure amplitude (`T_p=1+R=2Z_t/(Z_i+Z_t)`) nor the intensity
  (`T_I=1−R²`) transmission, and energy-non-conserving. For `Z_t>Z_i` (`R>0`) it gave `1−R` instead of
  `1+R` (wrong; only accidentally correct when `Z_t<Z_i`). The sibling `acoustic_elastic` coupling
  trait already had the correct `2Z₂/(Z₁+Z₂)`. Added an SSOT free `transmission_coefficient(z_i,z_t)`
  (pressure `T_p`, companion to the existing pressure `reflection_coefficient`) and delegated the
  method to it. Test: closed form, `T>1` into stiffer medium, `T=1+R`, lossless balance
  `R²+(Z_i/Z_t)T²=1`, matched-impedance `T=1`. (The skull/transducer siblings correctly use the
  *intensity* `4Z₁Z₂/(Z₁+Z₂)²` — left as-is, documented.)
- ✅ **Wire Christoffel into `AnisotropicStiffnessTensor`** — `[minor]` (2026-06-11). The Christoffel
  solver was export-only/unwired; the stiffness tensor (SSOT held by media) now exposes Christoffel-
  backed `phase_velocities`/`group_velocities` and `max_phase_velocity` (CFL reference speed — the
  direction supremum of the quasi-P branch, sampled over principal/face/body axes + a 96-point
  Fibonacci sphere). 1 test: isotropic `max == √((λ+2μ)/ρ)` exactly; TI `max ≥` on-axis qP.
- ✅ **Anisotropic group (energy) velocity** — `[minor]` (2026-06-10). The (now-correct) Christoffel
  solver had phase velocities + polarizations but **no group/energy velocity** — the quantity along
  which energy actually propagates (it walks off the phase direction in anisotropic media). Added
  `ChristoffelEquation::group_velocities` = `V_{g,i} = (1/ρV_p) Σ_jkl C_ijkl p_j p_k n̂_l` (Auld 1973
  §7), with a Voigt→full-tensor `c_ijkl` accessor (consistent with `christoffel_matrix`). 2
  value-semantic tests: **isotropic `V_g = V_p·n̂`** exactly for qP and both qS modes off-axis
  (magnitude = phase speed, parallel to n — the discriminating contraction check); TI medium gives
  finite components with an axial qP energy velocity on the symmetry axis; degenerate inputs rejected.
  Completes the anisotropic-wave API (phase velocity + polarization + group velocity).
- ✅ **`LinearAlgebra::qr_decomposition` — returned Qᵀ; fixed via nalgebra** — `[patch]` (2026-06-10,
  codebase audit). The hand-rolled Householder QR accumulated `Q ← Hⱼ·Q` (left-multiply) → `Q = Hₙ…H₁
  = Qᵀ`, so `A = QᵀR` not `A = QR` (the documented contract). It was **untested and unused** (dead
  public API; the eigendecomposition QR is a separate, correct helper). Delegated to nalgebra's
  Householder QR (same pattern as `svd` in the same file), removing the buggy hand-rolled version,
  and documented it as the reduced (thin) QR `A=Q·R`, `Q` m×k orthonormal, `R` k×n upper-triangular.
  New test: reconstruction `A=Q·R` + `QᵀQ=I` + upper-triangular `R` for square and over-determined
  matrices.
- ✅ **Real symmetric eigendecomposition — malformed Jacobi angle fix** — `[patch]` (2026-06-10,
  codebase audit). `EigenDecomposition::eigendecomposition` (real symmetric, used by MVDR
  beamforming) had a **mathematically malformed Jacobi rotation angle** in its `else` branch:
  `0.5(a_qq−a_pp)/atan2(a_pq, (a_qq−a_pp)/(2a_pq))` — not the Jacobi angle `½atan(2a_pq/(a_pp−a_qq))`.
  A wrong angle never annihilates the off-diagonal, so it can't converge → wrong eigenvalues for any
  **unequal-diagonal** matrix. The only test used `[[2,1],[1,2]]` (equal diagonals → correct π/4
  branch), hiding the bug. Fixed by **delegating the real path to the correct, reconstruction-tested
  complex Hermitian Jacobi** (a real symmetric matrix is Hermitian; eigenvectors stay real → `.re`),
  deleting ~70 lines of broken hand-rolled real Jacobi (DRY). 2 new tests: `[[4,1],[1,2]]` recovers
  `3±√2` with `Av=λv` + `A=VΛVᵀ`; a 3×3 reconstructs exactly with `Σλ=tr` + descending order. 50
  linear-algebra tests pass.
- ✅ **Christoffel anisotropic wave-speed solver — HARD-violation + degenerate-case fix** — `[patch]`
  (2026-06-10, codebase audit). `ChristoffelEquation::phase_velocities` used a hand-rolled Cardano
  cubic solver with a hardcoded `[1.0, 1.0, 1.0]` **fallback** when the discriminant ≤ 0 — a
  HARD-prohibited mock. But the Christoffel matrix is real-symmetric, so its characteristic cubic has
  a **repeated root (discriminant = 0) for every isotropic / on-axis medium** (two equal quasi-shear
  speeds), meaning `phase_velocities` returned bogus `[1,1,1]` for the most common case. Replaced the
  Cardano+fallback with `nalgebra::SymmetricEigen` (already used by `polarization_vectors`),
  unified both methods through one `sorted_eigen` (descending eigenvalue → qP, qS1, qS2; DRY;
  velocity↔polarization order now aligned), and added a positive-density guard (`Err`, not NaN). 3
  value-semantic tests: isotropic recovers exact Lamé `c_P=√((λ+2μ)/ρ)`, `c_S=√(μ/ρ)`×2 along
  multiple directions (the formerly-broken case); `Σρv²=tr(Γ)` invariant on a transversely-isotropic
  tensor; isotropic qP-longitudinal / qS-transverse polarizations.
- ✅ **Acousto-optic diffraction — complete theory (Raman–Nath / Bragg / Klein–Cook)** — `[minor]`
  (2026-06-09, user request). kwavers previously had only the photoelastic Δn=p_e·p field coupler
  (`AcousticOpticalSolver`), no diffraction model. Added the complete theory in new
  `kwavers_physics::analytical::acousto_optics`: `klein_cook_parameter` Q=2πλ₀L/(nΛ²) +
  `diffraction_regime` classifier; **Raman–Nath** thin-grating orders `Iₘ=Jₘ²(ν)`
  (`raman_nath_order_intensities`, `raman_nath_parameter`); **Bragg** thick-grating efficiency
  `η=sin²(ν/2)` (`bragg_diffraction_efficiency`); `diffraction_angle_rad` (grating equation,
  evanescent cut-off); and the general **Klein–Cook coupled-wave solver** `solve_coupled_orders`
  (RK4 on `dEₗ/dξ=−i(ν/2)(Eₗ₋₁+Eₗ₊₁)−i(Q/2)(l²+2lα)Eₗ`). 8 value-semantic tests: regime
  classification; Raman–Nath = exact Bessel + symmetry + `Σ Jₘ²=1` energy; Bragg closed form;
  diffraction angles; **the coupled solver reproduces Raman–Nath as Q→0** and **Bragg `sin²(ν/2)`
  at large Q (α=−½)**; energy conserved in all regimes. Completed (2026-06-10) with the
  **order frequency shift** `Δf=m·f_a` (`diffraction_frequency_shift_hz`, the AOM/frequency-shifter
  principle) and the explicit **Bragg angle** `θ_B=arcsin(λ₀/2nΛ)` (`bragg_angle_rad`), both tested
  (θ₁=2θ_B geometry, no-solution cut-off). Wired `AcousticOpticalSolver::diffraction_orders`
  to delegate to the model (+2 solver tests cross-checking against the closed form). Full PyO3
  bindings (6 fns) + `__init__.py` re-export; python crate compiles. Klein & Cook (1967); Korpel;
  Saleh & Teich §20.
- ✅ **Soft-tissue temperature-dependent Grüneisen (PA thermometry)** — `[minor]` (2026-06-09,
  sensors-chapter audit). The chapter said "temperature coupling is not currently implemented" —
  **partly stale**: `gruneisen_parameter_water(T)` (Sigrist & Kneubühl `Γ=0.0043+0.0053T`) already
  existed; the soft-tissue function did not, despite the SSOT constants
  `GRUNEISEN_SOFT_TISSUE`/`GRUNEISEN_SOFT_TISSUE_TEMP_COEFF` + documented formula. Added
  `kwavers_physics::analytical::photoacoustics::gruneisen_parameter_soft_tissue(T)` =
  `Γ_body + (dΓ/dT)(T−T_body)` (Xu & Wang 2006), using the SSOT constants; PyO3 binding +
  `__init__.py` re-export (python crate compiles). 1 value-semantic test: exact reference value at
  body temp, closed form at +10 °C, monotone increase, and the PA-thermometry sensitivity `dΓ/°C`.
  Chapter corrected (Γ(T) is available analytically for water+tissue; remaining gap is wiring it into
  the full-wave PA simulation medium during a coupled thermal–acoustic run).
- ✅ **Fay (post-shock sawtooth) harmonic solution (Ch3 §3.6)** — `[minor]` (2026-06-09, Ch3 audit).
  The Fubini pre-shock harmonic solution existed (`fubini_harmonic_amplitude/spectrum/waveform`) but
  the complementary **Fay/sawtooth** post-shock solution — half of the Blackstock (1966) Fay–Fubini
  pair the chapter cites — was absent. Added `kwavers_physics::analytical::wave::{sawtooth_harmonic_amplitude,
  sawtooth_harmonic_spectrum}` — `Bₙ(σ)=2/(n(1+σ))` (lossless sawtooth, Hamilton & Blackstock 1998
  §4.4). 4 value-semantic tests: 1/n harmonic decay (sawtooth signature, exact at multiple σ);
  1/(1+σ) distance decay + closed form; spectrum/degenerate; the **Fay–Fubini discontinuity at σ=1**
  (Fubini fundamental ≈0.88 vs sawtooth 1.0). Re-exported via `analytical::wave`. Chapter §3.6
  (Corollary 3.5) added.
- ✅ **Direct shear-wavelength estimator (§11.10.3)** — `[minor]` (2026-06-09, Ch11 audit). §11.10.3
  flagged "no dedicated wavelength estimator is implemented." Added
  `kwavers_analysis::signal_processing::wavelength_estimation::estimate_shear_wavelength(u, dx)` —
  estimates `λ_S` directly from a displacement profile (no known `c_S`) via the biased spatial
  autocorrelation `R(m)∝cos(2π m·dx/λ)`, whose first post-zero-crossing peak sits at lag `λ`,
  parabolically interpolated (FFT-free, zero-dep). 4 value-semantic tests: recovers a known
  wavelength to <2% (sub-sample) across 3 wavelengths; DC-offset-invariant; scales linearly with
  `λ`; rejects constant/monotone/too-short/`dx≤0`. Chapter §11.10.3 updated.
- ✅ **Acoustic gene-expression kinetics (sonogenetics §17.13.2)** — `[minor]` (2026-06-09, Ch17
  audit). §17.13.2 flagged "gene expression kinetics require coupling to a pharmacokinetic model
  (not implemented)." Added `kwavers_physics::analytical::sonogenetics::GeneExpressionKinetics` — the
  standard linear two-stage central-dogma / PK–PD cascade `dm/dt=β·a−δ_m m`, `dp/dt=κ·m−δ_p p`
  driven by the acoustic channel-activation level `a(t)` (couples to `hill_activation_probability`),
  with RK4 `step_rk4`/`integrate` and closed-form `steady_state`. 4 value-semantic tests: rate
  validation; **integrated trajectory reaches the closed-form steady state** `m_ss=βa/δ_m`,
  `p_ss=κβa/(δ_m δ_p)`; **transcript transient matches the analytic `m_ss(1−e^{−δ_m t})`**; linearity
  in activation + wash-out to ~0 after a pulse. Honestly scoped as the lumped linear model (full
  promoter/capsid molecular kinetics deferred). Chapter §17.13.2 updated.
- ✅ **METAVIR liver-fibrosis staging classifier (Algorithm 11.5)** — `[minor]` (2026-06-09, Ch11
  audit). §11.11.5 flagged "no dedicated elastography tissue classifier is implemented." Added
  `kwavers_analysis::signal_processing::tissue_staging`: `FibrosisStage{F0..F4}` (ordered),
  `classify_liver_fibrosis(μ_kPa)` + `classify_liver_fibrosis_from_speed(c_S, ρ)` (μ=ρc_S²) using the
  validated METAVIR cut-offs `[1.7,2.9,4.8,9.0]` kPa (half-open intervals), and `classify_liver_roi`
  implementing Algorithm 11.5's ROI logic (median stage + heterogeneity flag `IQR>0.3·median`). 4
  value-semantic tests: every stage, boundary convention (on-cutoff→higher stage), speed→μ path,
  ROI median+heterogeneity+invalid-sample filtering. Chapter §11.11.5/§11.13 updated. (Prostate/
  thyroid/breast staging tables remain reference data — a clean extension.)
- ✅ **Bootstrap confidence intervals (elastography CRLB module)** — `[minor]` (2026-06-09, Ch11
  audit). §11.12 flagged "bootstrap confidence intervals are not yet implemented" (the CRLB bounds
  existed). Added `kwavers_analysis::signal_processing::estimation_bounds::bootstrap_ci_mean` —
  percentile bootstrap (Efron 1979) of the sample mean with a self-contained **deterministic seeded
  PRNG** (splitmix64; reproducible, no `rand` dep) + `BootstrapCi{point,lower,upper}`. 4
  value-semantic tests: CI brackets the point + bit-identical from a fixed seed; **half-width tracks
  the analytical `1.96·σ/√N` standard error** (within 35%); widens with spread + confidence level;
  degenerate (empty/single/invalid-level) cases. Chapter §11.12/§11.13 updated (also cleared the
  stale "theory only" list — Murnaghan §11.9 + acousto-elastic pre-stress inversion are implemented).
- ✅ **Viscoelastic dispersion-fitting inversion (shear-wave spectroscopy)** — `[minor]`
  (2026-06-09, Ch11 audit). §11.8 flagged "A dispersion-fitting inversion kernel is not yet
  implemented" (the KV/Zener *forward* models existed). Added to `kwavers_medium::viscoelastic`:
  `recover_complex_modulus(ω,c_p,α,ρ)` — model-agnostic `G*=ρ(ω/k)²`, `k=ω/c_p−iα` (the physical
  branch where `Im k<0` for `G*=μ+iωη`) — and `KelvinVoigtModel::fit_dispersion(samples, ρ)`
  recovering `μ=⟨Re G*⟩`, `η_s=⟨Im G*/ω⟩` (Catheline 2004 / Deffieux 2009, the refs the chapter
  already cites) + a `shear_viscosity()` accessor and `DispersionSample`. 2 value-semantic tests:
  `recover_complex_modulus` exactly inverts the forward dispersion (caught + fixed a wavenumber sign
  error), and a forward→inverse round-trip recovers known `(μ,η_s)` to <0.1%. Chapter §11.8 updated.
  **Zener 3-param fit DONE (2026-06-09):** `ZenerModel::fit_dispersion` by **separable least
  squares** — for fixed `τ` the storage/loss are linear in `(G_r, Δ)` (closed-form 2×2 solve), the
  nonlinear `τ` found by log-scan + golden-section refinement of the stacked `G'/G''` residual; added
  `ZenerModel::{attenuation, relaxed_modulus, unrelaxed_modulus, relaxation_time}` accessors. Round-trip
  test recovers `(G_r,G_u,τ)` across the `ωτ=1` Debye peak to ≤2%. 12 viscoelastic tests pass.
- ✅ **Doc drift — Ch11 transducer model availability** — `[patch]` (2026-06-09). §11.12 claimed
  "CMUT and Mason-circuit models … are documented as theory (§2) and are not implemented" —
  **stale**: `BulkPiezoResonator` (Mason/IEEE) and `mems::{CmutCell, PmutCell}` (with collapse-mode,
  crosstalk, flexible-array beamformer) are all implemented. Chapter corrected to cite them.
- ✅ **Minnaert surface-tension correction (Eq 5.6)** — `[minor]` (2026-06-09, Ch5 audit). The
  chapter flagged "(5.6) not implemented"; only the large-bubble form (5.7) existed
  (`minnaert_resonance_hz`). Added `kwavers_physics::analytical::cavitation::minnaert_resonance_corrected_hz`
  — `f₀ = (1/2πR₀)√([3γP₀ + (3γ−1)·2σ/R₀]/ρ)` (Young–Laplace stiffness; Leighton 1994 §3.2),
  reducing to (5.7) as σ→0; returns 0 if surface tension destabilises the bubble. 3 value-semantic
  tests: σ→0 reduction (exact), closed-form match, negligible large-bubble limit (<0.1% at 1 mm),
  and >10% sub-micron correction (R₀=1 µm). PyO3 binding `minnaert_resonance_corrected_hz` added
  (registered + `__init__.py` re-export; python crate compiles). Chapter §5 updated.
- ✅ **Doc drift — elastography inversion methods** — `[patch]` (2026-06-09, Ch11 audit). §11.7.2
  claimed "regularised global Helmholtz / LFE inversion is not yet implemented" — **stale**: the
  `direct` method *is* a regularised global Helmholtz inversion (`J=‖∇²u+k²u‖²+λ‖∇k²‖²`,
  Gauss–Seidel), and LFE (`InversionMethod::LocalFrequencyEstimation`), directional phase-gradient
  (Wang 2014), and 3-D ToF are all implemented. Chapter corrected to cite them.
- ✅ **Ultrasonic neuromodulation — electrical (capacitive) pathway** — `[minor]`.
  New `kwavers_physics::acoustics::therapy::neuromodulation` module, closing the
  Blackmore et al. (2019) mechanism-(i) gap (membrane capacitance / intramembrane
  cavitation) that complements the existing mechanosensitive-channel sonogenetics
  (mechanism (ii)) and thermal (Yoo) pathways. Contents: (a) genuine Hodgkin–Huxley
  conductance neuron (`HhParams`/`HhState`/`simulate_hh`) validated against the 1952
  squid-axon reference — resting m∞/h∞/n∞, near-zero resting net current, sub-/supra-
  threshold response, ≈+40 mV AP overshoot, monotone f–I; (b) intramembrane-cavitation
  capacitance modulation `C_m(t)=C_m0(1+ε·sinωt)` with exact analytic `dC_m/dt` and a
  documented small-strain pressure→ε bridge `ε=p·R/(2K_A)`; (c) NICE coupling
  (`simulate_nice`) injecting the displacement current `−V·dC_m/dt` into HH. 14
  value-semantic tests. **Honest scope:** the symmetric-sinusoid capacitance gives a
  net depth-dependent *hyperpolarising* excitability shift (geometric `1/√(1−ε²)` term
  dominates the depolarising gating rectification); the asymmetric bilayer-sonophore
  waveform (below) is the one that reproduces net *excitation*.
  Follow-ups — **all now done** (2026-06-08):
  - ✅ `[minor]` **Asymmetric bilayer sonophore + NICE excitation** (`bls.rs`): exact
    curved-dome capacitance `C_m(Z)` (Plaksin 2014 Eq. 8) + kinematic leaflet deflection
    driving the displacement-current HH coupling. Reproduces the real NICE mechanism —
    membrane *hyperpolarisation during* US, net *charge accumulation*, and a
    *post-stimulus AP* with pulse-duration dependence. `CapacitanceSource` trait makes
    `simulate_nice` generic over the sinusoidal and BLS sources. Evidence: deflection
    `Z(t)` is a documented kinematic surrogate for the full BLS mechanical ODE (Eq. 2,
    molecular-force params not in the open preprint); Eqs. 1 & 8 reproduced exactly.
  - ✅ `[minor]` **SONIC cycle-averaged reduction** (`sonic.rs`, Lemaire 2019):
    charge-density slow integration with cycle-averaged HH gating; enables whole-protocol
    (second-scale) simulation. Differential-tested against carrier-resolved `simulate_nice`
    (matching spike count + AP timing within 1 ms).
  - ✅ `[minor]` **Pulse-train protocol + dosimetry** (`protocol.rs`): Blackmore Table 1
    parameter hierarchy (PL/PRF/BD/BI/N → BDC/TDC/BRF/TT), spatial-peak intensities
    (ISPPA/ISPBA/ISPTA), MI, FDA-limit screening, pulse envelope, and the
    Atkinson-Clement 2025 theta-burst preset (validated: DC=10%, ISPTA/ISPPA ratio).
  - ✅ `[minor]` **PyO3 bindings + book chapter**: `hodgkin_huxley_response`,
    `nice_bilayer_sonophore_response`, `nice_sonic_response`, `bilayer_capacitance_curve`,
    `pulse_train_dosimetry` (9 pytest cases); book §25.5.1 documents the electrical pathway.
  Honest-scope enhancements — **done** (2026-06-08, literature-grounded):
  - ✅ `[minor]` **Pospischil RS/FS cortical neuron** (`cortical.rs`, Pospischil 2008
    params confirmed vs PySONIC): Na/Kd/M-current/leak — the membrane Plaksin's NICE
    model actually uses. Introduced a `Membrane` trait (4-gate `[m,h,n,p]`); `simulate_nice`
    and `simulate_sonic` are now generic over it (squid HH stays validated). Cell-type
    selectivity tested (RS vs FS differ under identical drive). PyO3 `cortical_sonic_response`.
  - ✅ `[minor]` **ITRUSST safety** (`protocol.rs` `itrusst_assess`, Aubry 2024 consensus):
    MI ≤ 1.9 AND (ΔT ≤ 2 °C or brain CEM43 ≤ 2). PyO3 `itrusst_safety`.
  - ✅ `[minor]` **Pressure-driven (quasi-static) bilayer sonophore** (`bls_pressures.rs`):
    the full BLS force balance — intermolecular Eq. 4–5 (p_Δ=1e5, exponents 5/3.3, via
    quadrature), elastic tension k_A(Z/a)², electrical Maxwell stress Eq. 3, gas Eq. 6–7
    — with a rest-gap solver that **reproduces Plaksin's Δ≈1.26 nm from the resting-charge
    balance** (P_tot(0)=0). `quasistatic_deflection` solves Z from acoustic pressure
    (rectified: expands in rarefaction, flat in compression); validated vs Plaksin Fig. 1
    (≈10 nm, ≈20 mN/m @ 500 kPa/0.5 MHz). New `BilayerSonophoreQuasistatic` CapacitanceSource
    (pressure is the input, not deflection) evokes the post-stimulus AP through NICE. PyO3
    `nice_quasistatic_response`, `bls_deflection_curve`. Constants from PySONIC/Krasovitski 2011.
  - ✅ `[minor]` **Exact transient BLS dynamics** (`bls_dynamics.rs`, Plaksin Eq. 2 / PySONIC
    `derivatives`): full leaflet Rayleigh-Plesset ODE `[U,Z,ng]` — inertia, leaflet/fluid
    viscosity (−12Uδ₀μ_S/R²−4Uμ_L/|R|), molecular Eq.4-5, elastic, electrical, gas-diffusion
    flux — every constant verbatim from PySONIC. Z=0 curvature singularity (R→∞) handled
    EXACTLY as the reference (seed Z(0)=balanced_deflection(P_ac(dt)), no ad-hoc regularisation);
    adaptive step-doubling RK4 for the steric-wall stiffness. **Reproduces Plaksin Fig. 1:
    peak deflection ≈10–11 nm @ 500 kPa/0.5 MHz** (published ≈12), monotone in pressure,
    resonantly amplified above quasi-static. `BilayerSonophoreDynamic` CapacitanceSource
    (pressure-driven) evokes the post-stimulus AP. PyO3 `nice_dynamic_response`.
  **Caveat fully resolved**: deflection is now the exact transient solution (no quasi-static or
  kinematic approximation); lighter sources kept for speed/analysis.

- ✅ **Frangi vesselness / vasculature segmentation** — `[verify]` ALREADY IMPLEMENTED
  (false negative). `kwavers_analysis::signal_processing::vasculature` provides
  `compute_frangi_response` (multiscale Hessian-eigenvalue vesselness, Frangi 1998) and
  `VesselSegmentation::segment` (Frangi response + Otsu mask + centerline + flow). The audit
  searched for `FrangiFilter` (which doesn't exist); the real API is `VesselSegmentation`.
  Diagnostics §5.9 corrected (added a real vasculature code-map row; removed the wrong
  not-implemented marker).
- ✅ **CRLB estimation bounds** — `[minor]`.
  `kwavers_analysis::signal_processing::estimation_bounds`:
  `time_delay_crlb_variance` (Walker–Trahey `1/(8π²f₀²T_w·SNR)`), `time_delay_crlb_std`,
  `strain_crlb_std` (`c_P/(4πf₀√(T_w·SNR)·Δz)`), `shear_wave_speed_crlb_std`
  (`c_s²/(ω·L_x·√(N_t·SNR_v))`). 5 value-semantic tests (closed-form equality, SNR/bandwidth
  monotonicity, degenerate→∞). Elastography §10.12/§10.13 updated.
- ✅ **Residue-aware phase unwrapping (Goldstein detection + masked unwrap)** — `[minor]`
  (future-enhancement #5). `kwavers_signal::phase::goldstein::{phase_residues, residue_count,
  is_unwrap_reliable, masked_unwrap_2d}` — exact 2×2 plaquette residue charges, a reliability
  gate for the Itoh unwrapper, and a BFS flood-fill unwrap restricted to a validity mask
  (routes around residues; masked/unreachable → NaN), **automatic ground-to-border branch cuts**
  `goldstein_branch_cut_mask`, and the full `goldstein_unwrap_2d`. 6 value-semantic tests (smooth
  →0 residues; vortex→±1; masked plane recovery; residue-free Goldstein = plane; **dipole unwrap
  is seam-free/continuous**; degenerate). Elastography §11.13 updated. Full residue-aware MRE
  unwrapping now end-to-end.
- ✅ **Bulk-piezo thickness-mode resonator (Mason/IEEE)** — `[minor]` (future-enhancement #8).
  `kwavers_transducer::bulk_piezo::BulkPiezoResonator` (PZT-5H preset) — stiffened sound speed,
  antiresonance `f_p=c_D/2t`, clamped capacitance, series resonance `f_s` via bisection of the
  IEEE `k_t²=(πf_s/2f_p)tan(π(f_p−f_s)/2f_p)` relation, and `coupling_from_frequencies`. 4
  value-semantic tests (f_p∝1/t, sound-speed/capacitance formulas, f_s<f_p + kt² round-trip,
  stronger coupling widens the resonance gap). Closes the Sources §2 Mason theory gap; the bulk
  PZT therapy workhorse behind Chapter 33 §33.9.
- ✅ **CMUT/PMUT therapeutic-regime extension (output pressure + flexible limitation)** —
  `[minor]` (Chapter 33 §33.9). `CmutCell::{max_surface_velocity, max_output_pressure (gap-limited
  ceiling), flex_gap_derating}`, `PmutCell::{deflection_per_volt, max_output_pressure (∝ drive)}`,
  `plate::{flexible_output_factor, curvature_sag}`, and `comparison::{evaluate_therapy,
  TherapyVerdict}`. 4 value-semantic tests (CMUT output ∝ gap & drive-independent; PMUT output ∝
  drive, PZT>AlN; flexing reduces CMUT output and tighter gaps lose more; therapy verdict = PMUT).
  Proves the user's point: a CMUT's electrostatic gap-limited output saturates and *flexing it
  cuts output further* (sub-micron gap perturbed by curvature) — so for 2–5 MHz high-pressure
  therapy, PMUT/bulk-PZT win (opposite of the IVUS imaging verdict). PyO3 bindings + ch33 fig06
  added.
- ✅ **CMUT & PMUT micromachined-transducer models + Chapter 33 (IVUS)** — `[major]`
  (ADR 015; supersedes the bulk-piezo "Mason + CMUT" backlog item — folded into a CMUT-vs-PMUT
  comparison for flexible/IVUS). `kwavers_transducer::mems::{plate, cmut::CmutCell,
  pmut::{PmutCell, PiezoFilm}, comparison::evaluate_ivus}` — clamped-plate resonance + Lamb
  immersion, CMUT collapse voltage / bias coupling / capacitance / self-heating / fluid-loading
  bandwidth, PMUT composite resonance / film coupling (AlN vs PZT) / self-heating / transmit
  sensitivity, and the weighted IVUS figure of merit. 13 value-semantic tests (f∝h/a²,
  immersion downshift, V_c∝g^1.5, coupling bounds, PZT>AlN coupling & heating, CMUT-wins-IVUS
  verdict, drive-weight flip to PMUT). PyO3 bindings added (`mems_*`, `cmut_*`, `pmut_*`,
  `ivus_figure_of_merit`); new Chapter 33 (`cmut_vs_pmut.md`) + figure script
  `ch33_cmut_vs_pmut.py`. **ch33 figures DONE (2026-06-09):** rebuilt pykwavers and generated all 6
  `figures/ch33/*` (resonance_geometry, electrical, heating, bandwidth, ivus_fom, therapy_output).
  **Fixed an exposure-gap bug found doing so:** the 15 mems `#[pyfunction]`s were registered in
  `register_book` (so present in `_pykwavers`) but **omitted from the hand-maintained
  `python/pykwavers/__init__.py` re-export list** — so `kw.cmut_coupling_k2` etc. raised
  AttributeError at the public API even though `kw._pykwavers.*` worked. Added all 15 to the
  `from ._pykwavers import (...)` block and `__all__`; verified the figures regenerate.
- ✅ **Acousto-elasticity — Murnaghan stress-dependent wave speed + pre-stress inversion** —
  `[major]` (ADR 014; scope = analytical relation/inversion, full 3rd-order PDE deferred).
  `kwavers_physics::analytical::elastography::{acoustoelastic_sensitivity,
  acoustoelastic_shear_speed, estimate_prestress, estimate_prestress_sequence}` —
  `ρc_S²=μ+Aσ₀`, `A=(m+n)/(2(λ+μ))`, `σ₀=ρ(c_S²−c_S0²)/A`. 4 value-semantic tests (σ₀=0 →
  √(μ/ρ); A formula; pre-stress round-trip exact; cardiac-sequence per-frame recovery).
  Elastography §11.9 updated.
- ✅ **CEUS contrast pulse sequences** — `[minor]`.
  `kwavers_physics::acoustics::imaging::modalities::ceus::pulse_sequences::{pulse_inversion,
  amplitude_modulation, cps_combine}` — multi-pulse linear-cancellation combiners (Simpson 1999
  PI, Phillips 2001 CPS). 3 value-semantic tests with a quadratic scatterer model (PI cancels
  the fundamental and keeps 2f; AM cancels the linear response, nonlinear residual survives;
  CPS reproduces PI). Diagnostics §9.4 updated.
- ✅ **Acoustic CT — Radon transform + filtered backprojection** — `[major]` (ADR 013).
  `kwavers_diagnostics::reconstruction::radon::{radon_transform, filtered_backprojection}` —
  parallel-beam forward projection (bilinear ray sampling) + Ram-Lak ramp-filtered
  backprojection. 3 value-semantic tests (round-trip recovers a centred disk, Pearson>0.8 +
  centroid at centre; off-centre disk localizes to the correct quadrant within 4 px; empty→0).
  Inverse §6 updated; bent-ray SIRT/ART + reflection-CT remain. ADR docs/adr/013.
- ✅ **f-k (Stolt) migration** — `[minor]`.
  `kwavers_diagnostics::workflows::fk_migration::fk_stolt_migration` — exploding-reflector
  Stolt remap `ω = v·sign(k_z)√(k_x²+k_z²)` (v=c/2) with linear ω-interpolation and obliquity
  Jacobian, via the 2-D FFT helpers. 2 value-semantic tests (flat reflector → correct migrated
  depth within 3 bins; point scatterer focuses to (x0,z0) within ±2 lateral / ±5 axial bins and
  concentrates energy more than the raw hyperbola). Diagnostics §9.2.2 / Beamforming §7.5.2
  updated (no longer "not yet implemented").
- ✅ **Kelvin–Voigt viscoelastic medium kernel** — `[minor]`.
  `kwavers_medium::viscoelastic::KelvinVoigtModel` — frequency-domain complex shear modulus
  `G*(ω)=μ+iωη`, storage/loss moduli, loss tangent, Q, dispersive phase velocity
  `c_p(ω)=ω/Re(k)` and attenuation via `k=ω√(ρ/G*)`. 5 value-semantic tests (storage+i·loss,
  tan δ·Q=1, low-ω → elastic limit √(μ/ρ), dispersion + attenuation rise with ω, lossless
  η=0 limit). Elastography §11.8/§11.13 updated; Zener (SLS) variant remains.
- ✅ **L-BFGS quasi-Newton optimiser** — `[minor]`.
  `kwavers_math::optimization::{minimize, LbfgsConfig, LbfgsResult}` — Nocedal two-loop
  recursion + Armijo backtracking, limited-memory (`m` pairs), curvature-condition guarded
  updates. 3 value-semantic tests (SPD quadratic → A⁻¹b in ≤15 iters, separable quartic
  minimiser, zero-gradient immediate return). Inverse §9.1 updated. **Now wired into
  `FwiProcessor`** (future-enhancement #3): the two-loop recursion is factored into the
  reusable `kwavers_math::optimization::LbfgsMemory` (SSOT, shared by `minimize` and the FWI
  driver).
- ✅ **ConstrainedInversion (projected-gradient box constraints)** — `[minor]`.
  `kwavers_math::inverse_problems::{BoxConstraints, projected_gradient_descent}` — pointwise
  box projection (Π) with `sound_speed_tissue()`/`density_tissue()` presets + PGD over any
  gradient closure. 4 value-semantic tests (bound ordering, clamp out-of-range / keep
  in-range, PGD converges to clip(t,box) on a separable quadratic, zero-gradient fixpoint).
  Inverse §8.4 updated (no longer "design target").
- ✅ **2-D phase unwrapping** — `[minor]`. `kwavers_signal::phase::{unwrap_1d, unwrap_2d}`
  (separable Itoh path-following; exact for residue-free fields). 4 value-semantic tests
  (1-D ramp exact recovery across a genuine wrap, 2-D plane exact, identity on smooth,
  empty-input). Elastography §11.13 updated — residue-aware Goldstein branch-cut variant is
  the remaining upgrade.
- ✅ **Local Frequency Estimation (LFE)** for elastography — `[minor]`.
  `kwavers_solver::inverse::elastography::linear_methods::lfe::local_frequency_estimation_inversion`
  + `InversionMethod::LocalFrequencyEstimation`. Windowed energy-ratio
  `|k|² ≈ ⟨|∇u|²⟩/⟨u²⟩` (Oliphant/Manduca 2001). Test
  `test_local_frequency_estimation_recovers_known_speed` recovers a known cs=1.0 m/s plane
  wave within ±0.4 and verifies μ=ρcs². Chapters updated (Elastography §10.7.3/§10.13).

## Prioritized queue

**All originally-documented-but-missing components are implemented.** The queue is clear.

## Future enhancements (deepen what now exists)

These are *extensions* of shipped capabilities, not gaps in documented features. Each `[major]`
needs an ADR first.

1. ✅ **Chapter 33 figures** `[patch]` — DONE (2026-06-09). Rebuilt pykwavers and ran
   `ch33_cmut_vs_pmut.py` → all 6 `figures/ch33/*` (resonance_geometry, electrical, heating,
   bandwidth, ivus_fom, therapy_output). En route, fixed a real **mems-binding exposure bug**: the
   15 `#[pyfunction]`s were registered in `register_book` but omitted from the hand-maintained
   `python/pykwavers/__init__.py` re-export list, so `kw.cmut_*`/`kw.ivus_figure_of_merit` raised
   AttributeError; added all 15 to the import block + `__all__`.
2. ✅ **Zener (standard-linear-solid) viscoelastic kernel** `[minor]` — DONE.
   `kwavers_medium::viscoelastic::ZenerModel` — complex modulus `G_r+(G_u−G_r)iωτ/(1+iωτ)`,
   storage/loss, Debye loss peak at ωτ=1, bounded dispersion between relaxed/unrelaxed speeds.
   4 value-semantic tests (reject unphysical, storage G_r→G_u, loss peak=(G_u−G_r)/2 at ωτ=1,
   bounded dispersion). Elastography §11.8 updated.
3. ✅ **Wire L-BFGS into `FwiProcessor`** `[minor]` — DONE.
   `FwiProcessor::invert_lbfgs(observed, initial, geometry, grid, memory)` — quasi-Newton FWI
   using the shared `kwavers_math::optimization::LbfgsMemory` two-loop recursion for the search
   direction `d=−H·g`, reusing the existing forward/adjoint gradient (`misfit_and_gradient`,
   factored out of `descent_update`), un-normalized gradient (curvature pairs keep physical
   scaling), and Armijo projected line search. 2 value-semantic tests (single-shot recovers a
   localized +60 m/s anomaly: misfit < ½ initial, anomaly-cell + illuminated-region error fall;
   stationary at the zero-misfit truth, max |Δc| < 1e-6). `LbfgsMemory` is the SSOT used by both
   `minimize` and the FWI driver. Inverse §9.1.
4. ✅ **Bent-ray traveltime tomography** `[major]` (ADR 020) — DONE (2026-06-09). Found SIRT/ART/OSEM
   already implemented (`kwavers_solver::…::unified_sirt::{SirtAlgorithm, SirtReconstructor}`,
   Kaczmarz) + the `real_time_sirt` streaming pipeline; the genuine gap was the **bent-ray forward
   operator** (all projections were straight-line). Implemented
   `kwavers_diagnostics::reconstruction::bent_ray::{bent_ray_path, bent_ray_traveltime, BentRay}` —
   a Dijkstra shortest-path (Fermat) tracer over an 8-connected slowness grid with trapezoidal edge
   cost `½(s_u+s_v)·L`; returns traveltime + voxel path + the per-voxel path-length **system-matrix
   row** that plugs straight into the existing SIRT/ART reconstructor (`t = Σ s_v·row_v`). 6
   value-semantic tests (homogeneous axis/diagonal exact, row reproduces traveltime exactly,
   graph-metric bound ≥ Euclidean & ≤ 1.10×, **Fermat fast-channel lowers traveltime + the ray
   bends into the channel**, degenerate/OOB). Clippy-clean. **End-to-end inversion DONE (2026-06-09):**
   `reconstruction::bent_ray_tomography::{reconstruct_bent_ray_tomography, rms_misfit,
   TraveltimeDatum, BentRayTomographyConfig}` — the nonlinear trace↔solve fixed point re-traces rays
   through the evolving model and refines it by sparse-ART (Kaczmarz) sweeps over the path-length
   rows (the `Array2` row-major buffer is indexed directly by the `BentRay::row` flatten `i·ny+j`; no
   dense matrix formed). 2 value-semantic tests realizing the ADR-020 "tomographic recovery"
   verification: a 7%-wrong uniform guess → true uniform (mean interior error ~1%, ≥85% voxels <3%,
   misfit collapses >10×); a slow disk recovered (correlation >0.5, anomaly slower than background,
   misfit falls across outer iterations). **Remaining:** reflection-CT geometry (distinct
   acquisition) is a separate follow-on.
5. ✅ **Residue-aware phase unwrapping** — DONE (detection + masked unwrap; auto branch-cut mask placement remains).
6. ✅ **MEMS depth (CMUT/PMUT)** `[major]` — (output pressure + flexible-output limitation +
   **squeeze-film damping** DONE §33.6/§33.9; **inter-element acoustic crosstalk** DONE 2026-06-09).
   `mems::crosstalk` (additive `[minor]`): baffled-monopole **mutual radiation impedance**
   `Z_ij=jωρ A_iA_j/(2π d)·e^{-jkd}` + array `crosstalk_matrix` (reciprocal, zero-diagonal). 5
   value-semantic tests: closed-form magnitude + retardation phase `π/2−kd`, reciprocity,
   `∝ω`/`∝1/d`/far-field-decay scaling, matrix symmetry + nn/nnn `1/d` ratio + closed-form
   cross-check, degenerate inputs. Honest scope: **fluid path only** (`d≫a`, `ka≲1`); substrate
   Lamb/Stoneley path + coupled-field FEM out of scope (need a meshed model). Chapter §33.8 updated.
   **Collapse-mode nonlinear electrostatics** DONE (2026-06-09): `CmutCell::{bias_pulldown_fraction,
   biased_gap, biased_capacitance, bias_softened_resonance}` — the exact stable equilibrium
   `u(1−u)²=(4/27)(V/V_c)²` of `k x = ε₀A V²/(2(g₀−x)²)`, the bias-dependent operating gap/capacitance,
   and the **spring-softened resonance** `f(V)=f_imm√(1−2u/(1−u))` that vanishes at pull-in `V=V_c`.
   2 value-semantic tests: force-balance differential check at 3 biases + monotone pull-down +
   pull-in limit + collapse→None; capacitance rise + monotone resonance softening toward collapse.
   **Conformal flexible-array beamformer (populated by mems cells)** DONE (2026-06-09):
   `flexible::beamforming` — `focusing_delays` (conformal DAS `τ_i=(d_max−d_i)/c`, refocus after the
   array bends), `steering_delays` (far-field plane wave), `per_element_curvature` (Menger curvature
   from the tracked positions), and `cmut_flex_apodization` (per-element transmit weight from
   `CmutCell::flex_gap_derating` at the local curvature — the array "populated" by the CMUT model).
   Wired into `FlexibleTransducerArray::{focusing_delays, steering_delays, cmut_flex_apodization}`
   over its current deformed geometry. 5 value-semantic tests: in-phase arrival at the focus on a
   *deformed* array, flat-array symmetric peaked-centre delays, broadside/oblique steering ramp,
   curvature = 0 (flat) / 1/R (circle), and flex apodization derating curved + tighter-gap elements.
   All 3 MEMS-depth sub-items now done → **#6 closed**. **Remaining (separate future work):**
   post-collapse (membrane-contact) annular CMUT operation needs an insulator/contact-gap parameter
   the lumped model doesn't carry; substrate Lamb/Stoneley crosstalk + coupled-field FEM.
7. 🟡 **3rd-order (Murnaghan) elastic-wave** `[major]` (ADR 022) — **constitutive core DONE
   (2026-06-09); PDE solve staged.** Verified the gap is real, not a false-negative: the existing
   `NonlinearElasticWaveSolver` uses *hyperelastic* (Neo-Hookean/Mooney-Rivlin) invariant
   nonlinearity and its own header lists "Third-order elastic constants M and N" + "Acoustoelastic
   tensor" as **not implemented**. Implemented the missing constitutive layer in
   `kwavers_physics::analytical::murnaghan`: `MurnaghanConstants{λ,μ,l,m,n}`, `strain_energy(E)`,
   `second_pk_stress(E)` (`S=[λtrE+l(trE)²+m trE²]I+(2μ+2m trE)E+3nE²`), `apply_reference_tangent`,
   and the **finite-strain material tangent** `material_tangent` (`ℂ(E)=∂²W/∂E²=∂S/∂E`).
   **Convention pinned to the codebase SSOT:** discovered the chapter §11.9.1 + existing
   `acoustoelastic_sensitivity(m,n)` use the *power-sum* invariant form (`trE², trE³`), which gives
   different `(l,m,n)` than the principal-invariant (Hughes-Kelly) form — implemented the power-sum
   form so the constants are shared across the constitutive model and the AE relation. 10
   value-semantic tests: StVK reduction, linear/Hooke limit, uniaxial closed form, **energy–stress
   consistency `S=∂W/∂E` and tangent consistency `ℂ(E):H=∂S/∂E` by finite difference**, tangent
   major-symmetry, symmetry, reference tangent. Clippy-clean; also
   cleared 2 pre-existing unused-import warnings in `elastic_wave/tests.rs`. Chapter §11.9 updated.
   **Staged follow-ons (own [major] items, ADR 022):** (a) small-on-large acousto-elastic acoustic
   tensor `A⁰=ℂ(E₀)+initial-stress geometric terms` + Christoffel eigenproblem linking to the
   first-order `A` + `O(σ₀²)` terms — needs the Thurston-Brugger geometric terms and the exact
   Hughes-Kelly config to reproduce `A=(m+n)/(2(λ+μ))` (the finite tangent alone is insufficient);
   (b) time-domain 3rd-order forward PDE consuming the Murnaghan `S`.
15. ✅ **Exact discrete-adjoint FWI gradient** `[major]` (ADR 016) — DONE via a dedicated
    **self-adjoint second-order acoustic engine** (`FwiEngine::SecondOrderSelfAdjoint`,
    `inverse::fwi::time_domain::self_adjoint`). Background: the FD gradient test
    (`tests::gradient::test_fwi_adjoint_gradient_is_valid_descent_direction`) showed the
    FDTD/PSTD-driven path is only an **approximate** adjoint — `κ=(g·δm)/(dJ/ds) ≈ 238`/`191`
    across directions (stable under step refinement): a ~200× scale offset (adjoint re-injects
    through the scaled additive-source path `2·dt·c₀/(N·dx)` vs direct-pressure receiver
    sampling — not transposes) plus ~20% shape error (PML/leapfrog non-self-adjointness).
    Correct descent direction (Armijo absorbs it) but wrong absolute magnitude — fatal for
    Gauss-Newton, fixed-step updates, gradient-norm stopping. Path A (literal transpose of the
    shared CPML staggered solver) was rejected as high-risk to the parity-validated forward
    code; Path B (textbook self-adjoint engine) was chosen. The new engine uses a symmetric
    heterogeneous Dirichlet Laplacian + 3-point leapfrog + matched source/receiver injection;
    its discrete adjoint is the same scheme run backward, so the exact gradient
    `g_x=(−2/ρc³)Σ ξ^n p̈^n` gives **κ≈1 to <1e-4** for 3 independent directions
    (`self_adjoint::tests::self_adjoint_gradient_matches_finite_difference_kappa_one`), and full
    FWI/L-BFGS converges through it
    (`tests::lbfgs::self_adjoint_engine_lbfgs_reduces_misfit_and_recovers_anomaly`). Default
    `FwiEngine::Solver` (FDTD/PSTD) retained, still documented as approximate.
    **Self-adjoint absorbing layer — DONE:** the SA engine now supports an optional symmetric
    diagonal sponge (damped leapfrog `W p̈ + B ṗ = D p + s`, `build_edge_sponge`,
    `FwiProcessor::with_self_adjoint_damping`); re-derived exact adjoint preserves κ≈1 to <1e-4
    WITH the sponge (`self_adjoint_gradient_kappa_one_with_sponge`) and it absorbs >70% of
    outgoing energy vs reflecting walls (`self_adjoint_sponge_absorbs_outgoing_waves`). Remaining
    deferred: Path A (literal transpose of the shared CPML solver) only if FWI must invert with
    the exact CPML operator.
16. ✅ **MOFI — guidance-free rigid skull-template alignment** `[minor]` (ADR 017) — DONE.
    Implements Bates et al. (2026, *Ultrasound Med. Biol.*, "Automatic Skull-Template Alignment
    Without a Guidance Image"): align a CT-derived sound-speed template to acoustic data alone (no
    MRI) by minimising the FWI misfit over a rigid SE(2) reparametrisation `φ={θ,δ₁,δ₂}` of the
    template instead of the full pixel grid. `inverse::fwi::time_domain::mofi`: analytic bilinear
    reparametrisation Jacobian `∂c_φ/∂φ` (FD-verified), chained gradient
    `∂f/∂φ=(∂c_φ/∂φ)ᵀ ∂f/∂c` using the **exact** self-adjoint `∂f/∂c` (ADR 016), and SE(2)
    manifold optimisation (Appendix A: SO(2) log/exp rotation update, `δ←δ+R_θΔδ`, gradient
    normalisation + Armijo line search; θ/δ balanced in scaled space `(L·θ,δ)`). Recovers a known
    `(θ=6°, δ=(2,−1.5)mm)` misalignment of an asymmetric 2-D phantom from ring-array data to
    **<1° / <1 mm** with misfit collapsing >10× (`mofi::tests::mofi_recovers_known_rigid_misalignment`),
    + Jacobian-vs-FD and stationary-at-truth tests. API `mofi_align`/`MofiConfig`/`MofiResult`/
    `RigidTransform`. Scope: 2-D SE(2), single acquisition; 3-D SE(3) and non-rigid extensions deferred.
17. ✅ **Multi-pathway skull-registration pipeline (beyond rigid MOFI)** `[minor]` (ADR 018) — DONE.
    Four composable pathways + a pipeline, all on the exact self-adjoint `∂f/∂c` (ADR 016), in
    `inverse::fwi::time_domain::mofi`: (a) **misfit homotopy** (`align_homotopy`, Wasserstein→
    envelope→L2 warm-started) widens the capture basin — recovers 28° where plain L2 fails;
    (b) **coarse global pose initializer** (`coarse_pose_search`, robust Wasserstein search — NOT
    envelope, which is phase-blind/rotation-insensitive) rescues 45° misalignment; (c) **joint
    pose + sound-speed calibration** (`align_with_calibration`, block-coordinate pose↔α,
    `c=c_bg+α(c_tmpl−c_bg)`) recovers α=1.25 + pose; (d) **non-rigid FFD** (`align_nonrigid`,
    `nonrigid.rs`, bilinear control lattice + bending-energy reg, chained gradient
    `∂f/∂u_cp=−Σ g·∇c·w_cp`) recovers a smooth warp. `align_pipeline` chains coarse→rigid+cal→
    non-rigid; compound (pose+speed+warp) test aligns the model to <40 m/s RMS in the illuminated
    region. 8 MOFI value-semantic tests. API: `mofi_align_homotopy`/`_coarse_pose_search`/
    `_align_with_calibration`/`_align_nonrigid`/`_align_pipeline`. Scope: 2-D, single acquisition;
    3-D/SE(3), cubic-B-spline FFD, and TT-tomography image init deferred.
18. ✅ **MOFI/SA robustness hardening** `[patch]`/`[minor]` — DONE. Resolved the flagged fragilities:
    (a) `invert_lbfgs` convergence/zero-gradient guard is now **relative** to the initial gradient
    norm (was absolute `f64::EPSILON`), so the SA engine's small-amplitude gradients (‖g‖∞~1e-18)
    converge without rescaling; (b) FFD **smoothness weight is now relative** to J₀ (auto-scaled
    `w·J₀/dx²`), removing the absolute-scale fragility that could freeze the optimiser; both removed
    the `source×1e6` test workarounds. (c) `recommended_search_misfit()` + docs make the
    phase-blind-envelope coarse-search pitfall API-visible (use Wasserstein). All MOFI/L-BFGS tests
    pass unscaled.
18b. ✅ **Self-adjoint FWI gradient — `O(N)` reverse-reconstruction memory path** `[minor]` — DONE.
    The exact self-adjoint engine (ADR 016) stored the full forward wavefield history
    `Array4(nt,nx,ny,nz)` (e.g. ~838 MB/shot at nt=400, 64³) to feed the imaging condition. The
    lossless energy-conserving leapfrog is exactly time-reversible (`c_prev=1`), so
    `self_adjoint::forward_tail` now keeps only the final two states `(p^{N-1},p^{N-2})` + traces and
    `self_adjoint::gradient_reconstructed` re-derives the forward field **backward** in lockstep with
    the adjoint sweep — peak per-shot memory drops `O(nt·N) → O(N)` (a handful of 3-D arrays) at the
    cost of one extra Helmholtz apply per backward step (the standard FWI memory↔recompute trade).
    Both FWI gradient drivers route through a new shared `forward_misfit_raw_gradient` helper (DRY)
    that selects the reconstruction path for the **lossless** SA engine; the **damped** SA engine
    (anti-amplifies under reverse stepping) and the FDTD/PSTD `Solver` engine keep the stored history.
    Evidence: `self_adjoint::tests::reconstructed_gradient_matches_stored_history` asserts the
    reconstructed gradient equals the stored-history gradient to <1e-9 relative (and `forward_tail`
    seed states equal `history[N-1]/[N-2]` exactly); the κ≈1 and SA L-BFGS recovery tests now run
    through the reconstruction path. 96 FWI tests pass. (Follow-up `[patch]`: `self_adjoint/mod.rs`
    grew to ~760 lines — candidate SoC split into `forward`/`gradient`/`coeffs` leaf modules.)
19. ⬜ **MOFI 3-D SE(3) alignment** `[major]` (ADR required) — generalise the 2-D SE(2)
    reparametrisation to full 3-D rigid motion: 3-angle (or rotation-matrix) rotation + 3-D
    translation, trilinear interpolation, and the SE(3)/SO(3) log-exp manifold update with the
    analytic 3-D rotation Jacobian. Framework (chained gradient on the exact SA engine, homotopy,
    calibration, pipeline) carries over unchanged; only `transform.rs` and `manifold_update` change.
    Acceptance: recover a known 3-D pose to <1°/<1 mm on a 3-D phantom. Mechanical but sizable.
20. ✅ **Cubic B-spline FFD basis** `[minor]` — ALREADY IMPLEMENTED (stale-open; verified 2026-06-09).
    `kwavers_solver::…::mofi::nonrigid` has `FfdBasis::{Bilinear, CubicBSpline}` with the uniform
    cubic B-spline `axis_weights` (4-point support, partition of unity, C²). Test
    `nonrigid_ffd_cubic_bspline_recovers_smooth_deformation` passes (recovers a smooth warp). The
    backlog item predated the implementation.
21. ⬜ **Travel-time-tomography initializer for MOFI** `[major]` — stand up the existing
    `sound_speed_shift`/`real_time_sirt` travel-time subsystem as a template-free coarse sound-speed
    map to (a) image-to-image seed the pose and (b) calibrate template speeds. Lower marginal value:
    `coarse_pose_search` already provides a global, cycle-skip-free pose seed; this adds a second,
    physics-distinct initializer + speed prior. Acceptance: TT map seeds the pipeline and improves
    convergence on a contrast where coarse-search alone struggles.
22. ✅ **Cubic B-spline FFD basis** `[minor]` — DONE. `FfdBasis::{Bilinear, CubicBSpline}` on the
    FFD lattice (uniform cubic B-spline, 4×4 support, C²); `nonrigid_ffd_cubic_bspline_recovers_smooth_deformation`.
23. 🟡 **Marchenko + Wasserstein "prior-less" FWI** (ADR 019) — PARTIAL/staged.
    `inverse::marchenko`: verified windowed conv/corr operators; experimental 1-D iterative
    `redatum` (focusing functions + G⁻, structure per Wapenaar 2014 — **quantitative focusing
    convention not yet reference-validated**, documented); `marchenko_wasserstein_misfit`
    `J=W₁(G⁻_obs,G⁻_mod)` connector composing redatum with the (already-verified) Wasserstein
    misfit, tested well-posed. Staged milestones (own [major] items): (a) reference-validate 1-D
    `redatum` — **SA-engine oracle BUILT** (`marchenko::oracle_tests`, `#[ignore]`d acceptance
    target); empirically `corr(Marchenko,true)≈corr(naive,true)≈0.14` (coda≈0 ⇒ no engagement);
    root-caused blockers documented in ADR 019 (window/record geometry so internal multiples are
    in-window & on-record; conv/corr convention; G⁻ time-referencing; T_d amplitude); acceptance
    `corr>0.85 & >naive`; (b) multidimensional Marchenko (t-x, up/down decomposition);
    (c) Marchenko-Wasserstein FWI model-update loop on the SA-engine gradient.
    The Wasserstein "taming the math" half is already production-ready/verified
    (`wasserstein_is_convex_in_shift_on_positive_distribution`).
8. ✅ **Bulk-piezo Mason thickness-mode circuit** — DONE (see `BulkPiezoResonator`, above).
16. ⬜ **Rust-native segmentation-driven crossfire aperture optimiser** `[major]` — surfaced by the
    Ch31 audit. An earlier Python-side ray-trace optimiser (per-aperture air/bone/fat hazard-path
    scoring, an angular crossfire plan, complex ridge least-squares phase/amplitude weights, and
    dense-field hotspot null-refinement) was deliberately removed for PyO3-only compliance and
    replaced by the same-aperture theranostic inverse. A Rust-native re-implementation (in
    `kwavers_therapy::therapy::theranostic_guidance`, exposed via PyO3) would restore the
    segmentation-aware *placement-optimisation* capability — distinct from the current fixed-bowl
    placement + inversion — with value-semantic tests on path penalties and crossfire entrance-dose
    reduction. Needs an ADR (new optimisation module).
14. ✅ **Wire elastic / `MechanicalStress` into the `PhysicsCatalog`** `[major]` (ADR 021) — DONE
    (2026-06-09). **Stale-premise corrected:** the item was filed `[minor]` ("add a variant + one
    match arm against an existing solver"), but the prior `MechanicalStress` variant + its
    `ElasticWavePlugin` were *deliberately deleted* during the elastic-as-PSTD-plugin consolidation
    (a `μ ≡ 0` acoustic duplicate; see `forward/mod.rs`). The genuine elastic stepper is
    `ElasticPstdOrchestrator` (batch `propagate`, no `Plugin` adapter), so wiring needed a real
    adapter, not a match arm — reclassified `[major]`, ADR 021. **Delivered:** (a) extracted a
    public single-step SSOT primitive `ElasticPstdOrchestrator::step()` (the `propagate` loop now
    delegates) + `pressure_field()`/`stress_mut()`/`velocity_mut()` accessors; (b) new
    `pstd::extensions::MechanicalStressPlugin` — owns a real orchestrator, one genuine leapfrog
    λ/μ step per `Plugin::update`, provides isotropic pressure `p = -⅓ tr(σ)` to the unified field,
    requires nothing (self-contained elastic state); (c) additive serde-stable
    `PhysicsModelType::MechanicalStress { wave_kind: ElasticWaveKind::Isotropic }` + a real
    `build_plugin` arm (Theorem 21.1 exhaustiveness preserved). Value-semantic tests (9):
    `step`-loop == `propagate` bit-for-bit; `pressure_field` == −⅓ tr(σ) exactly; shear velocity
    gradient induces σxy (μ>0, not an acoustic alias); plugin steps + writes genuine evolving
    pressure (== orchestrator's, changes between steps); errors-before-init; field contract;
    catalog builds 1 plugin; composes with BubbleDynamics (2 plugins, scheduler resolves). Existing
    μ=0 reduction theorem test (`pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero`) still
    green. Clippy-clean (also bundled a pre-existing FWI 8-arg lint into `ReconstructionSeed`).
    Chapter 21 §21.3 updated (six variants, wired). **Remaining (future, own items):** anisotropic
    / nonlinear `ElasticWaveKind` modes; routing acoustic `context.sources` into an elastic source.
13. ⬜ **`Scalar` trait genericization of the solvers** `[arch]` — surfaced by the Ch20 audit and
    mandated by the project architecture standards, but **not currently implemented**: the CPU PSTD
    solver is monomorphic `f64` and the GPU path is `f32`, with no `Scalar` trait abstracting
    precision (separate code paths, not one zero-cost generic kernel). A genuine `Scalar` trait
    (associated `Accumulator`, native-precision arithmetic, sealed) genericizing `AbsorptionKernel`,
    the stepper field buffers, and the CPML updater would unify the precision tiers. Needs an ADR
    (large cross-cutting change; touches every kernel). Chapter 20 §20.10.3 now states the real
    monomorphic status honestly.
12. 🟡 **CPML cache tiling + 2 perf figures** `[minor]` — **tiling sub-item closed as
    not-applicable (2026-06-09)**: the premise was wrong. The CPML memory update
    (`cpml::update::axis`) is a **pointwise** recurrence `ψ ← b·ψ + a·∂f` (each cell reads only its
    own previous ψ and gradient — no spatial stencil, no neighbour reuse), already `rayon`-parallel
    over thin contiguous PML strips. Cache tiling improves only sweeps with spatial reuse; a
    pointwise streaming map is already bandwidth-optimal, so a `TILE_SIZE` const generic would add
    complexity for zero analytic benefit (CLAUDE.md: no unjustified optimizations). The tiling
    principle applies to the PSTD spatial-stencil/transpose path instead — already handled by
    `rustfft`'s adaptive planner. Chapter 20 (§ "Mitigation: tiling") corrected. **Remaining:** the
    2 perf artifacts (PSTD flamegraph — needs Linux `perf`/DTrace, not generable on this Windows
    host; KWCP-layout diagram — a static schematic) are doc figures, low value.
11. ✅ **2 sonogenetics figures** `[patch]` — DONE (2026-06-09). Added `fig06_pipeline_schematic`
    (acoustic field → ARF → tension → P_open → ion current → LIF spike) and `fig07_lif_raster_vs_duty`
    (LIF spike-raster across duty cycles, driven by the real Rust `kw.simulate_lif_neuron_py`) to
    `ch18_sonogenetics.py`; generated both into `figures/ch18/` and embedded them in Chapter 17 as
    Figs 17.7–17.8 (§17.1.1 and §17.9.1); §17.14 index updated. All 7 ch18 figures now resolve.
10. 🟡 **Transcranial pipeline ergonomics + 4 figures** `[minor]` — **turnkey + correctness gap
    closed (2026-06-09); figures remain.** The Ch15 audit said "there is no turnkey
    CT→medium→solve→correct→safety helper" — but `TreatmentPlanner::generate_plan` IS a turnkey
    planner (skull analysis → element placement → phasing → intensity → thermal → safety →
    treatment time). **It was, however, functionally broken:** (a) its CT aberration corrector was
    stored as a dead `_aberration_corrector` field and never applied — plans corrected only for
    geometric distance, defeating transcranial focusing's purpose; (b) `optimize_transducer_setup`
    mixed unit frames (millimetre element positions vs metre target centre, `radius=80` used as
    metres) so the geometric phase was dimensionally wrong. **Fixed:** reworked
    `optimize_transducer_setup` to place a focused bowl (radius = focal_distance) in the grid frame
    with consistent metre physics (positions stored in millimetres, the convention
    `simulate_acoustic_field` consumes), and **wired the live CT phase-screen aberration
    correction** so `φᵢ = φᵢ_geo + (−Δφᵢ)`; removed the dead corrector field. Value-semantic tests
    (2): homogeneous CT leaves the equidistant bowl in phase (span < 1e-6); a cortical-bone slab
    induces a ray-dependent phase spread (> 0.1 rad) that genuinely differs from the homogeneous
    plan — proving the corrector is applied, not dead. 18 treatment-planning tests green;
    clippy-clean. Chapter §15.11.5 rewritten (turnkey analytic route vs high-fidelity PSTD route).
    **Remaining:** the 4 Ch15 figures (TR workflow, skull hot-spot map, BBB safety window,
    propagation schematic) — doc artifacts, low value.
9. ✅ **Image registration (deformable/rigid)** — ALREADY IMPLEMENTED (Ch13 audit recorded a
   false negative; the Ch19 audit found the real code). `kwavers_physics::acoustics::imaging::
   fusion::registration::RitkRegistrationEngine` (backed by the `ritk-registration` workspace crate)
   provides `RegistrationMethod::{RigidBody, Affine, NonRigid}` — rigid/affine mutual-information
   registration and symmetric-Demons (Vercauteren 2009) non-rigid registration; driven via
   `register_for_method` / `rigid_registration_mutual_info`. Used by `multimodality_fusion::manager`
   and the physics fusion algorithms. Chapters 13 §13.7 and 19 §19.10.2 corrected to cite it.
   (Lesson: verify by algorithm, not by the guessed type name `DeformableRegistration`/`RITK`.)

## Notes

- `[major]` items each get an ADR before implementation (see `docs/adr/013–015`).
- Each landed item must: update the corresponding chapter (remove any theory-only marker),
  add value-semantic tests, and keep the local pre-merge gate green.
