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
  `ch33_cmut_vs_pmut.py`. NOTE: ch33 figure PNGs need a `maturin develop --release` rebuild of
  pykwavers (the script is ready; figures regenerate like every other chapter's).
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

1. ⬜ **Regenerate Chapter 33 figures** `[patch]` — `maturin develop --release` then run
   `ch33_cmut_vs_pmut.py` (the script + PyO3 bindings are ready; figures regenerate like every
   other chapter's).
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
4. ⬜ **Bent-ray SIRT/ART + reflection CT** `[major]` — upgrade the straight-ray Radon/FBP
   (ADR 013) for heterogeneous media; `real_time_sirt` is the starting point.
5. ✅ **Residue-aware phase unwrapping** — DONE (detection + masked unwrap; auto branch-cut mask placement remains).
6. ⬜ **MEMS depth (CMUT/PMUT)** `[major]` — (output pressure + flexible-output limitation +
   **squeeze-film damping** DONE, §33.6/§33.9) remaining: inter-element crosstalk, collapse-mode
   nonlinear drive, and a flexible-array beamformer populated by `mems` cells (Chapter 33 /
   `flexible`); enables a full coupled-field CMUT/PMUT forward model.
7. ⬜ **Full 3rd-order (Murnaghan) elastic-wave PDE solver** `[major]` — the time-domain
   nonlinear forward field behind the analytical acousto-elastic relation (ADR 014).
8. ✅ **Bulk-piezo Mason thickness-mode circuit** — DONE (see `BulkPiezoResonator`, above).
9. ⬜ **Image registration (deformable/rigid)** `[major]` — surfaced by the Ch13 audit: the
   theranostic loop (Algorithm 13.2 step 2) needs intra-/inter-modality registration to align
   diagnostic frames to the therapy frame, but kwavers has **no registration implementation**
   (only ITK DICOM loading). Chapter 13 now marks it not-implemented. A real implementation
   (e.g. rigid + B-spline FFD with a mutual-information / SSD metric and a gradient optimiser —
   the L-BFGS optimiser now in `kwavers_math` could drive it) would close the gap; needs an ADR
   (new bounded context: `registration`).

## Notes

- `[major]` items each get an ADR before implementation (see `docs/adr/013–015`).
- Each landed item must: update the corresponding chapter (remove any theory-only marker),
  add value-semantic tests, and keep the local pre-merge gate green.
