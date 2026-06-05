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
  minimiser, zero-gradient immediate return). Inverse §9.1 updated; wiring into `FwiProcessor`
  as the refinement phase is the remaining integration step.
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

## Prioritized queue (high value, self-contained first)

1. ⬜ **Radon transform + filtered backprojection (acoustic CT)** `[major]` — travel-time
   tomography (Inverse §6). Larger; SIRT/ART for bent rays is a follow-on. Test: reconstruct a
   Shepp–Logan-style slowness phantom within tolerance.
2. ⬜ **CEUS contrast pulse sequences** `[minor]` — pulse-inversion / amplitude-modulation
   tissue-harmonic suppression (Diagnostics §9.4). Test: 2nd-harmonic bubble signal survives
   PI summation while linear tissue cancels.
3. ⬜ **Murnaghan third-order / acousto-elastic inversion** `[major]` — stress-dependent
    wave-speed inversion (Elastography §11.9). Needs a 3rd-order elastic forward kernel.
4. ⬜ **Piezoelectric (Mason) + CMUT transducer models** `[major]` — electromechanical
    element model (Sources §2). Large; currently kwavers injects a prescribed kinematic source.

## Notes

- Items 1, 3, 4 are `[major]` (new forward kernels / subsystems) and should each get an ADR
  before implementation.
- Each landed item must: update the corresponding chapter (remove the theory-only marker),
  add value-semantic tests, and keep the local pre-merge gate green.
