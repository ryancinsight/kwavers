<a id="kw-phys-001"></a>

## KW-PHYS-001 — Rectified-diffusion rate is not the Eller–Flynn rate (missing oscillation-amplitude factor) [patch] — done (Nov 19, 2026)

priority: correctness; needs: none; scope: `rectified_diffusion_rate` in `kwavers-physics` (`acoustics/mechanics/cavitation/core/growth.rs`); milestone: none

- **Outcome.** The rate includes the second-order oscillation amplitude, thermal response, and Laplace dissolution. The saturated steady comparison verifies the model's implementation.
- **Benchmark.** Compare measured equilibrium-radius growth with the signed net radius rate at the corresponding initial radius. The measurements vary with radius, exposure history, and experimental conditions.

### Resolution (Nov 19, 2026)

- **Root cause.** The old formula `4πR·D·C·(P_ac/P_0)·√Pe` was not the Eller–Flynn rate at all: it was linear in the pressure ratio with no oscillation-amplitude dependence, carried a spurious `√Pe` factor (the additive transient term is proportional to `R_0/√(πDt)` and decays to zero at long times), and had no Laplace dissolution term. It was ~1000× the genuine rectified flux (~790× the growth term, ~1040× the net flux at 30 µm).
- **Fix.** Replaced with the genuine generalised Eller–Flynn steady (long-time) rate of Crum & Hansen (1982, JASA 72, 1586–1592) as restated by Crum & Mao (1996, p. 2902) and Zhang (2012, PhD thesis, §3.2, eqs. (3.6)–(3.19)): net molar flux `4πR_0·D·C_0·⟨R/R_0⟩·(1 − ⟨(R/R_0)⁴(P_in/P_0)⟩/⟨(R/R_0)⁴⟩)`, second order in the fractional oscillation amplitude `ε = |α|(P_ac/P_0)`, split into a rectified growth term and a Laplace dissolution term, so the returned flux is *signed* (growth above threshold, dissolution below). Thermal function Φ of Devin (1959, eq. 50) / Prosperetti et al. (1988, eq. 41), with the effective polytropic exponent `κ = Re Φ/3` (Prosperetti 1977, JASA 61, 17) used in every equation; thermal damping per Crum & Mao eq. (3.10), `δ_th = Im Φ/Re Φ`. Signature unchanged; `rectified_diffusion_rate_checked` guard unchanged. New public API: `rectified_diffusion_terms(...)` returning a `RectifiedDiffusionTerms` struct (net/growth/dissolution flux, ε, κ, δ_tot), and `oscillation_amplitude_fraction(...)` so callers can check `ε ≪ 1`. The old docstring's Church (1988) citation (wrong title/pages) was corrected.
- **Two corrections found while implementing.** (a) The K-factor denominator's surface term is `2σ/(R_0P_0)` (thesis eq. (3.15)), not `σ/(R_0P_0)`. (b) The thesis uses the thermal κ everywhere in the dynamics (ω₀², β², damping, K, cycle averages), not the adiabatic γ — an adiabatic cross-check understates ε and the growth term by ~30% (threshold 55 µm / 32 kPa with γ vs 47 µm / 27 kPa with κ). The Rust implementation and an independent Python implementation agree to 6+ significant figures on every intermediate (ε, K, κ, δ_tot, β², averages, brackets, fluxes) at four check points.
- **Measured numbers — Eller (1969) benchmark** (26.6 kHz, 20 kPa, D = 2e-9 m²/s, C = 0.26 mol/m³; dR/dt via isothermal ideal-gas `dn/dR`):

  | R₀ (µm) | net (µm/min) | growth term only (µm/min) | old formula |
  |--------:|-------------:|--------------------------:|------------:|
  | 20 | −1.65 | +0.94 | 516 |
  | 30 | −0.50 | +0.66 | 524 |
  | 40 | −0.13 | +0.53 | 528 |
  | 50 | +0.05 | +0.47 | 530 |

  Smith & Wang (2022), [figure 3](https://doi.org/10.1017/jfm.2022.218), replot measured rates of approximately +0.5, +5, +9, and +16 µm/min at initial radii of 30, 36, 42.5, and 45 µm. The saturated steady net rates are approximately −0.50, −0.24, −0.08, and −0.03 µm/min, respectively. The model crosses zero at approximately 46.9 µm for 20 kPa or at 26.85 kPa for 30 µm.
- **Acceptance.** The benchmark checks signed flux, the separate rectified term, and the small oscillation amplitude under nominal steady-model inputs. The experimental comparison also depends on finite-time gas transport and experimental conditions.
- **Tests.** Physics suite 1591 passed / 0 failed (baseline 1588/0; +3 new tests). Growth module 10/10: new `test_eller_1969_benchmark`, `test_eller_threshold_pressure`, `test_thermal_phi_limits` (Φ adiabatic/isothermal limits, δ_th vs Devin eqs. (2.50)/(2.52)); `test_rectified_diffusion_amplitude_scaling` replaces the old linearity test (which encoded the bug: linear in P_ac, √Pe) with P_ac² growth scaling + exact C-linearity; `test_rectified_diffusion_zero_acoustic_pressure` updated (net flux is now negative — Laplace dissolution — growth term exactly 0.0); `test_rectified_diffusion_zero_diffusivity` now asserts exactly 0.0 (no Pe division, so the old NaN allowance is gone); `test_rectified_diffusion_positive_flux` now uses an above-threshold point (30 µm / 35 kPa / 26.6 kHz, +0.75 µm/min) and documents that the old 5 µm / 120 kPa / 1 MHz point is below threshold (net −11.6 µm/min, growth term +25 µm/min). All three `rectified_diffusion_rate_checked` guard tests unchanged and passing. `cargo fmt --check` clean, no warnings.
- **Open physics.** Finite-time diffusion requires a concentration history or a resolved gas field. A near-wall or shape-mode streaming closure requires evidence for the local flow and interface conditions. The explicit-concentration API supports independently specified bulk concentration and flat-interface solubility for steady, spherical transport.
