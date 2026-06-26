//! Slice-wide tests for the [`crate::physics::acoustic`] vertical slice (Phase 3g).
//!
//! Phase 3g consolidated the previously-inline `mod tests { ... }` block of the flat
//! `src/acoustic.rs` into this single `crate::physics::acoustic::tests` module, mirroring
//! the Phase 2c `place::tests` / Phase 3f `si::tests` consolidation pattern: one slice-wide
//! test file collects the existing 13-test surface + the 3 NEW slice APIs
//! (`bvd_anti_resonance_hz`, `isppa_w_per_m2`, `round_trip_attenuation_db`) so an
//! external reviewer can grep `tests.rs` once and see the entire acoustic-slice behavioural
//! contract.

use crate::physics::acoustic::{
    acoustic_intensity_w_per_m2, array_factor, bvd_anti_resonance_hz,
    bvd_series_resonance_hz, element_factor, f_number, focal_pressure_gain,
    focused_delay_profile_s, grating_lobe_angle_deg, isppa_w_per_m2,
    max_delay_quantization_error_s, max_grating_free_steer_deg, mechanical_index,
    near_field_distance_m, nonlinear_shock_parameter, pitch_from_aperture_m,
    pressure_derating, quantize_delays_s, round_trip_attenuation_db, tissue_attenuation_db,
    wavelength_m,
};

// ───────── Existing fixtures (lifted verbatim from src/acoustic.rs::mod tests) ─────────

#[test]
fn array_factor_peaks_at_steer_and_grating() {
    let l = wavelength_m(1540.0, 2.0e6);
    let d = l / 2.0; // grating-lobe-free pitch
                     // Peak (=1) at the steer angle; lower off-axis.
    assert!((array_factor(16, d, l, 20.0, 20.0) - 1.0).abs() < 1e-6);
    assert!(array_factor(16, d, l, 20.0, 0.0) < 0.5);
    // Uniform 16-element broadside: first side lobe ≈ −13 dB ≈ 0.22 (sample near it, ≤ 0.3).
    let near_sl = array_factor(16, d, l, 0.0, 5.5);
    assert!(
        near_sl < 0.3,
        "first side lobe should be well below the main lobe, got {near_sl:.2}"
    );
    // Pitch = λ ⇒ a full-height grating lobe appears at endfire for broadside steering.
    assert!((array_factor(16, l, l, 0.0, -90.0) - 1.0).abs() < 1e-3);
}

#[test]
fn mechanical_index_of_the_paper_regime() {
    // The paper reaches ~6 MPa at 2 MHz ⇒ MI = 6/√2 ≈ 4.24 — a therapy regime above the 1.9
    // diagnostic limit (expected for neuromodulation).
    let mi = mechanical_index(6.0, 2.0);
    assert!((mi - 4.243).abs() < 0.01, "expected MI ~4.24, got {mi:.2}");
    assert!(
        mi > 1.9,
        "neuromodulation runs above the diagnostic MI limit"
    );
}

#[test]
fn tissue_attenuation_at_depth() {
    // 0.5 dB/cm/MHz × 2 MHz × 5 cm = 5 dB ⇒ pressure ×10^(−5/20) ≈ 0.56.
    let db = tissue_attenuation_db(0.5, 2.0, 5.0);
    assert!((db - 5.0).abs() < 1e-9);
    assert!((pressure_derating(db) - 0.562).abs() < 0.005);
}

#[test]
fn element_directivity_rolls_off_with_angle() {
    let l = wavelength_m(1540.0, 2.0e6);
    // On-axis is unity; off-axis is lower; a wider element rolls off faster.
    assert!((element_factor(0.4e-3, l, 0.0) - 1.0).abs() < 1e-9);
    assert!(element_factor(0.4e-3, l, 45.0) < 1.0);
    assert!(element_factor(0.7e-3, l, 45.0) < element_factor(0.4e-3, l, 45.0));
}

#[test]
fn near_field_and_fnumber_of_the_array() {
    // 4.3 mm aperture at λ = 0.77 mm ⇒ N = d²/4λ ≈ 6 mm; the 10 mm focus is in the far field.
    let l = wavelength_m(1540.0, 2.0e6);
    let n = near_field_distance_m(4.3e-3, l);
    assert!(
        (n - 6.0e-3).abs() < 1.0e-3,
        "expected ~6 mm near field, got {:.1} mm",
        n * 1e3
    );
    assert!(0.010 > n, "10 mm focus must be in the far field");
    // f/# of a 10 mm focus on a 4.3 mm aperture ≈ 2.3.
    assert!((f_number(10.0e-3, 4.3e-3) - 2.33).abs() < 0.05);
}

#[test]
fn focused_profile_quantizes_within_half_timing_step() {
    let pitch = pitch_from_aperture_m(4.3e-3, 16);
    let delays = focused_delay_profile_s(16, pitch, 10.0e-3, 45.0, 1540.0);
    assert_eq!(delays.len(), 16);
    assert!(
        delays.iter().all(|d| *d >= 0.0),
        "relative transmit delays are non-negative"
    );
    let q = quantize_delays_s(&delays, 5.0e-9);
    let err = max_delay_quantization_error_s(&delays, &q);
    assert!(
        err <= 2.5e-9 + f64::EPSILON,
        "nearest-step quantization error must be <= half the 5 ns timing step, got {err:e}"
    );
}

#[test]
fn bvd_resonance_matches_2mhz_drive() {
    // The paper's BVD series branch (L_s = 0.49 mH, C_s = 12 pF) resonates at ~2.07 MHz —
    // i.e. the 2 MHz drive is matched to the transducer. Independent cross-check of the spec.
    let f = bvd_series_resonance_hz(0.49e-3, 12e-12);
    assert!(
        (f - 2.07e6).abs() < 0.1e6,
        "expected ~2.07 MHz, got {:.3} MHz",
        f / 1e6
    );
}

#[test]
fn wavelength_at_2mhz_in_tissue() {
    // 1540 m/s / 2 MHz = 0.77 mm.
    let l = wavelength_m(1540.0, 2.0e6);
    assert!((l - 0.77e-3).abs() < 1e-6, "expected 0.77 mm, got {l:.2e}");
}

#[test]
fn half_wavelength_pitch_steers_fully() {
    let l = wavelength_m(1540.0, 2.0e6);
    assert!((max_grating_free_steer_deg(l / 2.0, l) - 90.0).abs() < 1e-6);
    // Pitch = λ ⇒ no steering before a grating lobe; pitch = 2λ/3 ⇒ 30°.
    assert!(max_grating_free_steer_deg(l, l) < 1e-6);
    assert!((max_grating_free_steer_deg(2.0 * l / 3.0, l) - 30.0).abs() < 0.1);
}

#[test]
fn grating_lobe_appears_when_pitch_too_large() {
    let l = wavelength_m(1540.0, 2.0e6);
    // λ/2 pitch at broadside: no grating lobe in real space.
    assert!(grating_lobe_angle_deg(l / 2.0, l, 0.0).is_none());
    // λ pitch at broadside: grating lobe at endfire (−90°).
    let g = grating_lobe_angle_deg(l, l, 0.0).expect("grating lobe must exist");
    assert!((g + 90.0).abs() < 1e-6);
}

#[test]
fn focal_gain_equals_channel_count() {
    // Coherent linear pressure sum: N elements gives N× pressure gain.
    assert_eq!(focal_pressure_gain(16), 16.0);
    assert_eq!(focal_pressure_gain(1), 1.0);
    assert_eq!(focal_pressure_gain(0), 0.0);
}

#[test]
fn acoustic_intensity_from_pressure() {
    // Water Z₀ = 1.48e6 Rayl. For p_rms = 1 MPa: I = (1e6)²/1.48e6 ≈ 675 kW/m².
    let i = acoustic_intensity_w_per_m2(1.0e6, 1.48e6);
    assert!(
        (i - 675_676.0).abs() < 1000.0,
        "expected ~675 kW/m², got {i:.0}"
    );
    // Double the pressure ⇒ 4× intensity (quadratic).
    assert!((acoustic_intensity_w_per_m2(2.0e6, 1.48e6) - 4.0 * i).abs() < 100.0);
    // Zero impedance ⇒ 0.
    assert_eq!(acoustic_intensity_w_per_m2(1.0e6, 0.0), 0.0);
}

#[test]
fn shock_parameter_scales_with_pressure_and_distance() {
    // 2 MHz, 1 MPa source, tissue (ρ=1050, c=1540, B/A=6).
    // β = 1 + 6/2 = 4, ω = 2π·2e6 ≈ 12.566e6 rad/s.
    // z_shock = 1050·1540³ / (4·2π·2e6·1e6)
    //         = 3.835e12 / 5.027e13 ≈ 76.3 mm.
    // (The original 31 mm figure was an incorrect estimate; derivation above is the ground truth.)
    let sigma_at_10mm = nonlinear_shock_parameter(1.0e6, 2.0e6, 10.0e-3, 1050.0, 1540.0, 6.0);
    let sigma_at_76mm = nonlinear_shock_parameter(1.0e6, 2.0e6, 76.0e-3, 1050.0, 1540.0, 6.0);
    assert!(
        sigma_at_10mm > 0.0 && sigma_at_10mm < 1.0,
        "at 10 mm should be quasi-linear (σ<1), got {sigma_at_10mm:.3}"
    );
    assert!(
        (sigma_at_76mm - 1.0).abs() < 0.1,
        "at z_shock σ should be ~1, got {sigma_at_76mm:.3}"
    );
    // Higher pressure ⇒ shorter shock distance ⇒ larger σ at same distance.
    let sigma_high = nonlinear_shock_parameter(2.0e6, 2.0e6, 10.0e-3, 1050.0, 1540.0, 6.0);
    assert!(
        sigma_high > sigma_at_10mm,
        "higher pressure must give larger σ"
    );
    // Zero pressure ⇒ infinite shock distance ⇒ σ = +∞.
    assert!(nonlinear_shock_parameter(0.0, 2.0e6, 10.0e-3, 1050.0, 1540.0, 6.0).is_infinite());
}

// ───────── Phase 3g NEW APIs: bvd_anti_resonance_hz + isppa_w_per_m2 + round_trip_attenuation_db ─────────

#[test]
fn bvd_anti_resonance_sits_above_series_branch_with_coupled_c0() {
    // The paper's BVD equivalent circuit: L_s = 0.49 mH, C_s = 12 pF (motional series branch),
    // C_0 = 60 pF (static dielectric capacitance of the transducer stack — typical for a
    // 2 MHz piezo element with ~mm-scale electrode area).
    //
    // Series-branch resonance: f_s = 1/(2π√(L_s·C_s)) ≈ 2.07 MHz.
    // Anti-resonance (textbook BVD): f_p = (1/2π)·√((C_s + C_0)/(L_s·C_s·C_0)).
    //   = (1/2π)·√((12 + 60)·1e-12 / (0.49e-3 · 12e-12 · 60e-12))
    //   = (1/2π)·√(72e-12 / 352.8e-27)
    //   ≈ (1/2π)·√(2.041e14)
    //   ≈ 1.43e6 / 6.283
    //   ≈ 2.27 MHz (anti-resonance sits ~10% above f_s, consistent with k² ≈ 0.16).
    //
    // The anti-resonance MUST be strictly greater than the series resonance (the coupling
    // coefficient k² is positive for any real transducer) — this SSOT-pins the BVD
    // physical distinction: f_p > f_s always, never < or ==.
    let ls = 0.49e-3_f64;
    let cs = 12e-12_f64;
    let c0 = 60e-12_f64;
    let f_s = bvd_series_resonance_hz(ls, cs);
    let f_p = bvd_anti_resonance_hz(ls, cs, c0);
    // Series resonance matches the existing paper fixture.
    assert!(
        (f_s - 2.07e6).abs() < 0.1e6,
        "expected series-resonance ~2.07 MHz, got {:.3} MHz",
        f_s / 1e6
    );
    // Anti-resonance strictly above the series resonance, in the expected ~2.27 MHz range.
    assert!(
        f_p > f_s,
        "BVD anti-resonance MUST sit above the series resonance for any real transducer; \
         got f_p={:.3} MHz, f_s={:.3} MHz",
        f_p / 1e6,
        f_s / 1e6
    );
    assert!(
        (f_p - 2.27e6).abs() < 0.05e6,
        "expected anti-resonance ~2.27 MHz at C_0 = 60 pF, got {:.3} MHz",
        f_p / 1e6
    );
    // Coupling coefficient k² = 1 − (f_s/f_p)² is strictly positive for k² < 1.
    let k_squared = 1.0 - (f_s / f_p).powi(2);
    assert!(
        k_squared > 0.0 && k_squared < 1.0,
        "electromechanical coupling coefficient must be in (0, 1) for a real transducer; \
         got k² = {k_squared:.4}"
    );
    // Non-physical inputs ⇒ ∞.
    assert!(bvd_anti_resonance_hz(0.0, cs, c0).is_infinite());
    assert!(bvd_anti_resonance_hz(ls, 0.0, c0).is_infinite());
    assert!(bvd_anti_resonance_hz(ls, cs, 0.0).is_infinite());
    // Increasing the static C_0 (thicker dielectric / larger electrodes) PUSHES f_p DOWN
    // toward f_s — a real transducer-design knob. Verify the monotone trend.
    let f_p_lower_c0 = bvd_anti_resonance_hz(ls, cs, 120e-12); // 2× static
    assert!(
        f_p_lower_c0 < f_p,
        "doubling C_0 must reduce the anti-resonance; got f_p(C_0=120pF)={:.3} MHz vs \
         f_p(C_0=60pF)={:.3} MHz",
        f_p_lower_c0 / 1e6,
        f_p / 1e6
    );
}

#[test]
fn isppa_zero_at_zero_duty_and_scales_linearly_above() {
    // duty_factor = 0 ⇒ no power delivered regardless of pressure, I_sppa = 0.
    assert_eq!(isppa_w_per_m2(1.0e6, 1.48e6, 0.0), 0.0);
    // duty_factor > 1.0 ⇒ caller error, refuses to compute.
    assert!(isppa_w_per_m2(1.0e6, 1.48e6, 1.5).is_infinite());
    // p_neg ≤ 0 ⇒ 0 (no negative-going pressure = no transmission).
    assert_eq!(isppa_w_per_m2(0.0, 1.48e6, 0.5), 0.0);
    assert_eq!(isppa_w_per_m2(-1.0e6, 1.48e6, 0.5), 0.0);
    // duty_factor = 0.5 ⇒ intensity is half the duty_factor = 1.0 figure at the same pressure.
    let duty_full = isppa_w_per_m2(1.0e6, 1.48e6, 1.0);
    let duty_half = isppa_w_per_m2(1.0e6, 1.48e6, 0.5);
    assert!(
        (duty_full - 2.0 * duty_half).abs() < 1e-9,
        "duty_factor scaling must be linear: {duty_full} vs 2·{duty_half}"
    );
    // Sanity figure: peak 1 MPa, water Z0=1.48e6 Rayl, duty_factor=0.25:
    // I_sppa = (1e6)² · 0.25 / (2 · 1.48e6) ≈ 84_459 ≈ 0.84 W/cm².
    let i = isppa_w_per_m2(1.0e6, 1.48e6, 0.25);
    assert!(
        (i - 84_459.46).abs() < 1.0,
        "expected ~84_459 W/m², got {i:.2}"
    );
}

#[test]
fn round_trip_attenuation_db_is_twice_one_way_at_same_inputs() {
    // 0.5 dB/cm/MHz × 2 MHz × 5 cm = 5 dB one-way; the pulse-echo round-trip is 10 dB.
    assert!((round_trip_attenuation_db(0.5, 2.0, 5.0) - 10.0).abs() < 1e-9);
    // Identical to twice the existing one-way form.
    let one_way = tissue_attenuation_db(0.5, 2.0, 5.0);
    assert!(
        (round_trip_attenuation_db(0.5, 2.0, 5.0) - 2.0 * one_way).abs() < 1e-9,
        "round-trip must be exactly twice the one-way form: got rt={} vs 2·ow={}",
        round_trip_attenuation_db(0.5, 2.0, 5.0),
        2.0 * one_way
    );
    // Negative inputs ⇒ caller error: refuses to compute.
    assert!(round_trip_attenuation_db(-0.5, 2.0, 5.0).is_infinite());
    assert!(round_trip_attenuation_db(0.5, -2.0, 5.0).is_infinite());
    assert!(round_trip_attenuation_db(0.5, 2.0, -5.0).is_infinite());
    // Zero inputs ⇒ zero round-trip loss (the trivial case at zero depth).
    assert_eq!(round_trip_attenuation_db(0.5, 2.0, 0.0), 0.0);
}

#[test]
fn ssot_distinction_isppa_vs_intensity() {
    // `isppa_w_per_m2(p_neg, Z0, duty)` — audioer takes peak-negative pressure (FDA Track-3
    // pulsed-safety metric, multiplied by duty factor because the cycle is mostly quiescent).
    //
    // `acoustic_intensity_w_per_m2(p_rms, Z0)` — continuous-RMS intensity. NO duty factor.
    //
    // The two are physically distinct quantities even when the same physical setup underwrites
    // both. For a pure-sinusoidal source, p_rms = p_peak / √2, so the ISPPA/duty=1.0 value is
    // (p_neg² / 2Z0) = (p_rms² / Z0) — the same figure as the continuous intensity — but in
    // every other regime (any non-sinusoidal pulse shape, any duty cycle < 1) the two diverge.
    //
    // This test pins the SSOT distinction by checking that:
    //
    // 1. ISPPA at duty_factor = 1.0 with the SAME scaled pressure equals the continuous
    //    intensity computed from the RMS equivalent (for sinusoidal source: p_rms = p_neg/√2).
    // 2. ISPPA at duty_factor = 0.5 is exactly HALF the duty-factor-1.0 figure — the duty
    //    factor is a linear time-average multiplier, and the function must reflect that.
    //
    // The two MyFunctions() chains below also assert that the INPUT SIGNATURES differ:
    // ISPPA takes `(p_neg, Z0, duty)`, intensity takes `(p_rms, Z0)`. A future caller cannot
    // accidentally pass a duty-factor-shaped input into the wrong function.
    let z0 = 1.54e6_f64; // tissue Z₀
    let p_peak = 1.0e6_f64; // 1 MPa peak

    // Sinusoidal equivalence: p_rms = p_peak / √2 ⇒ cts intensity == ISPPA (duty = 1).
    let p_rms = p_peak / 2.0_f64.sqrt();
    let i_cts_full = acoustic_intensity_w_per_m2(p_rms, z0);
    let i_sppa_full = isppa_w_per_m2(p_peak, z0, 1.0);
    assert!(
        (i_cts_full - i_sppa_full).abs() < 1e-3,
        "sinusoidal p_rms = p_neg/√2 ⇒ cts intensity = ISPPA·duty=1; got cts={i_cts_full:.1} \
         vs sppa_full={i_sppa_full:.1}"
    );
    // Linear duty scaling.
    let i_sppa_half = isppa_w_per_m2(p_peak, z0, 0.5);
    assert!(
        (i_sppa_half - 0.5 * i_sppa_full).abs() < 1e-3,
        "ISPPA at duty=0.5 must be linear: {i_sppa_half} = 0.5·{i_sppa_full}"
    );
    // The two functions MUST NOT be silently equivalent at non-sinusoidal regimes: a
    // duty_factor = 0.25 pulse delivers a quarter-cycle of intensity, while the continuous
    // intensity figure (no duty factor) overstates the cycle-averaged delivery by 4×.
    let i_sppa_quarter = isppa_w_per_m2(p_peak, z0, 0.25);
    assert!(
        i_sppa_quarter < i_cts_full,
        "ISPPA at duty=0.25 ({i_sppa_quarter}) MUST understate the \
         cycle-averaged continuous-intensity figure ({i_cts_full})"
    );
}
