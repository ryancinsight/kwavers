use super::*;
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA, TWO_PI};

#[test]
fn minnaert_water_air_bubble() {
    let f = minnaert_resonance_hz(10e-6, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    assert!((f - 327_000.0).abs() / 327_000.0 < 0.05, "f={}", f);
}

#[test]
fn minnaert_surface_tension_correction_reduces_to_uncorrected_at_zero_sigma() {
    // σ = 0 ⇒ the corrected form is exactly the large-bubble Minnaert frequency.
    let r0 = 10e-6;
    let uncorrected = minnaert_resonance_hz(r0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    let corrected =
        minnaert_resonance_corrected_hz(r0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, 0.0);
    assert!((corrected - uncorrected).abs() <= 1e-9 * uncorrected);
}

#[test]
fn minnaert_surface_tension_correction_matches_closed_form_and_scales_with_radius() {
    const SIGMA: f64 = 0.0725; // water [N/m]
    let (gamma, rho) = (1.4, DENSITY_WATER_NOMINAL);

    // Closed-form check: f₀² = [3γP₀ + (3γ−1)·2σ/R₀] / (ρ·(2πR₀)²).
    let r0 = 1e-6;
    let f = minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let laplace = 2.0 * SIGMA / r0;
    let stiffness = 3.0 * gamma * ATMOSPHERIC_PRESSURE + (3.0 * gamma - 1.0) * laplace;
    let expected = (stiffness / rho).sqrt() / (TWO_PI * r0);
    assert!((f - expected).abs() <= 1e-9 * expected, "f={f} expected={expected}");

    // Large bubble (R₀ = 1 mm): 2σ/R₀ ≪ P₀ ⇒ correction is negligible (<0.1%).
    // (At 100 µm it is already a non-negligible ~0.5%, consistent with the 1/R₀
    // scaling — surface tension matters increasingly as the bubble shrinks.)
    let r_big = 1e-3;
    let big_corr = minnaert_resonance_corrected_hz(r_big, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let big_unc = minnaert_resonance_hz(r_big, gamma, ATMOSPHERIC_PRESSURE, rho);
    assert!((big_corr - big_unc).abs() / big_unc < 1e-3, "large-bubble correction must be tiny");

    // Small bubble (R₀ = 1 µm): surface tension raises f₀ by >10% (chapter §5).
    let small_corr = minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let small_unc = minnaert_resonance_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho);
    assert!(
        (small_corr - small_unc) / small_unc > 0.10,
        "sub-micron surface-tension correction must exceed 10%: {small_corr} vs {small_unc}"
    );
    // Invalid σ ⇒ 0.
    assert_eq!(
        minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, f64::NAN),
        0.0
    );
}

#[test]
fn closed_form_cavitation_estimators_reject_invalid_domains() {
    assert_eq!(
        minnaert_resonance_hz(0.0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        minnaert_resonance_hz(10e-6, -1.0, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        blake_threshold_pa(10e-6, ATMOSPHERIC_PRESSURE, f64::NAN),
        0.0
    );
    assert_eq!(
        rayleigh_collapse_time_s(100e-6, -ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        histotripsy_lesion_radius_m(-1.0, 5e-6, ATMOSPHERIC_PRESSURE, 2000.0),
        0.0
    );
}

#[test]
fn rayleigh_collapse_positive() {
    let tc = rayleigh_collapse_time_s(100e-6, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    assert!(tc > 0.0 && tc < 1e-4);
}

#[test]
fn rp_rk4_initial_condition() {
    let t: Vec<f64> = (0..10).map(|i| i as f64 * 1e-9).collect();
    let (r, _) = rayleigh_plesset_rk4(
        10e-6,
        0.0,
        0.0,
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    assert!((r[0] - 10e-6).abs() < 1e-15);
    assert!((r[9] - 10e-6).abs() / 10e-6 < 0.01);
}

#[test]
fn km_rk4_length_matches() {
    let t: Vec<f64> = (0..5).map(|i| i as f64 * 1e-9).collect();
    let (r, v) = keller_miksis_rk4(
        10e-6,
        0.0,
        0.0,
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
        SOUND_SPEED_WATER_SIM,
    );
    assert_eq!(r.len(), 5);
    assert_eq!(v.len(), 5);
}

#[test]
fn km_shelled_zero_xi_matches_bare_bitexact() {
    // xi_s = 0 must reproduce the bare Keller–Miksis solution bit-for-bit:
    // keller_miksis_rk4 is defined as keller_miksis_shelled_rk4(..., 0.0, ..).
    let t: Vec<f64> = (0..400).map(|i| i as f64 * 2e-9).collect();
    let bare = keller_miksis_rk4(
        2e-6,
        0.0,
        0.5 * MPA_TO_PA,
        MHZ_TO_HZ,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.072,
        1.0e-3,
        1.4,
        2_330.0,
        SOUND_SPEED_WATER_SIM,
    );
    let shelled0 = keller_miksis_shelled_rk4(
        2e-6,
        0.0,
        0.5 * MPA_TO_PA,
        MHZ_TO_HZ,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.072,
        1.0e-3,
        1.4,
        2_330.0,
        0.0,
        SOUND_SPEED_WATER_SIM,
    );
    assert_eq!(bare.0, shelled0.0, "radius diverges at xi_s=0");
    assert_eq!(bare.1, shelled0.1, "wall velocity diverges at xi_s=0");
}

#[test]
fn km_shelled_damps_radial_excursion() {
    // Sub-threshold stable-cavitation LIFU regime (ch24 BBB opening): 2 µm
    // coated microbubble, 350 kPa @ 1 MHz, fine fixed-step RK4 (~333 ps). A
    // finite shell viscosity adds wall damping, reducing the sustained
    // oscillation energy relative to the bare bubble.
    let n = 30_000usize; // ch24 N_STEPS_KM: dt = 10 µs / 30000 ≈ 333 ps
    let dt = (10.0 / MHZ_TO_HZ) / n as f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    // Sub-inertial drive: at 100 kPa a 2 µm bubble oscillates stably (no full
    // collapse to the floor), the regime where the fixed-step RK4 is valid and
    // shell damping is the dominant effect. (At ≥350 kPa the *bare* bubble
    // collapses inertially to the clamp — an explicit-integrator limitation, not
    // a shell-model property.)
    let amp = 100e3;
    let (r_bare, _) = keller_miksis_shelled_rk4(
        2e-6,
        0.0,
        amp,
        MHZ_TO_HZ,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.072,
        1.0e-3,
        1.4,
        2_330.0,
        0.0,
        SOUND_SPEED_WATER_SIM,
    );
    let (r_shell, _) = keller_miksis_shelled_rk4(
        2e-6,
        0.0,
        amp,
        MHZ_TO_HZ,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.072,
        1.0e-3,
        1.4,
        2_330.0,
        1.5e-9,
        SOUND_SPEED_WATER_SIM,
    );
    assert!(r_bare.iter().all(|x| x.is_finite()) && r_shell.iter().all(|x| x.is_finite()));
    // Robust damping signature: the *sustained* oscillation energy (variance of
    // R about its mean over the full window) is reduced by a finite shell
    // viscosity. Global-peak monotonicity is not guaranteed because the added
    // damping also phase-shifts the nonlinear transient.
    let var = |r: &[f64]| {
        let m = r.iter().sum::<f64>() / r.len() as f64;
        r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / r.len() as f64
    };
    let var_bare = var(&r_bare);
    let var_shell = var(&r_shell);
    assert!(
        var_shell < var_bare,
        "shell viscosity did not damp oscillation: var_bare={var_bare:.4e} var_shell={var_shell:.4e}"
    );
    // The shell genuinely changes the trajectory (not a no-op delegation).
    let max_diff = r_bare
        .iter()
        .zip(&r_shell)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-9,
        "shell viscosity had no effect: max_diff={max_diff:.2e}"
    );
}

#[test]
fn km_shelled_inertial_collapse_stays_finite() {
    // Super-threshold drive (350 kPa, ch24's highest LIFU amplitude) collapses
    // the 2 µm coated bubble inertially. The stiff shell-damping term would
    // diverge the explicit RK4, but the inertial-collapse arrest must keep the
    // returned trajectory finite and bounded (held at the collapse radius).
    let n = 30_000usize;
    let dt = (10.0 / MHZ_TO_HZ) / n as f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let (r, v) = keller_miksis_shelled_rk4(
        2e-6,
        0.0,
        350e3,
        MHZ_TO_HZ,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.072,
        1.0e-3,
        1.4,
        2_330.0,
        1.5e-9,
        SOUND_SPEED_WATER_SIM,
    );
    assert!(
        r.iter().all(|x| x.is_finite() && *x > 0.0),
        "radius non-finite after collapse arrest"
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "wall velocity non-finite after collapse arrest"
    );
    // The bubble did reach a strong compression before arrest (genuine collapse,
    // not an early bail-out).
    let rmin = r.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        rmin < 1.0e-6,
        "expected strong compression below r0/2, got rmin={rmin:.3e}"
    );
}

#[test]
fn mechanical_index_known_value() {
    let mi = mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
    assert!((mi - 1.0).abs() < 1e-9, "mi={}", mi);
}

#[test]
fn mechanical_index_scales_inversely_with_sqrt_freq() {
    let mi_1 = mechanical_index(MPA_TO_PA, MHZ_TO_HZ);
    let mi_4 = mechanical_index(MPA_TO_PA, 4.0 * MHZ_TO_HZ);
    assert!((mi_1 / mi_4 - 2.0).abs() < 1e-9, "ratio={}", mi_1 / mi_4);
}

#[test]
fn mechanical_index_rejects_invalid_domain() {
    assert_eq!(mechanical_index(MPA_TO_PA, 0.0), 0.0);
    assert_eq!(mechanical_index(MPA_TO_PA, -MHZ_TO_HZ), 0.0);
    assert_eq!(mechanical_index(f64::NAN, MHZ_TO_HZ), 0.0);
}

#[test]
fn icd_zero_driving_gives_zero() {
    let t: Vec<f64> = (0..100).map(|i| i as f64 * 1e-9).collect();
    let (r, rdot) = rayleigh_plesset_rk4(
        10e-6,
        0.0,
        0.0,
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    let icd = inertial_cavitation_dose(&r, &rdot, 10e-6);
    assert_eq!(icd, 0.0, "icd={}", icd);
}

#[test]
fn icd_strong_driving_nonzero() {
    let f0 = 500e3_f64;
    let n_pts = 2000usize;
    let dt = 1.0 / (20.0 * f0);
    let t: Vec<f64> = (0..n_pts).map(|i| i as f64 * dt).collect();
    let r0 = 5e-6;
    let (r, rdot) = rayleigh_plesset_rk4(
        r0,
        0.0,
        5e6,
        f0,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    let icd = inertial_cavitation_dose(&r, &rdot, r0);
    assert!(
        icd > 0.0,
        "expected ICD > 0 for strong driving, got {}",
        icd
    );
}

#[test]
fn lesion_radius_scales_with_icd_cube_root() {
    let r0 = 5e-6;
    let p0 = ATMOSPHERIC_PRESSURE;
    let sigma_y = 2000.0;
    let r1 = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
    let r8 = histotripsy_lesion_radius_m(8.0, r0, p0, sigma_y);
    assert!((r8 / r1 - 2.0).abs() < 1e-9, "r8/r1={}", r8 / r1);
}

#[test]
fn lesion_radius_dimensional_consistency() {
    let r0 = 5e-6;
    let p0 = ATMOSPHERIC_PRESSURE;
    let sigma_y = ATMOSPHERIC_PRESSURE;
    let r_l = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
    assert!((r_l - r0).abs() < 1e-18, "r_l={}, r0={}", r_l, r0);
}

#[test]
fn period_doubling_ratio_no_subharmonic_is_small() {
    let n = 512usize;
    let f0 = MHZ_TO_HZ;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| 1e-7 * (TWO_PI * f0 * i as f64 * dt).sin())
        .collect();
    let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
    let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
    assert!(pd < 0.1, "expected near-zero PD ratio, got {}", pd);
}

#[test]
fn period_doubling_ratio_dominant_subharmonic_exceeds_one() {
    let n = 512usize;
    let f0 = MHZ_TO_HZ;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            1e-7 * (TWO_PI * f0 * t).sin() + 3e-7 * (TWO_PI * f0 * 0.5 * t).sin()
        })
        .collect();
    let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
    let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
    assert!(
        pd > 1.0,
        "expected PD ratio > 1 for dominant subharmonic, got {}",
        pd
    );
}

#[test]
fn bubble_spectrum_length() {
    let r: Vec<f64> = (0..64)
        .map(|i| 10e-6 + 1e-7 * (i as f64 * 0.1).sin())
        .collect();
    let (f, p) = bubble_power_spectrum(&r, 1e-9, 64);
    assert_eq!(f.len(), 33);
    assert_eq!(p.len(), 33);
    assert!(f[0] == 0.0);
    assert!(p.iter().all(|&v| v >= 0.0));
}
