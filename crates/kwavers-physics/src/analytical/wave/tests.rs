use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use std::f64::consts::PI;

use super::*;

#[test]
fn standing_wave_nodes_at_zero() {
    let p = standing_wave_1d(1000.0, 1.0, &[0.0], 0.5);
    assert!((p[0]).abs() < 1e-12);
}

#[test]
fn plane_wave_pressure_velocity_preserves_impedance_ratio() {
    let x = [0.0, 0.25, 0.5];
    let (pressure, velocity) =
        plane_wave_pressure_velocity_1d(1.0e5, PI, &x, PI / 2.0, 998.0, 1500.0)
            .expect("invariant: finite positive acoustic medium");

    assert_eq!(pressure.len(), x.len());
    assert_eq!(velocity.len(), x.len());
    for (&p, &u) in pressure.iter().zip(velocity.iter()) {
        assert!((u - p / (998.0 * 1500.0)).abs() < 1.0e-14);
    }
    assert!(pressure[0].abs() < 1.0e-10);
    assert!(pressure[1] > 0.0);
}

#[test]
fn plane_wave_pressure_velocity_rejects_invalid_medium() {
    assert!(plane_wave_pressure_velocity_1d(1.0, 1.0, &[0.0], 0.0, 0.0, 1500.0).is_err());
    assert!(plane_wave_pressure_velocity_1d(1.0, 1.0, &[0.0], 0.0, 998.0, 0.0).is_err());
    assert!(plane_wave_pressure_velocity_1d(1.0, 1.0, &[f64::NAN], 0.0, 998.0, 1500.0).is_err());
}

#[test]
fn gaussian_modulated_pulse_peaks_at_center() {
    let x = [-1.0e-3, 0.0, 1.0e-3];
    let pulse = gaussian_modulated_pulse_1d(&x, 0.0, 2.0e-3, 4.0e-3, 1.0e5)
        .expect("invariant: valid positive pulse parameters");

    assert_eq!(pulse[1], 1.0e5);
    assert!(pulse[0].abs() < pulse[1]);
    assert_eq!(pulse[0], pulse[2]);
}

#[test]
fn dalembert_split_solution_moves_half_amplitude_copies() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let g = [0.0, 0.0, 10.0, 0.0, 0.0];
    let p = dalembert_split_solution_1d(&x, &g, 1.0)
        .expect("invariant: valid sorted coordinate axis and matching samples");

    assert_eq!(p, vec![0.0, 5.0, 0.0, 5.0, 0.0]);
}

#[test]
fn wave_helpers_reject_invalid_input() {
    assert!(gaussian_modulated_pulse_1d(&[0.0], 0.0, 0.0, 1.0, 1.0).is_err());
    assert!(dalembert_split_solution_1d(&[0.0, 0.0], &[1.0, 2.0], 1.0).is_err());
    assert!(dalembert_split_solution_1d(&[0.0, 1.0], &[1.0], 1.0).is_err());
}

#[test]
fn geometric_spreading_envelopes_match_normalized_laws() {
    let radii = [0.01, 0.02, 0.04];
    let (spherical, cylindrical) = geometric_spreading_intensity_envelopes(&radii)
        .expect("invariant: finite positive radius axis");

    assert_eq!(spherical, vec![1.0, 0.25, 0.0625]);
    assert_eq!(cylindrical, vec![1.0, 0.5, 0.25]);
}

#[test]
fn geometric_spreading_envelopes_reject_invalid_radii() {
    assert!(geometric_spreading_intensity_envelopes(&[]).is_err());
    assert!(geometric_spreading_intensity_envelopes(&[0.0]).is_err());
    assert!(geometric_spreading_intensity_envelopes(&[1.0, f64::NAN]).is_err());
}

#[test]
fn reflection_plus_transmission_identity() {
    let z1 = 1_480_000.0_f64;
    let z2 = 7_800_000.0_f64;
    let r = reflection_pressure_coeff(z1, z2);
    let t = transmission_pressure_coeff(z1, z2);
    assert!((1.0 + r - t).abs() < 1e-10);
}

#[test]
fn shock_distance_positive() {
    let xs = shock_formation_distance(MPA_TO_PA, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM, 1000.0, 3.5);
    assert!(xs > 0.0);
}

#[test]
fn fdtd_cfl_1d() {
    let cfl = fdtd_cfl_limit(1);
    assert!((cfl - 1.0).abs() < 1e-10);
}

#[test]
fn fdtd_cfl_3d() {
    let cfl = fdtd_cfl_limit(3);
    assert!((cfl - 1.0 / 3.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn fdtd_cfl_stability_region_marks_component_ball() {
    let cfl_x = [0.0, 0.8, 1.0];
    let cfl_z = [0.0, 0.8];
    let region =
        fdtd_cfl_stability_region_2d(&cfl_x, &cfl_z).expect("invariant: finite Courant axes");

    assert_eq!(region, vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn centered_fd_modified_wavenumber_matches_stencil_symbols() {
    let kh = [0.0, PI / 2.0, PI];
    let second = centered_fd_modified_wavenumber(&kh, 2).expect("invariant: supported order");
    let fourth = centered_fd_modified_wavenumber(&kh, 4).expect("invariant: supported order");
    let sixth = centered_fd_modified_wavenumber(&kh, 6).expect("invariant: supported order");

    assert_eq!(second[0], 0.0);
    assert!((second[1] - 1.0).abs() < 1.0e-12);
    assert!(second[2].abs() < 1.0e-12);
    assert!((fourth[1] - 4.0 / 3.0).abs() < 1.0e-12);
    assert!((sixth[1] - 22.0 / 15.0).abs() < 1.0e-12);
}

#[test]
fn kspace_temporal_correction_matches_sinc() {
    let kh = [0.0, PI];
    let correction =
        kspace_temporal_correction(&kh, 0.5).expect("invariant: finite correction inputs");

    assert_eq!(correction[0], 1.0);
    assert!((correction[1] - (PI / 4.0).sin() / (PI / 4.0)).abs() < 1.0e-12);
}

#[test]
fn numerical_method_helpers_reject_invalid_inputs() {
    assert!(centered_fd_modified_wavenumber(&[0.0], 8).is_err());
    assert!(centered_fd_modified_wavenumber(&[f64::NAN], 2).is_err());
    assert!(kspace_temporal_correction(&[0.0], f64::INFINITY).is_err());
    assert!(fdtd_cfl_stability_region_2d(&[0.0], &[f64::NAN]).is_err());
}

#[test]
fn fubini_n1_sigma0_is_one() {
    // limit: B_1(0) = 1 (from the pre-shock Fubini series)
    let b = fubini_harmonic_amplitude(1, 0.0);
    assert!((b - 1.0).abs() < 1e-8, "b={}", b);
}

#[test]
fn fubini_waveform_zero_sigma_is_sinusoid() {
    let freq = 1.0e6_f64;
    let p0 = 1.0e6_f64;
    let t = [0.0, 0.25 / freq, 0.5 / freq, 0.75 / freq, 1.0 / freq];
    let p = fubini_waveform(&t, p0, freq, 0.0, 8);

    for (&actual, &ti) in p.iter().zip(t.iter()) {
        let expected = p0 * (2.0 * PI * freq * ti).sin();
        // Same f64 sinusoid expression as the sigma=0 kernel branch; 1e-6 Pa is
        // below one micro-part-per-million of the 1 MPa test amplitude.
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn fubini_waveform_matches_harmonic_expansion() {
    let freq = 1.0e6_f64;
    let p0 = 1.0e6_f64;
    let sigma = 0.5_f64;
    let n_max = 5;
    let t = [0.0, 0.125 / freq, 0.25 / freq, 0.375 / freq, 0.5 / freq];
    let p = fubini_waveform(&t, p0, freq, sigma, n_max);

    for (&actual, &ti) in p.iter().zip(t.iter()) {
        let expected = (1..=n_max).fold(0.0, |acc, n| {
            let harmonic = n as f64;
            acc + p0 * fubini_harmonic_amplitude(n, sigma) * (harmonic * 2.0 * PI * freq * ti).sin()
        });
        // Same sequential harmonic order as fubini_waveform; 1e-6 Pa is below
        // one micro-part-per-million of the 1 MPa test amplitude.
        assert!((actual - expected).abs() < 1.0e-6);
    }
}

#[test]
fn pstd_error_is_zero() {
    let err = pstd_phase_error(&[0.1, 0.5, 1.0, PI]);
    assert!(err.iter().all(|&e| e == 0.0));
}

#[test]
fn westervelt_length_consistency() {
    let z = vec![0.0, 0.01, 0.02];
    let w = westervelt_harmonic_evolution(
        &z,
        1e5,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
        1000.0,
        3.5,
        1.0,
        3,
    );
    assert_eq!(w.len(), 3);
    assert_eq!(w[0].len(), 3);
}

#[test]
fn fdtd_phase_error_small_kh_near_zero() {
    // For k·Δx → 0, the FDTD dispersion error must vanish (long wavelengths are exact).
    // With CFL = 0.5, the pre-fix formula gave ~100% error even at kh → 0.
    let small_kh = vec![1e-6_f64];
    let err = fdtd_phase_error_1d(&small_kh, 0.5);
    assert!(
        err[0].abs() < 1e-6,
        "Expected near-zero FDTD dispersion error at small kh, got {}",
        err[0]
    );
}

#[test]
fn fdtd_phase_error_cfl_unity_is_zero() {
    // At CFL = 1 the 1-D FDTD scheme is non-dispersive: error = 0 for all kh.
    use std::f64::consts::PI;
    let kh_arr: Vec<f64> = (1..10).map(|i| i as f64 * PI / 10.0).collect();
    let err = fdtd_phase_error_1d(&kh_arr, 1.0);
    for &e in &err {
        assert!(
            e.abs() < 1e-10,
            "Expected zero dispersion at CFL=1, got {}",
            e
        );
    }
}

#[test]
fn stokes_kirchhoff_dc_is_zero() {
    // α_SK(0) = 0: zero absorption at zero frequency.
    let alpha = stokes_kirchhoff_absorption_np_m(&[0.0], 4.33e-6, 1500.0);
    assert_eq!(alpha[0], 0.0);
}

#[test]
fn stokes_kirchhoff_quadratic_scaling() {
    // α_SK ∝ f²: doubling frequency must quadruple absorption.
    let delta = 4.33e-6_f64; // m²/s, water 20°C
    let c0 = 1500.0_f64;
    let alpha = stokes_kirchhoff_absorption_np_m(&[1e6, 2e6], delta, c0);
    let ratio = alpha[1] / alpha[0];
    assert!(
        (ratio - 4.0).abs() < 1e-10,
        "Expected quadratic scaling (ratio=4), got {ratio}"
    );
}

#[test]
fn stokes_kirchhoff_formula_match() {
    // Direct formula check at f = 1 MHz, water 20°C.
    // α = δ·(2πf)²/(2c³)
    let f = 1e6_f64;
    let delta = 4.33e-6_f64;
    let c0 = 1500.0_f64;
    let expected = delta * (2.0 * PI * f).powi(2) / (2.0 * c0 * c0 * c0);
    let alpha = stokes_kirchhoff_absorption_np_m(&[f], delta, c0);
    assert!(
        (alpha[0] - expected).abs() < 1e-20,
        "Formula mismatch: got {}, expected {expected}",
        alpha[0]
    );
}
