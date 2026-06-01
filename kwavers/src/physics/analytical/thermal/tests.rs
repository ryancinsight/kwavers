use super::*;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::core::constants::tissue_acoustics::DENSITY_BLOOD;
use crate::core::constants::tissue_thermal::{SPECIFIC_HEAT_BLOOD, SPECIFIC_HEAT_TISSUE};

#[test]
fn bioheat_at_t0_is_body_temp() {
    let t = bioheat_focal_temperature_rise(
        &[0.0],
        10.0,
        1e-6,
        0.5,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_TISSUE,
        5.0,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_BLOOD,
        BODY_TEMPERATURE_C,
    );
    assert!((t[0] - BODY_TEMPERATURE_C).abs() < 1e-8);
}

#[test]
fn bioheat_monotone_increasing() {
    let tvec: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let temp = bioheat_focal_temperature_rise(
        &tvec,
        10.0,
        1e-6,
        0.5,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_TISSUE,
        5.0,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_BLOOD,
        BODY_TEMPERATURE_C,
    );
    for i in 1..temp.len() {
        assert!(
            temp[i] >= temp[i - 1],
            "T[{}]={} < T[{}]={}",
            i,
            temp[i],
            i - 1,
            temp[i - 1]
        );
    }
}

#[test]
fn bioheat_approaches_steady_state() {
    let t_long = 3600.0_f64; // 1 hour — far beyond τ
    let t = bioheat_focal_temperature_rise(
        &[0.0, t_long],
        10.0,
        1e-6,
        0.5,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_TISSUE,
        5.0,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_BLOOD,
        BODY_TEMPERATURE_C,
    );
    // Should saturate: T(∞) < T_body + some bound
    assert!(t[1] > BODY_TEMPERATURE_C && t[1] < 200.0);
    // Verify saturation: T(t_long) ≈ T(t_long/2)
    let t_half = bioheat_focal_temperature_rise(
        &[t_long / 2.0],
        10.0,
        1e-6,
        0.5,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_TISSUE,
        5.0,
        DENSITY_BLOOD,
        SPECIFIC_HEAT_BLOOD,
        BODY_TEMPERATURE_C,
    );
    assert!((t[1] - t_half[0]).abs() / t[1].abs() < 0.01);
}

#[test]
fn beer_lambert_intensity_at_zero_is_surface() {
    let i = acoustic_intensity_depth_profile(&[0.0], 7.0, 1.0);
    assert!((i[0] - 1.0).abs() < 1e-12);
}

#[test]
fn beer_lambert_intensity_monotone_decreasing() {
    let z: Vec<f64> = vec![0.0, 0.01, 0.02, 0.04];
    let i = acoustic_intensity_depth_profile(&z, 7.0, 1.0);
    for k in 1..i.len() {
        assert!(
            i[k] < i[k - 1],
            "I[{}]={} ≥ I[{}]={}",
            k,
            i[k],
            k - 1,
            i[k - 1]
        );
    }
}

#[test]
fn beer_lambert_power_deposition_peak_at_surface() {
    // Q(0) = 2·α·I₀ is the maximum; Q(z) < Q(0) for z > 0
    let z: Vec<f64> = vec![0.0, 0.01];
    let q = acoustic_power_deposition_depth_profile(&z, 7.0, 1.0);
    assert!((q[0] - 2.0 * 7.0 * 1.0).abs() < 1e-12, "Q(0)={}", q[0]);
    assert!(q[1] < q[0]);
}

#[test]
fn hifu_gain_positive() {
    let g = hifu_focal_pressure_gain(0.1, 1.5, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
    assert!(g > 1.0, "g={}", g);
}

#[test]
fn gaussian_deposition_peak_at_focus() {
    let r = vec![0.0];
    let z: Vec<f64> = vec![-5e-3, 0.0, 5e-3];
    let q = gaussian_power_deposition_2d(
        &r,
        &z,
        MHZ_TO_HZ,
        0.0,
        MPA_TO_PA,
        SOUND_SPEED_WATER_SIM,
        DENSITY_BLOOD,
        1.0,
        1e-3,
    );
    // Q at focus (z=0) should exceed Q at z=±5mm
    assert!(q[1] > q[0] && q[1] > q[2]);
}

#[test]
fn heat_source_density_zero_pressure_is_zero() {
    let q = acoustic_heat_source_density(&[0.0], 7.0, 1060.0, 1540.0);
    assert!(q[0].abs() < 1e-30, "Q(0)={}", q[0]);
}

#[test]
fn heat_source_density_formula_matches_analytic() {
    // Q = α·p²/(ρ·c)
    // For p = 1 MPa, α = 7 Np/m, ρ = 1060 kg/m³, c = 1540 m/s:
    // Q = 7 × (1e6)² / (1060 × 1540) ≈ 4283.5 W/m³  (per Pascal²)
    let p = 1.0e6_f64;
    let alpha = 7.0_f64;
    let rho = 1060.0_f64;
    let c = 1540.0_f64;
    let expected = alpha * p * p / (rho * c);
    let q = acoustic_heat_source_density(&[p], alpha, rho, c);
    assert!(
        (q[0] - expected).abs() / expected < 1e-12,
        "Q={} expected={}",
        q[0],
        expected
    );
}

#[test]
fn heat_source_density_quadratic_in_pressure() {
    // Doubling pressure → quadrupling Q
    let alpha = 5.0_f64;
    let rho = 1040.0_f64;
    let c = 1543.0_f64;
    let q1 = acoustic_heat_source_density(&[1.0e5], alpha, rho, c)[0];
    let q2 = acoustic_heat_source_density(&[2.0e5], alpha, rho, c)[0];
    assert!(
        (q2 / q1 - 4.0).abs() < 1e-10,
        "quadratic scaling violated: q2/q1={}",
        q2 / q1
    );
}

#[test]
fn acoustic_intensity_from_amplitude_formula() {
    // I = p²/(2ρc)
    let p = 1.0e6_f64; // 1 MPa
    let rho = DENSITY_WATER_NOMINAL;
    let c = SOUND_SPEED_WATER_SIM;
    let expected = p * p / (2.0 * rho * c);
    let i = acoustic_intensity_from_amplitude(&[p], rho, c);
    assert!(
        (i[0] - expected).abs() / expected < 1e-12,
        "I={} expected={expected}",
        i[0]
    );
}

#[test]
fn acoustic_intensity_quadratic_in_pressure() {
    // Doubling pressure → 4× intensity
    let rho = 1060.0_f64;
    let c = 1540.0_f64;
    let i1 = acoustic_intensity_from_amplitude(&[5.0e5], rho, c)[0];
    let i2 = acoustic_intensity_from_amplitude(&[1.0e6], rho, c)[0];
    assert!(
        (i2 / i1 - 4.0).abs() < 1e-10,
        "quadratic violated: i2/i1={}",
        i2 / i1
    );
}

#[test]
fn acoustic_intensity_heat_source_identity() {
    // Q = α·p²/(ρc) = 2α·I: acoustic_heat_source_density = 2α · acoustic_intensity_from_amplitude
    let p = 2.5e5_f64;
    let alpha = 6.0_f64;
    let rho = 1050.0_f64;
    let c = 1580.0_f64;
    let q = acoustic_heat_source_density(&[p], alpha, rho, c)[0];
    let i = acoustic_intensity_from_amplitude(&[p], rho, c)[0];
    assert!(
        (q - 2.0 * alpha * i).abs() / q < 1e-12,
        "Q={q} 2αI={}",
        2.0 * alpha * i
    );
}
