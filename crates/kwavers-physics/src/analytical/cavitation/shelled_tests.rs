use super::*;
use crate::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude;
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn km_shelled_zero_xi_matches_bare_bitexact() {
    // xi_s = 0 must reproduce the bare Keller–Miksis solution bit-for-bit:
    // keller_miksis_rk4 is defined as keller_miksis_shelled_rk4(..., 0.0, ..).
    let t: Vec<f64> = (0..400).map(|i| i as f64 * 2e-9).collect();
    let bare = keller_miksis_rk4(
        2e-6,
        0.0,
        PressureAmplitude::from_pascals(0.5 * MPA_TO_PA),
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
        PressureAmplitude::from_pascals(0.5 * MPA_TO_PA),
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
        PressureAmplitude::from_pascals(amp),
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
        PressureAmplitude::from_pascals(amp),
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
        PressureAmplitude::from_pascals(350e3),
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
