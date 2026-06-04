//! Tests for the broadband residual-gas (bubble-cloud) absorption operator.
//!
//! These pin the two correctness claims:
//! 1. the spectral shape `ĝ(f)` is the genuine resonant Commander–Prosperetti
//!    spectrum — it has an interior peak near the Minnaert resonance, which a
//!    power law `f^y` (strictly monotonic) cannot reproduce;
//! 2. the PSTD solver applies that exact spectrum per wavenumber: a pure-mode
//!    pressure field is attenuated by `dt·c₀·m·ĝ(c|k|)`, and two modes are
//!    attenuated in the ratio of the true CP spectrum, not `(f₁/f₂)^y`.

use super::{cp_dispersion_stiffness, cp_spectral_shape};
use crate::forward::pstd::config::PSTDConfig;
use crate::multiphysics::residual_gas_coupling::BubblyMediumProps;
use crate::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_domain::source::GridSource;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use ndarray::Array3;
use std::f64::consts::PI;

/// Minnaert resonance frequency [Hz] of an air bubble of radius `r` in a liquid
/// at ambient pressure `p0`: f_M = (1/2πr)·√(3γp0/ρ).
fn minnaert_freq(r: f64, p0: f64, gamma: f64, rho: f64) -> f64 {
    (3.0 * gamma * p0 / rho).sqrt() / (2.0 * PI * r)
}

#[test]
fn spectral_shape_is_resonant_not_power_law() {
    let r = 2.0e-6;
    let props = BubblyMediumProps::air_water(r, 0.3e6);
    let c = SOUND_SPEED_WATER_SIM;
    let rho = DENSITY_WATER_NOMINAL;
    let f_res = minnaert_freq(r, props.p0, props.polytropic, rho);

    // Normalised at the drive frequency.
    let s_drive = cp_spectral_shape(props.frequency, props.frequency, c, rho, &props);
    assert!((s_drive - 1.0).abs() < 1e-12, "shape(drive) must equal 1");

    // DC carries no attenuation.
    assert_eq!(cp_spectral_shape(0.0, props.frequency, c, rho, &props), 0.0);

    // Interior peak near resonance: a strictly monotonic power law cannot do this.
    let s_low = cp_spectral_shape(f_res / 4.0, props.frequency, c, rho, &props);
    let s_res = cp_spectral_shape(f_res, props.frequency, c, rho, &props);
    let s_high = cp_spectral_shape(f_res * 4.0, props.frequency, c, rho, &props);
    assert!(
        s_res > s_low && s_res > s_high,
        "CP spectrum must peak near resonance (non-power-law): low={s_low:.4}, res={s_res:.4}, high={s_high:.4}"
    );

    // Quantify departure from a power law: no single exponent y fits both the
    // below- and above-resonance ratios. For a power law,
    // ln(s(f)/s(drive))/ln(f/drive) would be a constant y at every f.
    let y_below = (s_low / s_drive).ln() / (f_res / 4.0 / props.frequency).ln();
    let y_above = (s_high / s_drive).ln() / (f_res * 4.0 / props.frequency).ln();
    assert!(
        (y_below - y_above).abs() > 0.5,
        "effective exponent must vary strongly across resonance (genuinely non-power-law): \
         y_below={y_below:.3}, y_above={y_above:.3}"
    );
}

/// Fill `p[i,j,k] = cos(2π·m·i/nx)` — a single x-axis wavenumber mode.
fn fill_mode(p: &mut Array3<f64>, m: usize) {
    let (nx, ny, nz) = p.dim();
    for i in 0..nx {
        let v = (2.0 * PI * m as f64 * i as f64 / nx as f64).cos();
        for j in 0..ny {
            for k in 0..nz {
                p[[i, j, k]] = v;
            }
        }
    }
}

#[test]
fn solver_applies_true_cp_spectrum_per_wavenumber() {
    let nx = 32usize;
    let dx = 1.0e-4;
    let grid = Grid::new(nx, 4, 4, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let dt = 1.0e-8;
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Lossless,
        dt,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    let c = SOUND_SPEED_WATER_SIM;
    let rho = DENSITY_WATER_NOMINAL;
    // Residual cloud: 2 µm bubbles, uniform β, drive below resonance.
    let r = 2.0e-6;
    let props = BubblyMediumProps::air_water(r, 0.3e6);
    let beta = 1.0e-3;
    let void = Array3::from_elem((nx, 4, 4), beta);
    assert!(
        solver.set_residual_gas_absorption(void.view(), c, rho, &props),
        "operator installs when gas is present"
    );

    // Mode frequencies f_m = c·|k_m|/2π = c·m/(nx·dx).
    let freq_of = |m: usize| c * m as f64 / (nx as f64 * dx);
    let m1 = 2usize;
    let m2 = 6usize;

    // Apply one step to each pure mode; measure the realised attenuation factor
    // κ_m = (p_before − p_after)/p_before at i=0 (cos = 1).
    let kappa = |solver: &mut PSTDSolver, m: usize| -> f64 {
        fill_mode(&mut solver.fields.p, m);
        let before = solver.fields.p[[0, 0, 0]];
        solver.apply_residual_gas_absorption().unwrap();
        let after = solver.fields.p[[0, 0, 0]];
        (before - after) / before
    };
    let k1 = kappa(&mut solver, m1);
    let k2 = kappa(&mut solver, m2);

    // The magnitude m(x) is the CP attenuation at the drive frequency.
    let mag = kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_attenuation(
        props.frequency,
        beta,
        r,
        c,
        rho,
        props.mu_liquid,
        props.p0,
        props.polytropic,
    );

    // Solver path must equal the analytic per-mode attenuation dt·c·m·ĝ(f_m).
    for &(m, k_meas) in &[(m1, k1), (m2, k2)] {
        let expected =
            dt * c * mag * cp_spectral_shape(freq_of(m), props.frequency, c, rho, &props);
        let rel = (k_meas - expected).abs() / expected.abs().max(1e-30);
        assert!(
            rel < 1e-6,
            "mode {m}: solver κ={k_meas:.6e} vs analytic dt·c·m·ĝ={expected:.6e} (rel {rel:.2e})"
        );
    }

    // Two-mode ratio follows the TRUE CP spectrum, not a power law (f1/f2)^y.
    let ratio_meas = k1 / k2;
    let ratio_cp = cp_spectral_shape(freq_of(m1), props.frequency, c, rho, &props)
        / cp_spectral_shape(freq_of(m2), props.frequency, c, rho, &props);
    assert!(
        (ratio_meas - ratio_cp).abs() / ratio_cp < 1e-6,
        "mode ratio must match CP spectrum: meas={ratio_meas:.5}, cp={ratio_cp:.5}"
    );
    // Power-law prediction with any fixed exponent y: (f1/f2)^y. f1<f2 so this is
    // < 1 for y>0; the CP ratio straddling resonance is materially different.
    let f_ratio = freq_of(m1) / freq_of(m2);
    let pow_law_y11 = f_ratio.powf(1.1);
    assert!(
        (ratio_cp - pow_law_y11).abs() / ratio_cp > 0.05,
        "CP ratio {ratio_cp:.5} must differ from power-law (f1/f2)^1.1={pow_law_y11:.5}"
    );
}

#[test]
fn solver_applies_cp_dispersion_per_wavenumber() {
    let nx = 32usize;
    let dx = 1.0e-4;
    let grid = Grid::new(nx, 4, 4, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Lossless,
        dt: 1.0e-8,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    let c = SOUND_SPEED_WATER_SIM;
    let rho = DENSITY_WATER_NOMINAL;
    // 2 µm bubbles → Minnaert resonance ≈ 1.6 MHz, between mode 2 (0.94 MHz) and
    // mode 6 (2.81 MHz), so the dispersion stiffness changes sign between them.
    let r = 2.0e-6;
    let props = BubblyMediumProps::air_water(r, 0.3e6);
    let beta = 1.0e-3;
    let beta_ref = 1.0e-4; // SHAPE_REFERENCE_VOID_FRACTION
    let void = Array3::from_elem((nx, 4, 4), beta);
    assert!(solver.set_residual_gas_absorption(void.view(), c, rho, &props));

    let freq_of = |m: usize| c * m as f64 / (nx as f64 * dx);

    // Drive the dispersion EOS term by placing a pure mode in ρ_total (div_u) and
    // zeroing p; the loss term then contributes nothing (FFT of p=0 is 0), so the
    // whole pressure change is the dispersion correction.
    let disp = |solver: &mut PSTDSolver, m: usize| -> f64 {
        solver.fields.p.fill(0.0);
        fill_mode(&mut solver.div_u, m);
        solver.apply_residual_gas_absorption().unwrap();
        solver.fields.p[[0, 0, 0]]
    };
    let d2 = disp(&mut solver, 2);
    let d6 = disp(&mut solver, 6);

    // p = c₀²·(β/β_ref)·ĥ(f_m) at i=0 (ρ_total mode = 1 there).
    for &(m, d_meas) in &[(2usize, d2), (6usize, d6)] {
        let expected =
            c * c * (beta / beta_ref) * cp_dispersion_stiffness(freq_of(m), c, rho, &props);
        let rel = (d_meas - expected).abs() / expected.abs().max(1e-30);
        assert!(
            rel < 1e-6,
            "mode {m}: solver dispersion {d_meas:.6e} vs analytic c₀²·s·ĥ {expected:.6e} (rel {rel:.2e})"
        );
    }

    // Anomalous dispersion: below resonance the stiffness deviation is negative
    // (slowdown, the Wood regime), above resonance it is positive.
    assert!(
        d2 < 0.0,
        "below-resonance dispersion must slow the wave: {d2:.4e}"
    );
    assert!(
        d6 > 0.0,
        "above-resonance dispersion must be anomalous (>0): {d6:.4e}"
    );
}

#[test]
fn no_gas_is_noop_and_clear_works() {
    let grid = Grid::new(16, 4, 4, 1e-4, 1e-4, 1e-4).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Lossless,
        dt: 1e-8,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    // No operator installed → apply is a no-op.
    fill_mode(&mut solver.fields.p, 3);
    let before = solver.fields.p.clone();
    solver.apply_residual_gas_absorption().unwrap();
    assert!(
        solver
            .fields
            .p
            .iter()
            .zip(before.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15),
        "no installed operator must leave p untouched"
    );

    // Zero void fraction → build returns None (nothing installed).
    let zero = Array3::zeros((16, 4, 4));
    let props = BubblyMediumProps::air_water(2e-6, 0.3e6);
    assert!(!solver.set_residual_gas_absorption(
        zero.view(),
        SOUND_SPEED_WATER_SIM,
        DENSITY_WATER_NOMINAL,
        &props
    ));

    // Install then clear.
    let void = Array3::from_elem((16, 4, 4), 1e-3);
    assert!(solver.set_residual_gas_absorption(
        void.view(),
        SOUND_SPEED_WATER_SIM,
        DENSITY_WATER_NOMINAL,
        &props
    ));
    solver.clear_residual_gas_absorption();
    fill_mode(&mut solver.fields.p, 3);
    let before2 = solver.fields.p.clone();
    solver.apply_residual_gas_absorption().unwrap();
    assert!(
        solver
            .fields
            .p
            .iter()
            .zip(before2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15),
        "cleared operator must leave p untouched"
    );
}
