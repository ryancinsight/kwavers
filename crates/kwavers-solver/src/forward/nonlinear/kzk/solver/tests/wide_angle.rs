//! Wide-angle diffraction regression tests.

use crate::forward::nonlinear::kzk::complex_parabolic_diffraction::ParabolicDiffractionOperator;
use crate::forward::nonlinear::kzk::wide_angle_diffraction::WideAngleDiffractionOperator;
use crate::forward::nonlinear::kzk::{DiffractionScheme, KZKConfig, KZKSolver};
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::Complex64;
use leto::Array2;

fn gaussian_field(config: &KZKConfig, beam_waist: f64) -> Array2<Complex64> {
    let mut field = Array2::<Complex64>::zeros((config.nx, config.ny));
    let cx = config.nx as f64 / 2.0;
    let cy = config.ny as f64 / 2.0;

    for i in 0..config.nx {
        for j in 0..config.ny {
            let x = (i as f64 - cx) * config.dx;
            let y = (j as f64 - cy) * config.dx;
            let r2 = x * x + y * y;
            field[[i, j]] = Complex64::new((-r2 / (beam_waist * beam_waist)).exp(), 0.0);
        }
    }

    field
}

fn plane_wave_field(config: &KZKConfig, mode_x: usize, mode_y: usize) -> Array2<Complex64> {
    let mut field = Array2::<Complex64>::zeros((config.nx, config.ny));
    let nx = config.nx as f64;
    let ny = config.ny as f64;

    for i in 0..config.nx {
        for j in 0..config.ny {
            let phase = TWO_PI * (mode_x as f64 * i as f64 / nx + mode_y as f64 * j as f64 / ny);
            field[[i, j]] = Complex64::from_polar(1.0, phase);
        }
    }

    field
}

#[test]
fn wide_angle_recovers_paraxial_for_narrow_beam() {
    let config = KZKConfig {
        nx: 64,
        ny: 64,
        dx: 0.5e-3,
        frequency: 1.0e6,
        c0: SOUND_SPEED_WATER_SIM,
        ..Default::default()
    };
    let beam_waist = 10.0e-3;
    let step_size = 0.5e-3;
    let n_steps = 10;

    let mut parabolic = ParabolicDiffractionOperator::new(&config);
    let mut wide_angle = WideAngleDiffractionOperator::new(&config);
    let mut parabolic_field = gaussian_field(&config, beam_waist);
    let mut wide_angle_field = parabolic_field.clone();

    for _ in 0..n_steps {
        let mut parabolic_view = parabolic_field.view_mut();
        parabolic.apply_complex(&mut parabolic_view, step_size);

        let mut wide_angle_view = wide_angle_field.view_mut();
        wide_angle.apply_complex(&mut wide_angle_view, step_size);
    }

    let diff_norm_sq: f64 = parabolic_field
        .iter()
        .zip(wide_angle_field.iter())
        .map(|(parabolic_value, wide_angle_value)| {
            (*parabolic_value - *wide_angle_value).norm_sqr()
        })
        .sum();
    let reference_norm_sq: f64 = parabolic_field.iter().map(|value| value.norm_sqr()).sum();
    let relative_l2_error = (diff_norm_sq / reference_norm_sq).sqrt();

    assert_relative_eq!(relative_l2_error, 0.0, epsilon = 0.01);
}

#[test]
fn wide_angle_handles_steep_angles() {
    let mut config = KZKConfig {
        nx: 64,
        ny: 64,
        nz: 8,
        nt: 16,
        dx: 1.0e-3,
        dz: 1.0e-3,
        dt: 1.0e-8,
        include_absorption: false,
        include_nonlinearity: false,
        diffraction_scheme: DiffractionScheme::WideAngle,
        ..Default::default()
    };
    let source = leto::Array2::from_elem([config.nx, config.ny], 1.0_f64);

    assert!(
        crate::forward::nonlinear::kzk::validate_config(&config).is_ok(),
        "wide-angle validation must allow steep beams"
    );

    config.diffraction_scheme = DiffractionScheme::Parabolic;
    assert!(
        crate::forward::nonlinear::kzk::validate_config(&config).is_err(),
        "parabolic validation must reject the same steep geometry"
    );

    config.diffraction_scheme = DiffractionScheme::WideAngle;
    let mut solver = KZKSolver::new(config).expect("wide-angle KZK solver must construct");
    solver.set_source(source, 1.0e6);
    solver.step();
}

#[test]
fn wide_angle_energy_conservation() {
    let config = KZKConfig {
        nx: 64,
        ny: 64,
        dx: 0.5e-3,
        frequency: 1.0e6,
        c0: SOUND_SPEED_WATER_SIM,
        ..Default::default()
    };
    let mut operator = WideAngleDiffractionOperator::new(&config);
    let mut field = plane_wave_field(&config, 1, 2);
    let initial_energy: f64 = field.iter().map(|value| value.norm_sqr()).sum();

    let mut field_view = field.view_mut();
    operator.apply_complex(&mut field_view, 5.0e-3);

    let final_energy: f64 = field.iter().map(|value| value.norm_sqr()).sum();
    let energy_ratio = final_energy / initial_energy;
    assert_relative_eq!(energy_ratio, 1.0, epsilon = 1.0e-10);
}

#[test]
fn wide_angle_evanescent_decay() {
    let config = KZKConfig {
        nx: 64,
        ny: 64,
        dx: 0.1e-3,
        frequency: 1.0e6,
        c0: SOUND_SPEED_WATER_SIM,
        ..Default::default()
    };
    let step_size = 1.0e-4;
    let evanescent_mode_x = 5;
    let mut operator = WideAngleDiffractionOperator::new(&config);
    let mut field = plane_wave_field(&config, evanescent_mode_x, 0);

    let initial_amplitude = field[[0, 0]].norm();
    let mut field_view = field.view_mut();
    operator.apply_complex(&mut field_view, step_size);
    let final_amplitude = field[[0, 0]].norm();

    let dkx = TWO_PI / (config.nx as f64 * config.dx);
    let kx = evanescent_mode_x as f64 * dkx;
    let k0 = TWO_PI * config.frequency / config.c0;
    let expected_decay = (-(kx * kx - k0 * k0).sqrt() * step_size).exp();
    let measured_decay = final_amplitude / initial_amplitude;

    assert_relative_eq!(measured_decay, expected_decay, epsilon = 1.0e-10);
}
