use std::sync::Arc;

use crate::forward::nonlinear::kzk::phase_screen::PhaseScreenOperator;
use crate::forward::nonlinear::kzk::{DiffractionScheme, KZKConfig, KZKSolver};
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::Complex64;
use leto::{Array2, Array3};

#[test]
fn sponge_attenuates_boundary_field() {
    let config = KZKConfig {
        nx: 16,
        ny: 16,
        nt: 4,
        dx: 0.5e-3,
        dz: 0.5e-3,
        dt: 1.0e-8,
        include_absorption: false,
        include_nonlinearity: false,
        sponge_fraction: Some(0.25),
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config).expect("sponge-layer solver construction must work");
    solver.pressure.fill(Complex64::new(1.0, 0.0));

    solver.apply_diffraction(solver.config.dz);

    let edge = solver.pressure[[0, 0, 0]].norm();
    let centre = solver.pressure[[solver.config.nx / 2, solver.config.ny / 2, 0]].norm();
    assert!(edge < 0.5, "edge amplitude should be strongly attenuated");
    assert_relative_eq!(centre, 1.0, epsilon = 1.0e-12);
}

#[test]
fn broadband_matches_wideangle_for_cw() {
    let base = KZKConfig {
        nx: 16,
        ny: 16,
        nz: 4,
        nt: 32,
        dx: 0.5e-3,
        dz: 0.5e-3,
        dt: 1.0 / (32.0 * 1.0e6),
        c0: SOUND_SPEED_WATER_SIM,
        include_absorption: false,
        include_nonlinearity: false,
        frequency: 1.0e6,
        ..Default::default()
    };
    let source = Array2::from_elem((base.nx, base.ny), 1.0_f64);

    let mut broadband_solver = KZKSolver::new(KZKConfig {
        diffraction_scheme: DiffractionScheme::Broadband,
        ..base.clone()
    })
    .expect("broadband solver must construct");
    broadband_solver.set_source(source.clone(), base.frequency);

    let mut wide_angle_solver = KZKSolver::new(KZKConfig {
        diffraction_scheme: DiffractionScheme::WideAngle,
        ..base
    })
    .expect("wide-angle solver must construct");
    wide_angle_solver.set_source(source, 1.0e6);

    broadband_solver.step();
    wide_angle_solver.step();

    let diff_norm_sq: f64 = broadband_solver
        .pressure
        .iter()
        .zip(wide_angle_solver.pressure.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum();
    let ref_norm_sq: f64 = wide_angle_solver
        .pressure
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let relative_error = (diff_norm_sq / ref_norm_sq).sqrt();
    assert_relative_eq!(relative_error, 0.0, epsilon = 1.0e-9);
}

#[test]
fn phase_screen_shifts_phase_proportional_to_delta() {
    let k0 = TWO_PI * 1.0e6 / SOUND_SPEED_WATER_SIM;
    let dz = 0.75e-3;
    let delta = 0.02;
    let mut operator = PhaseScreenOperator::new(k0, dz, 2, 2);
    let delta_slice = Array2::from_elem((2, 2), delta);
    let mut field = Array3::from_elem((2, 2, 3), Complex64::new(1.0, 0.0));

    operator.update(delta_slice.view());
    operator.apply(&mut field);

    let expected = Complex64::from_polar(1.0, k0 * delta * dz);
    for value in field.iter() {
        assert_relative_eq!(value.re, expected.re, epsilon = 1.0e-12);
        assert_relative_eq!(value.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn inhomogeneous_solver_construction() {
    let speed_map = Arc::new(Array3::<f64>::zeros((8, 8, 4)));
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 4,
        nt: 8,
        dx: 0.5e-3,
        dz: 0.5e-3,
        dt: 1.0e-8,
        include_absorption: false,
        include_nonlinearity: false,
        // Use WideAngle so the small test grid (theta_max = 45°) passes validation.
        diffraction_scheme: DiffractionScheme::WideAngle,
        speed_map: Some(speed_map),
        ..Default::default()
    };

    let mut solver =
        KZKSolver::new(config).expect("inhomogeneous-medium solver must construct cleanly");
    solver.step();
}
