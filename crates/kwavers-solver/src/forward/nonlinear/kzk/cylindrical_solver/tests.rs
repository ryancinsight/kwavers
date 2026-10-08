//! Tests for the axisymmetric KZK solver.

use super::{CylindricalKZKConfig, CylindricalKZKSolver};
use eunomia::assert_relative_eq;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array1;

fn gaussian_profile(config: &CylindricalKZKConfig, radius_scale: f64) -> Array1<f64> {
    let mut source = Array1::<f64>::zeros(config.nr);
    for i in 0..config.nr {
        let r = (i as f64 + 0.5) * config.dr;
        source[i] = (-(r * r) / (radius_scale * radius_scale)).exp();
    }
    source
}

fn weighted_energy(solver: &CylindricalKZKSolver) -> f64 {
    let mut total = 0.0;
    for i in 0..solver.config.nr {
        let ring_weight = TWO_PI * solver.r[i] * solver.config.dr;
        for t in 0..solver.config.nt {
            total += solver.pressure[[i, t]].norm_sqr() * ring_weight;
        }
    }
    total
}

#[test]
fn cylindrical_axis_boundary_condition() {
    let config = CylindricalKZKConfig {
        nr: 64,
        nt: 64,
        dz: 0.1e-3,
        dr: 0.1e-3,
        include_absorption: false,
        include_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = CylindricalKZKSolver::new(config).expect("cylindrical solver must construct");
    let source = gaussian_profile(&solver.config, 1.0e-3);
    solver.set_source(source, 1.0e6);

    solver.apply_diffraction(solver.config.dz * 0.5);

    let max_amp = (0..solver.config.nt)
        .map(|t| {
            solver.pressure[[0, t]]
                .norm()
                .max(solver.pressure[[1, t]].norm())
        })
        .fold(0.0_f64, f64::max);
    let max_axis_slope = (0..solver.config.nt)
        .map(|t| ((solver.pressure[[1, t]] - solver.pressure[[0, t]]).norm()) / solver.config.dr)
        .fold(0.0_f64, f64::max);

    assert!(
        max_axis_slope <= 0.1 * max_amp / solver.config.dr.max(f64::EPSILON),
        "axis derivative should stay small under the symmetry boundary condition"
    );
}

#[test]
fn cylindrical_diffraction_energy_conserved() {
    let config = CylindricalKZKConfig {
        nr: 96,
        nt: 64,
        dr: 0.1e-3,
        dz: 0.05e-3,
        include_absorption: false,
        include_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = CylindricalKZKSolver::new(config).expect("cylindrical solver must construct");
    let source = gaussian_profile(&solver.config, 0.8e-3);
    solver.set_source(source, 1.0e6);

    let initial_energy = weighted_energy(&solver);
    solver.apply_diffraction(solver.config.dz);
    let final_energy = weighted_energy(&solver);

    assert_relative_eq!(final_energy / initial_energy, 1.0, epsilon = 1.0e-2);
}

#[test]
fn cylindrical_solver_smoke_test() {
    let config = CylindricalKZKConfig {
        nr: 64,
        nz: 16,
        nt: 64,
        dr: 0.1e-3,
        dz: 0.05e-3,
        ..Default::default()
    };
    let mut solver = CylindricalKZKSolver::new(config).expect("cylindrical solver must construct");
    let source = gaussian_profile(&solver.config, 1.0e-3);
    solver.set_source(source, 1.0e6);

    for _ in 0..10 {
        solver.step();
    }

    let field = solver.current_field();
    assert_eq!(field.len(), solver.config.nr);
    assert!(field.iter().all(|value| value.is_finite()));
}
