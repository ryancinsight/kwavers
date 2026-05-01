//! Value-semantic regression tests for the KZK solver.

use super::KZKSolver;
use crate::math::fft::Complex64;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};
use crate::solver::forward::nonlinear::kzk::KZKConfig;
use ndarray::Array2;

#[test]
fn test_kzk_solver_creation() {
    let config = KZKConfig::default();
    let solver = KZKSolver::new(config);
    assert!(solver.is_ok());
}

/// Test Gaussian beam propagation (COMPREHENSIVE - Tier 3)
///
/// This test uses a 64×64×128 grid for thorough validation.
/// Execution time: >30s, classified as Tier 3 comprehensive validation.
#[test]
#[ignore = "Tier 3: Comprehensive validation (>30s execution time)"]
fn test_gaussian_beam_propagation() {
    let mut config = KZKConfig {
        nx: 64,
        ny: 64,
        nz: 128,
        nt: 100,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    config.include_nonlinearity = false;

    let mut solver = KZKSolver::new(config.clone()).unwrap();

    let mut source = Array2::zeros((config.nx, config.ny));
    let cx = config.nx as f64 / 2.0;
    let cy = config.ny as f64 / 2.0;
    let sigma: f64 = 10.0;

    for j in 0..config.ny {
        for i in 0..config.nx {
            let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma.powi(2);
            source[[i, j]] = (-r2).exp();
        }
    }

    solver.set_source(source, 1e6);

    for _ in 0..10 {
        solver.step();
    }

    let intensity = solver.get_intensity();
    assert!(intensity.sum() > 0.0);
}

/// Test Gaussian beam propagation (FAST - Tier 1)
///
/// Fast version with reduced grid (16×16×32) for CI/CD.
/// Execution time: <2s, classified as Tier 1 fast validation.
#[test]
fn test_gaussian_beam_propagation_fast() {
    let mut config = KZKConfig {
        nx: 16,
        ny: 16,
        nz: 32,
        nt: 20,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    config.include_nonlinearity = false;

    let mut solver = KZKSolver::new(config.clone()).unwrap();

    let mut source = Array2::zeros((config.nx, config.ny));
    let cx = config.nx as f64 / 2.0;
    let cy = config.ny as f64 / 2.0;
    let sigma: f64 = 3.0;

    for j in 0..config.ny {
        for i in 0..config.nx {
            let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma.powi(2);
            source[[i, j]] = (-r2).exp();
        }
    }

    solver.set_source(source, 1e6);

    for _ in 0..3 {
        solver.step();
    }

    let intensity = solver.get_intensity();
    assert!(intensity.sum() > 0.0);
}

#[test]
fn test_conservation_diagnostics_integration() {
    let mut config = KZKConfig {
        nx: 16,
        ny: 16,
        nz: 32,
        nt: 20,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    config.include_nonlinearity = false;

    let mut solver = KZKSolver::new(config.clone()).unwrap();

    let tolerances = ConservationTolerances {
        absolute_tolerance: 1e-6,
        relative_tolerance: 1e-4,
        check_interval: 2,
    };
    solver.enable_conservation_diagnostics(tolerances);

    let mut source = Array2::zeros((config.nx, config.ny));
    let cx = config.nx as f64 / 2.0;
    let cy = config.ny as f64 / 2.0;
    let sigma: f64 = 3.0;

    for j in 0..config.ny {
        for i in 0..config.nx {
            let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma.powi(2);
            source[[i, j]] = (-r2).exp();
        }
    }

    solver.set_source(source, 1e6);

    for _ in 0..4 {
        solver.step();
    }

    assert!(solver.conservation_tracker.is_some());
    assert!(solver.is_solution_valid());

    let summary = solver.get_conservation_summary();
    assert!(summary.is_some());
}

#[test]
fn test_conservation_energy_calculation() {
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };

    let mut solver = KZKSolver::new(config.clone()).unwrap();

    solver.pressure.fill(Complex64::new(1000.0, 0.0));

    let energy = solver.calculate_total_energy();

    assert!(energy > 0.0);

    let p = 1000.0;
    let rho0 = config.rho0;
    let c0 = config.c0;
    let volume = (config.nx as f64 * config.dx)
        * (config.ny as f64 * config.dx)
        * (config.nt as f64 * config.dt * c0);
    let expected = p * p / (2.0 * rho0 * c0 * c0) * volume;

    let relative_error = (energy - expected).abs() / expected;
    assert!(relative_error < 1e-10, "Energy calculation error too large");
}

#[test]
fn test_conservation_diagnostics_disable() {
    let config = KZKConfig::default();
    let mut solver = KZKSolver::new(config).unwrap();

    solver.enable_conservation_diagnostics(ConservationTolerances::default());
    assert!(solver.conservation_tracker.is_some());

    solver.disable_conservation_diagnostics();
    assert!(solver.conservation_tracker.is_none());

    solver.step();
    assert!(solver.is_solution_valid());
}

#[test]
fn test_conservation_check_interval() {
    let mut config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    config.include_nonlinearity = false;

    let mut solver = KZKSolver::new(config.clone()).unwrap();

    let tolerances = ConservationTolerances {
        check_interval: 5,
        ..Default::default()
    };
    solver.enable_conservation_diagnostics(tolerances);

    let source = Array2::from_elem((config.nx, config.ny), 1000.0);
    solver.set_source(source, 1e6);

    for _ in 0..4 {
        solver.step();
    }

    solver.step();

    if let Some(ref tracker) = solver.conservation_tracker {
        assert!(
            !tracker.history.is_empty(),
            "Conservation check should have been performed at step 5"
        );
    }
}

/// `solve(0)` must succeed and leave the field unchanged.
///
/// ## Theorem
/// Zero propagation steps applies no operators; the identity is preserved.
#[test]
fn test_kzk_solve_zero_steps() {
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        include_diffraction: false,
        include_absorption: false,
        include_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config).unwrap();
    let source = Array2::from_elem((8, 8), 500.0_f64);
    solver.set_source(source, 1e6);
    let p_before = solver.pressure.clone();

    solver.solve(0).expect("solve(0) must succeed");
    assert_eq!(
        solver.pressure, p_before,
        "solve(0) must not change the field"
    );
}

/// `solve(10)` advances the internal step counter by 10.
///
/// ## Theorem
/// Each call to `step()` increments `current_z_step` by 1; `solve(n)` calls
/// `step()` exactly n times, so the counter increases by n.
#[test]
fn test_kzk_solve_basic_propagation() {
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        include_nonlinearity: false,
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config).unwrap();
    let source = Array2::from_elem((8, 8), 1000.0_f64);
    solver.set_source(source, 1e6);

    solver.solve(10).expect("solve(10) must succeed");
    assert_eq!(solver.current_z_step, 10, "current_z_step should be 10");
}

/// `solve(nz + 1)` must return an error.
///
/// ## Rationale
/// The axial grid has exactly `nz` planes; propagating beyond it is undefined.
#[test]
fn test_kzk_solve_exceeds_nz_returns_error() {
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config.clone()).unwrap();
    let result = solver.solve(config.nz + 1);
    assert!(result.is_err(), "solve(nz+1) must return Err, got Ok");
    let msg = result.unwrap_err();
    assert!(
        msg.contains("n_steps") && msg.contains("nz"),
        "error message should mention n_steps and nz, got: {msg}"
    );
}

/// `solve(nz)` (full grid) must complete without error.
///
/// ## Theorem
/// `solve(n)` for n ≤ nz must not return an error; boundary case n == nz is valid.
#[test]
fn test_kzk_solve_full_propagation() {
    let config = KZKConfig {
        nx: 4,
        ny: 4,
        nz: 8,
        nt: 5,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        include_nonlinearity: false,
        ..Default::default()
    };
    let nz = config.nz;
    let mut solver = KZKSolver::new(config).unwrap();
    let source = Array2::from_elem((4, 4), 100.0_f64);
    solver.set_source(source, 1e6);

    let result = solver.solve(nz);
    assert!(result.is_ok(), "solve(nz) must succeed, got: {:?}", result);
    assert_eq!(
        solver.current_z_step, nz,
        "step counter must equal nz after full propagation"
    );
}

/// `solve(5)` produces the same result as 5 sequential `step()` calls.
///
/// ## Theorem
/// `solve(n)` is exactly equivalent to calling `step()` n times on an
/// identical initial state; both paths must yield bitwise-equal fields.
#[test]
fn test_kzk_solve_matches_manual_step_loop() {
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 10,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        include_nonlinearity: false,
        ..Default::default()
    };
    let source = Array2::from_elem((8, 8), 800.0_f64);

    let mut solver_a = KZKSolver::new(config.clone()).unwrap();
    solver_a.set_source(source.clone(), 1e6);
    solver_a.solve(5).unwrap();

    let mut solver_b = KZKSolver::new(config).unwrap();
    solver_b.set_source(source, 1e6);
    for _ in 0..5 {
        solver_b.step();
    }

    assert_eq!(
        solver_a.pressure, solver_b.pressure,
        "solve(5) must match 5×step()"
    );
    assert_eq!(solver_a.current_z_step, solver_b.current_z_step);
}
