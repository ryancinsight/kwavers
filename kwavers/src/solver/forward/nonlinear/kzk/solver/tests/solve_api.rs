//! `KZKSolver::solve(n)` API invariant tests.

use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::solver::forward::nonlinear::kzk::{KZKConfig, KZKSolver};
use ndarray::Array2;

/// `solve(0)` succeeds and does not change the field.
///
/// ## Theorem
/// Zero propagation steps applies no operators; the identity is preserved.
/// # Panics
/// - Panics if `solve(0) must succeed`.
///
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
    solver.set_source(source, MHZ_TO_HZ);
    let p_before = solver.pressure.clone();

    solver.solve(0).expect("solve(0) must succeed");
    assert_eq!(
        solver.pressure, p_before,
        "solve(0) must not change the pressure field"
    );
}

/// `solve(10)` advances the step counter by exactly 10.
///
/// ## Theorem
/// `solve(n)` calls `step()` exactly n times; counter increments by n.
/// # Panics
/// - Panics if `solve(10) must succeed`.
///
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
    solver.set_source(Array2::from_elem((8, 8), 1000.0_f64), 1e6);

    solver.solve(10).expect("solve(10) must succeed");
    assert_eq!(solver.current_z_step, 10, "current_z_step must equal 10");
}

/// `solve(nz + 1)` returns an error mentioning `n_steps` and `nz`.
///
/// ## Rationale
/// The axial grid has exactly `nz` planes; propagating beyond is undefined.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
    assert!(result.is_err(), "solve(nz+1) must return Err");
    let msg = result.unwrap_err();
    assert!(
        msg.contains("n_steps") && msg.contains("nz"),
        "error message must reference n_steps and nz; got: {msg}"
    );
}

/// `solve(nz)` (full grid) succeeds; boundary case n == nz is valid.
/// # Panics
/// - Panics with `"solve(nz={nz}) must succeed; got: {e:?}"`.
///
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
    solver.set_source(Array2::from_elem((4, 4), 100.0_f64), 1e6);

    solver
        .solve(nz)
        .unwrap_or_else(|e| panic!("solve(nz={nz}) must succeed; got: {e:?}"));
    assert_eq!(
        solver.current_z_step, nz,
        "step counter must equal nz after full propagation"
    );
}

/// `solve(5)` produces a bitwise-identical result to 5 sequential `step()` calls.
///
/// ## Theorem
/// `solve(n)` is exactly equivalent to `step()` called n times from the same
/// initial state; both paths must yield the same pressure field.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
        "solve(5) pressure must equal 5×step() pressure"
    );
    assert_eq!(
        solver_a.current_z_step, solver_b.current_z_step,
        "step counters must match"
    );
}

/// `step()` on zero pressure maps to zero pressure.
///
/// ## Theorem
/// All KZK operators (diffraction, absorption, nonlinearity) map the zero
/// field to itself; the homogeneous solution is p=0.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_zero_pressure_step_is_identity() {
    let config = KZKConfig {
        nx: 16,
        ny: 16,
        nz: 32,
        nt: 20,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config).unwrap();
    let pressure_before = solver.pressure.clone();
    let pressure_prev_before = solver.pressure_prev.clone();

    solver.step();

    assert_eq!(
        solver.pressure, pressure_before,
        "zero-field step must preserve zero pressure"
    );
    assert_eq!(
        solver.pressure_prev, pressure_prev_before,
        "zero-field step must preserve zero previous pressure"
    );
    assert_eq!(solver.current_z_step, 1, "step counter must advance by 1");
}
