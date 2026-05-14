//! Conservation diagnostics tests for the KZK solver.

use crate::math::fft::Complex64;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};
use crate::solver::forward::nonlinear::kzk::{KZKConfig, KZKSolver};
use ndarray::Array2;

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

    assert!(
        solver.conservation_tracker.is_some(),
        "tracker must be active"
    );
    assert!(
        solver.is_solution_valid(),
        "solution must be valid after 4 steps"
    );

    let summary = solver
        .get_conservation_summary()
        .expect("conservation summary must be available");
    // Summary string must be non-empty (tracker produces a diagnostic line).
    assert!(
        !summary.is_empty(),
        "conservation summary string must not be empty"
    );
}

/// Energy = p²/(2·ρ₀·c₀²) integrated over the domain volume.
///
/// With uniform pressure `p = P`, the analytical total energy is:
/// `E = P²/(2·ρ₀·c₀²) · V`
/// where `V = Lx · Ly · Lz = (nx·dx)·(ny·dx)·(nt·dt·c₀)`.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
    assert!(
        energy > 0.0,
        "energy must be positive for non-zero pressure"
    );

    let p = 1000.0_f64;
    let rho0 = config.rho0;
    let c0 = config.c0;
    let volume = (config.nx as f64 * config.dx)
        * (config.ny as f64 * config.dx)
        * (config.nt as f64 * config.dt * c0);
    let expected = p * p / (2.0 * rho0 * c0 * c0) * volume;

    let relative_error = (energy - expected).abs() / expected;
    assert!(
        relative_error < 1e-10,
        "energy relative error {relative_error} exceeds 1e-10"
    );
}

#[test]
fn test_conservation_diagnostics_disable() {
    // Minimal grid: the disable behavior (enable→None→step survives) is grid-size-independent.
    // KZKConfig::default() allocates 2×128×128×1000 Complex64 ≈ 524 MB unnecessarily here.
    let config = KZKConfig {
        nx: 8,
        ny: 8,
        nz: 16,
        nt: 2,
        dx: 1e-3,
        dz: 1e-3,
        dt: 1e-8,
        ..Default::default()
    };
    let mut solver = KZKSolver::new(config).unwrap();

    solver.enable_conservation_diagnostics(ConservationTolerances::default());
    assert!(
        solver.conservation_tracker.is_some(),
        "tracker must be active after enable"
    );

    solver.disable_conservation_diagnostics();
    assert!(
        solver.conservation_tracker.is_none(),
        "tracker must be None after disable"
    );

    solver.step();
    assert!(
        solver.is_solution_valid(),
        "step after disable must leave valid state"
    );
}

/// The conservation check fires once at step 5 (check_interval = 5).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
    solver.enable_conservation_diagnostics(ConservationTolerances {
        check_interval: 5,
        ..Default::default()
    });

    let source = Array2::from_elem((config.nx, config.ny), 1000.0);
    solver.set_source(source, 1e6);

    for _ in 0..5 {
        solver.step();
    }

    if let Some(ref tracker) = solver.conservation_tracker {
        assert!(
            !tracker.history.is_empty(),
            "conservation check must have fired by step 5"
        );
    }
}
