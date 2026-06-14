//! Gaussian beam propagation tests (Tier 1 fast + Tier 3 comprehensive).

use crate::forward::nonlinear::kzk::{KZKConfig, KZKSolver};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use ndarray::Array2;

/// Tier 3 — comprehensive validation (>30 s).
/// Uses a 64×64×128 grid.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
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

    solver.set_source(source, MHZ_TO_HZ);
    for _ in 0..10 {
        solver.step();
    }

    let intensity = solver.get_intensity();
    assert!(
        intensity.sum() > 0.0,
        "intensity must be positive after propagation"
    );
}

/// Tier 1 — fast CI/CD validation (<2 s).
/// Uses a 16×16×32 grid.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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

    solver.set_source(source, MHZ_TO_HZ);
    for _ in 0..3 {
        solver.step();
    }

    let intensity = solver.get_intensity();
    assert!(
        intensity.sum() > 0.0,
        "intensity must be positive after propagation"
    );
}
