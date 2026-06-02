//! KZK solver construction tests.
//!
//! # Invariants verified
//!
//! After `KZKSolver::new(config)`:
//!   1. `current_z_step = 0` — no propagation has occurred.
//!   2. `current_time = 0.0` — simulation clock at initial state.
//!   3. `conservation_tracker = None` — diagnostics disabled by default.
//!   4. Pressure field shape = `[nx, ny, nt]` — matches config dimensions.
//!   5. Pressure field initialized to zero — no spurious initial energy.
//!
//! These invariants are contractual: any solver that violates them will produce
//! physically incorrect results on the first call to `step(dz)`.

use crate::forward::nonlinear::kzk::{KZKConfig, KZKSolver};

#[test]
fn test_kzk_solver_creation() {
    let config = KZKConfig::default();
    let nx = config.nx;
    let ny = config.ny;
    let nt = config.nt;

    let solver = KZKSolver::new(config).expect("KZKSolver::new with default config must succeed");

    // Invariant 1: no propagation at construction.
    assert_eq!(
        solver.current_z_step, 0,
        "current_z_step must be 0 at construction"
    );

    // Invariant 2: simulation clock at zero.
    assert_eq!(
        solver.current_time, 0.0,
        "current_time must be 0.0 at construction"
    );

    // Invariant 3: conservation diagnostics disabled by default.
    assert!(
        solver.conservation_tracker.is_none(),
        "conservation_tracker must be None at construction"
    );

    // Invariant 4: pressure field shape matches config.
    let shape = solver.pressure.shape();
    assert_eq!(
        shape,
        [nx, ny, nt],
        "pressure shape must be [nx={nx}, ny={ny}, nt={nt}], got {:?}",
        shape
    );

    // Invariant 5: pressure initialized to zero (no spurious initial energy).
    let max_amp = solver
        .pressure
        .iter()
        .map(|c| c.norm())
        .fold(0.0_f64, f64::max);
    assert_eq!(
        max_amp, 0.0,
        "initial pressure must be exactly zero, got max_amp={max_amp}"
    );
}
