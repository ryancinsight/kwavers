//! KZK solver construction tests.

use crate::solver::forward::nonlinear::kzk::{KZKConfig, KZKSolver};

#[test]
fn test_kzk_solver_creation() {
    let config = KZKConfig::default();
    let solver = KZKSolver::new(config);
    assert!(
        solver.is_ok(),
        "KZKSolver::new with default config must succeed"
    );
}
