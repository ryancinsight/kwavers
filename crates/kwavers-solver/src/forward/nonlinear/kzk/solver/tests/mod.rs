//! Value-semantic regression tests for the KZK solver.
//!
//! Split by concern:
//! - [`creation`]     — solver construction and defaults
//! - [`beam`]         — Gaussian beam propagation (Tier 1 fast + Tier 3 comprehensive)
//! - [`conservation`] — energy/momentum conservation diagnostics
//! - [`solve_api`]    — `solve(n)` API invariants (zero steps, counter, bounds, parity)
//! - [`wide_angle`]   — exact Helmholtz diffraction selection and behavior

mod beam;
mod conservation;
mod creation;
mod solve_api;
mod wide_angle;

use crate::forward::nonlinear::kzk::{DiffractionScheme, KZKConfig, KZKSolver};

#[test]
fn wide_angle_solver_construction_runs() {
    let config = KZKConfig {
        nx: 16,
        ny: 16,
        nz: 8,
        nt: 8,
        dx: 1.0e-3,
        dz: 1.0e-3,
        dt: 1.0e-8,
        include_absorption: false,
        include_nonlinearity: false,
        diffraction_scheme: DiffractionScheme::WideAngle,
        ..Default::default()
    };

    let mut solver =
        KZKSolver::new(config).expect("wide-angle KZK solver construction must succeed");
    solver.step();
}
