//! Finite-Difference Time Domain (FDTD) solver
//!
//! This module implements the FDTD method using Yee's staggered grid scheme
//! for solving Maxwell's equations and acoustic wave equations.
//!
//! # Theory
//!
//! The FDTD method discretizes both space and time using finite differences.
//! Key features include:
//!
//! - **Explicit time stepping**: Direct temporal updates
//! - **Staggered grid (Yee cell)**: Enforces divergence conditions
//! - **Second-order precision**: In both space and time
//! - **CFL-limited stability**: Time step constrained by CFL condition
//!
//! # Algorithm
//!
//! For acoustic waves, the update equations on a staggered grid are:
//! ```text
//! p^{n+1} = p^n - Δt·ρc²·(∇·v)^{n+1/2}
//! v^{n+3/2} = v^{n+1/2} - Δt/ρ·∇p^{n+1}
//! ```
//!
//! # Literature References
//!
//! 1. **Yee, K. S. (1966)**. "Numerical solution of initial boundary value
//!    problems involving Maxwell's equations in isotropic media." *IEEE
//!    Transactions on Antennas and Propagation*, 14(3), 302-307.
//!    DOI: 10.1109/TAP.1966.1138693
//!    - Original Yee algorithm for electromagnetic waves
//!    - Introduction of the staggered grid concept
//!
//! 2. **Virieux, J. (1986)**. "P-SV wave propagation in heterogeneous media:
//!    Velocity-stress finite-difference method." *Geophysics*, 51(4), 889-901.
//!    DOI: 10.1190/1.1442147
//!    - Extension to elastic wave propagation
//!    - Velocity-stress formulation
//!
//! 3. **Graves, R. W. (1996)**. "Simulating seismic wave propagation in 3D
//!    elastic media using staggered-grid finite differences." *Bulletin of the
//!    Seismological Society of America*, 86(4), 1091-1106.
//!    - 3D implementation details
//!    - Higher-order accuracy schemes
//!
//! 4. **Moczo, P., Kristek, J., & Gális, M. (2014)**. "The finite-difference
//!    modelling of earthquake motions: Waves and ruptures." *Cambridge University
//!    Press*. ISBN: 978-1107028814
//!    - Comprehensive treatment of FDTD for wave propagation
//!    - Stability analysis and optimization techniques
//!
//! 5. **Taflove, A., & Hagness, S. C. (2005)**. "Computational electrodynamics:
//!    The finite-difference time-domain method" (3rd ed.). *Artech House*.
//!    ISBN: 978-1580538329
//!    - Definitive reference for FDTD methods
//!    - Topics including subgridding and PML
//!
//! # Implementation Details
//!
//! ## Spatial Derivatives
//!
//! We support 2nd, 4th, and 6th order accurate finite differences:
//! - 2nd order: [-1, 0, 1] / (2Δx)
//! - 4th order: [1/12, -2/3, 0, 2/3, -1/12] / Δx
//! - 6th order: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / Δx
//!
//! ## Stability Condition
//!
//! The CFL condition for FDTD is:
//! ```text
//! Δt ≤ CFL / (c√(1/Δx² + 1/Δy² + 1/Δz²))
//! ```
//! where CFL ≈ 0.95 for stability margin.
//!
//! ## Subgridding
//!
//! Local mesh refinement following:
//! - Berenger, J. P. (2002). "Application of the CFS PML to the absorption of
//!   evanescent waves in waveguides." *IEEE Microwave and Wireless Components
//!   Letters*, 12(6), 218-220.
//!
//! # Design Principles
//! - SOLID: Single responsibility for finite-difference wave propagation
//! - CUPID: Composable with other solvers via plugin architecture
//! - KISS: Explicit time-stepping algorithm
//! - DRY: Reuses grid utilities and boundary conditions
//! - YAGNI: Implements only necessary features for acoustic simulation

// Public modules
pub mod boundary_stencils;
pub mod config;
pub mod finite_difference;
pub mod interpolation;
pub mod metrics;
pub mod plugin;
pub mod solver;
pub mod staggered_grid;

// Re-exports for convenience
pub use config::FdtdConfig;
pub use plugin::FdtdPlugin;
pub use solver::FdtdSolver;
pub use staggered_grid::{FieldComponent, StaggeredGrid};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use ndarray::Array3;

    #[test]
    fn test_fdtd_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_finite_difference_coefficients() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid).unwrap();

        // Check that solver was created successfully with default config
        assert_eq!(solver.config.spatial_order, 4);
    }

    #[test]
    fn test_derivative_computation() {
        use super::finite_difference::FiniteDifference;

        let fd = FiniteDifference::new(2).unwrap();

        // Create a linear field (derivative should be constant)
        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k] = i as f64; // Linear in x
                }
            }
        }

        let deriv = fd.compute_derivative(&field.view(), 0, 1.0).unwrap();

        // Check that derivative is approximately 1.0 in the interior
        for i in 1..9 {
            for j in 1..9 {
                for k in 1..9 {
                    assert!(
                        (deriv[[i, j, k] - 1.0).abs() < 1e-10,
                        "Expected derivative 1.0, got {}",
                        deriv[[i, j, k]
                    );
                }
            }
        }
    }

    #[test]
    fn test_cfl_condition() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid).unwrap();

        let c_max = crate::constants::physics::SOUND_SPEED_WATER;
        let dt = solver.max_stable_dt(c_max);

        // Check that time step is reasonable
        assert!(dt > 0.0);
        assert!(dt < 1e-3); // Should be smaller than spatial step
    }
}

// Tests moved to individual modules
