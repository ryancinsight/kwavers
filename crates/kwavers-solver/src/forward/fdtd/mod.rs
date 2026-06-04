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
pub mod avx512_stencil; // Phase 9.1: AVX-512 optimized FDTD stencil
pub mod config;
pub mod dispatch; // Phase 9.1: Runtime SIMD strategy dispatch
pub mod electromagnetic;
pub mod kspace_correction;
pub mod metrics;
pub mod plugin;
pub mod pressure_updater; // SRP: pressure field update methods
pub mod simd_stencil;
pub mod solver;
pub mod velocity_updater; // SRP: velocity field update methods

// Re-exports for convenience
pub use avx512_stencil::{FdtdAvx512Config, FdtdAvx512Metrics, FdtdAvx512StencilProcessor};
pub use config::{FdtdConfig, KSpaceCorrectionMode};
pub use dispatch::{
    get_simd_config, init_simd, DispatchMetrics, FdtdStencilDispatcher, StencilStrategy,
};
pub use electromagnetic::ElectromagneticFdtdSolver;
pub use plugin::FdtdPlugin;
pub use simd_stencil::FdtdSimdStencilProcessor;
pub use solver::FdtdGpuAccelerator;
pub use solver::FdtdSolver;
pub use source_handler::SourceHandler;

pub mod source_handler;

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;

    #[test]
    fn test_fdtd_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = FdtdConfig::default();
        let medium = kwavers_domain::medium::HomogeneousMedium::water(&grid);
        let source = kwavers_domain::source::GridSource::default();
        let _solver = FdtdSolver::new(config, &grid, &medium, source).unwrap();
    }

    #[test]
    fn test_finite_difference_coefficients() {
        // Theorem (4th-order central difference accuracy):
        // For f(x) = sin(k·x) with k = 2π/L and L = n·dx, the 4th-order
        // interior stencil (-f[i+2]+8f[i+1]-8f[i-1]+f[i-2])/(12Δx) approximates
        // ∂f/∂x = k·cos(k·x) with error O(k⁵Δx⁴/30).
        // At 32 ppw (nx=32, dx=1.0, k=2π/32≈0.196): truncation error < 2e-6 per point.
        //
        // Analytical reference: Fornberg (1988) Tables of FD weights.
        use kwavers_math::numerics::operators::{CentralDifference4, DifferentialOperator};
        use ndarray::Array3;
        use std::f64::consts::PI;

        let nx = 32usize;
        let dx = 1.0_f64;
        let k = 2.0 * PI / (nx as f64 * dx); // 1 full period across domain

        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        // Build f(x) = sin(k·x) on a 1D-embedded 3D array (ny=nz=1)
        let mut field = Array3::<f64>::zeros((nx, 1, 1));
        for i in 0..nx {
            field[[i, 0, 0]] = (k * i as f64 * dx).sin();
        }

        let deriv = op.apply_x(field.view()).unwrap();

        // Compute RMS error at interior points [2, nx-3] against k·cos(k·x)
        let mut rms_sq = 0.0_f64;
        let n_interior = (nx - 4) as f64;
        for i in 2..nx - 2 {
            let analytical = k * (k * i as f64 * dx).cos();
            let numerical = deriv[[i, 0, 0]];
            let err = numerical - analytical;
            rms_sq += err * err;
        }
        let rms = (rms_sq / n_interior).sqrt();

        // 4th-order: error ~ k⁵·Δx⁴/30. With k≈0.196, k⁵≈2.9e-4, /30 → ~1e-5.
        // Generous tolerance of 1e-4 to accommodate boundary treatment effects.
        assert!(
            rms < 1e-4,
            "4th-order CD RMS derivative error {rms:.3e} > 1e-4; stencil coefficients incorrect"
        );

        // Also verify the analytical error bound: stencil width must be 5 per Fornberg
        assert_eq!(
            op.stencil_width(),
            5,
            "4th-order stencil must be 5 points wide"
        );
        assert_eq!(op.order(), 4, "operator order must be 4");
    }

    #[test]
    fn test_cfl_condition() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = FdtdConfig::default();
        let medium = kwavers_domain::medium::HomogeneousMedium::water(&grid);
        let source = kwavers_domain::source::GridSource::default();
        let solver = FdtdSolver::new(config, &grid, &medium, source).unwrap();

        let c_max = kwavers_core::constants::SOUND_SPEED_WATER;
        let dt = solver.max_stable_dt(c_max);

        // Check that time step is reasonable
        assert!(dt > 0.0);
        assert!(dt < 1e-3); // Should be smaller than spatial step
    }
}

// Tests moved to individual modules
