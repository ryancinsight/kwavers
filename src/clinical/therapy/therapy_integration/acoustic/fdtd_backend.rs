//! FDTD Backend Adapter for Clinical Acoustic Solver
//!
//! This module provides an adapter that wraps the existing `FdtdSolver` to
//! implement the `AcousticSolverBackend` trait, enabling it to be used by
//! the clinical therapy acoustic solver.
//!
//! # Design Pattern
//!
//! This is an **Adapter Pattern** implementation that bridges the gap between
//! the generic `FdtdSolver` interface and the specialized `AcousticSolverBackend`
//! trait required by clinical applications.
//!
//! # FDTD Method
//!
//! The Finite-Difference Time-Domain (FDTD) method solves the first-order
//! acoustic wave equations:
//!
//! ```text
//! ∂v/∂t = -(1/ρ₀)∇p        (momentum conservation)
//! ∂p/∂t = -ρ₀c₀²∇·v        (mass conservation)
//! ```
//!
//! using central finite differences in space and forward Euler (or higher-order)
//! time integration.
//!
//! # Advantages
//!
//! - **Robust**: Handles heterogeneous media with sharp discontinuities
//! - **Straightforward**: Direct discretization of physical equations
//! - **Memory Efficient**: Only nearest-neighbor stencils required
//! - **Parallelizable**: GPU acceleration available via feature flag
//!
//! # Limitations
//!
//! - **Dispersion**: Numerical dispersion at high frequencies (requires 4-10 PPW)
//! - **CFL Constraint**: Time step limited by stability condition
//! - **Grid Resolution**: More points needed than spectral methods
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use crate::clinical::therapy::therapy_integration::acoustic::fdtd_backend::FdtdBackend;
//! use crate::clinical::therapy::therapy_integration::acoustic::backend::AcousticSolverBackend;
//!
//! let backend = FdtdBackend::new(&grid, medium, spatial_order)?;
//! let mut solver: Box<dyn AcousticSolverBackend> = Box::new(backend);
//!
//! for _ in 0..1000 {
//!     solver.step()?;
//! }
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::{GridSource, Source};
use crate::physics::mechanics::acoustic_wave::SpatialOrder;
use crate::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use ndarray::Array3;
use std::sync::Arc;

use super::backend::AcousticSolverBackend;

/// FDTD solver backend adapter
///
/// Wraps the existing `FdtdSolver` to provide the `AcousticSolverBackend`
/// interface required by clinical therapy applications.
///
/// # Fields
///
/// - `solver`: The underlying FDTD solver performing wave propagation
/// - `current_time`: Accumulated simulation time (steps * dt)
/// - `grid_dims`: Cached grid dimensions for efficient access
///
/// # Stability
///
/// The backend automatically computes a stable time step during initialization
/// based on the CFL condition:
/// ```text
/// dt = (CFL_factor) * dx_min / c_max
/// ```
/// where `CFL_factor = 0.5` (conservative, less than 1/√3 ≈ 0.577).
#[derive(Debug)]
pub struct FdtdBackend {
    /// Underlying FDTD solver
    solver: FdtdSolver,
    /// Current simulation time (s)
    current_time: f64,
    /// Cached grid dimensions (nx, ny, nz)
    grid_dims: (usize, usize, usize),
}

impl FdtdBackend {
    /// Create new FDTD backend
    ///
    /// Initializes an FDTD solver with appropriate configuration for clinical
    /// therapy applications. The solver is configured with:
    ///
    /// - Automatic stable time step computation (CFL-based)
    /// - PML absorbing boundaries (20 grid points, alpha=2.0)
    /// - Specified spatial accuracy order (2nd, 4th, or 6th)
    /// - Empty initial source (sources added dynamically)
    ///
    /// # Arguments
    ///
    /// - `grid`: Computational grid defining spatial domain
    /// - `medium`: Acoustic medium properties (density, sound speed, absorption)
    /// - `spatial_order`: Finite difference accuracy order (2, 4, or 6)
    ///
    /// # Returns
    ///
    /// New FDTD backend ready for time stepping.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid spatial order (must be 2, 4, or 6)
    /// - Grid too small for PML boundaries
    /// - Medium properties invalid (zero density, negative sound speed)
    /// - Time step computation fails
    ///
    /// # Time Step Selection
    ///
    /// The time step is computed as:
    /// ```text
    /// dt = 0.5 * dx_min / c_max
    /// ```
    /// This ensures the CFL number is ≤ 0.5, well below the stability limit
    /// of 1/√3 ≈ 0.577 for 3D.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use crate::physics::mechanics::acoustic_wave::SpatialOrder;
    ///
    /// // Create backend with 2nd-order accuracy
    /// let backend = FdtdBackend::new(&grid, medium, SpatialOrder::Second)?;
    ///
    /// // Create backend with 4th-order accuracy (less dispersion)
    /// let backend = FdtdBackend::new(&grid, medium, SpatialOrder::Fourth)?;
    /// ```
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        spatial_order: SpatialOrder,
    ) -> KwaversResult<Self> {
        // Compute stable time step using CFL condition
        let c_max = Self::estimate_max_sound_speed(medium, grid)?;
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt = Self::compute_stable_timestep(dx_min, c_max);

        // Configure FDTD solver for clinical applications
        let spatial_order_value = match spatial_order {
            SpatialOrder::Second => 2,
            SpatialOrder::Fourth => 4,
            SpatialOrder::Sixth => 6,
        };

        let config = FdtdConfig {
            spatial_order: spatial_order_value,
            staggered_grid: true,
            cfl_factor: 0.5,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            nt: 1, // Single step mode (controlled externally)
            dt,
            sensor_mask: None, // No sensors (fields accessed directly)
        };

        // Create empty source (sources added dynamically)
        let source = GridSource::new_empty();

        // Initialize FDTD solver
        let solver = FdtdSolver::new(config, grid, medium, source)?;

        Ok(Self {
            solver,
            current_time: 0.0,
            grid_dims: (grid.nx, grid.ny, grid.nz),
        })
    }

    /// Compute stable time step for FDTD
    ///
    /// Uses the CFL (Courant-Friedrichs-Lewy) stability condition for 3D:
    /// ```text
    /// c·dt/dx ≤ 1/√3  =>  dt ≤ dx/(√3·c)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `dx`: Minimum grid spacing (m)
    /// - `c_max`: Maximum sound speed in medium (m/s)
    ///
    /// # Returns
    ///
    /// Conservative time step satisfying CFL condition with safety factor of 0.5.
    ///
    /// # Mathematical Derivation
    ///
    /// The CFL condition arises from von Neumann stability analysis of the
    /// discretized wave equation. For the FDTD scheme in 3D:
    /// ```text
    /// |amplification factor| ≤ 1  requires  CFL = c·dt·√(1/dx² + 1/dy² + 1/dz²) ≤ 1
    /// ```
    /// For cubic grids (dx=dy=dz): `CFL = c·dt·√3/dx ≤ 1`
    ///
    /// We use a safety factor of 0.5 to account for:
    /// - Source term stability
    /// - Boundary condition reflections
    /// - Nonlinear effects (if enabled)
    /// - Numerical round-off accumulation
    fn compute_stable_timestep(dx: f64, c_max: f64) -> f64 {
        const CFL_SAFETY_FACTOR: f64 = 0.5;
        const SQRT_3: f64 = 1.732050807568877;
        CFL_SAFETY_FACTOR * dx / (c_max * SQRT_3)
    }

    /// Estimate maximum sound speed in medium
    ///
    /// Samples the medium at several locations to estimate the maximum sound
    /// speed, which is needed for CFL time step computation.
    ///
    /// # Sampling Strategy
    ///
    /// For efficiency, we sample a sparse 8×8×8 grid rather than every point.
    /// This is sufficient because:
    /// - Sound speed varies smoothly in biological tissue
    /// - We apply a safety factor in CFL computation
    /// - Over-estimating c_max is conservative (smaller dt, more stable)
    ///
    /// # Arguments
    ///
    /// - `medium`: Medium to sample
    /// - `grid`: Grid defining spatial domain
    ///
    /// # Returns
    ///
    /// Maximum sound speed found (m/s), or error if all samples invalid.
    fn estimate_max_sound_speed(medium: &dyn Medium, grid: &Grid) -> KwaversResult<f64> {
        const SAMPLE_POINTS: usize = 8;
        let mut c_max = 0.0;

        let di = (grid.nx / SAMPLE_POINTS).max(1);
        let dj = (grid.ny / SAMPLE_POINTS).max(1);
        let dk = (grid.nz / SAMPLE_POINTS).max(1);

        for k in (0..grid.nz).step_by(dk.max(1)) {
            for j in (0..grid.ny).step_by(dj.max(1)) {
                for i in (0..grid.nx).step_by(di.max(1)) {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                    if c > c_max {
                        c_max = c;
                    }
                }
            }
        }

        if c_max <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Maximum sound speed must be positive".into(),
            ));
        }

        Ok(c_max)
    }
}

impl AcousticSolverBackend for FdtdBackend {
    fn step(&mut self) -> KwaversResult<()> {
        // Advance FDTD solver by one time step
        self.solver.step_forward()?;

        // Update accumulated time
        self.current_time += self.solver.config.dt;

        Ok(())
    }

    fn get_pressure_field(&self) -> &Array3<f64> {
        &self.solver.fields.p
    }

    fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (
            &self.solver.fields.ux,
            &self.solver.fields.uy,
            &self.solver.fields.uz,
        )
    }

    fn get_intensity_field(&self) -> KwaversResult<Array3<f64>> {
        // Compute intensity using plane wave approximation: I = p²/(ρc)
        let p = &self.solver.fields.p;
        let rho = &self.solver.materials.rho0;
        let c = &self.solver.materials.c0;

        // Acoustic impedance Z = ρc
        let impedance = rho * c;

        // Intensity: I = p²/Z
        let intensity = p.mapv(|p_val| p_val * p_val) / &impedance;

        Ok(intensity)
    }

    fn get_dt(&self) -> f64 {
        self.solver.config.dt
    }

    fn add_source(&mut self, _source: Arc<dyn Source>) -> KwaversResult<()> {
        // Note: FdtdSolver does not expose a public API for adding dynamic sources
        // after construction. This is a known limitation that should be addressed
        // by adding a public method to FdtdSolver.
        // For now, we return an error indicating this is not yet supported.
        Err(KwaversError::NotImplemented(
            "Dynamic source addition not yet supported by FDTD backend. \
             Sources must be configured at solver creation time."
                .into(),
        ))
    }

    fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn get_grid_dimensions(&self) -> (usize, usize, usize) {
        self.grid_dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;

    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 0.0005, 0.0005, 0.0005).expect("Failed to create test grid")
    }

    fn create_test_medium(grid: &Grid) -> HomogeneousMedium {
        HomogeneousMedium::new(
            1000.0, // Water density (kg/m³)
            1500.0, // Water sound speed (m/s)
            0.0,    // Optical absorption mu_a
            0.0,    // Optical scattering mu_s_prime
            grid,
        )
    }

    #[test]
    fn test_fdtd_backend_creation() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second);
        assert!(backend.is_ok(), "Backend creation failed");

        let backend = backend.unwrap();
        assert_eq!(backend.get_grid_dimensions(), (32, 32, 32));
        assert!(backend.get_dt() > 0.0, "Time step must be positive");
        assert_eq!(backend.get_current_time(), 0.0);
    }

    #[test]
    fn test_fdtd_backend_cfl_condition() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second).unwrap();

        let dt = backend.get_dt();
        let dx = grid.min_spacing();
        let c_max = 1500.0; // Water sound speed

        // Verify CFL condition: c·dt/dx ≤ 1/√3
        let cfl = c_max * dt / dx;
        let cfl_limit = 1.0 / 3.0_f64.sqrt();

        assert!(
            cfl < cfl_limit,
            "CFL condition violated: {} >= {}",
            cfl,
            cfl_limit
        );

        // Verify conservative safety factor (should be ~0.5)
        assert!(cfl < 0.6, "CFL factor should be conservative: {}", cfl);
    }

    #[test]
    fn test_fdtd_backend_time_stepping() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let mut backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second).unwrap();

        let dt = backend.get_dt();
        assert_eq!(backend.get_current_time(), 0.0);

        // Step once
        backend.step().expect("First step failed");
        assert!((backend.get_current_time() - dt).abs() < 1e-15);

        // Step again
        backend.step().expect("Second step failed");
        assert!((backend.get_current_time() - 2.0 * dt).abs() < 1e-14);
    }

    #[test]
    fn test_fdtd_backend_field_access() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second).unwrap();

        // Test pressure field
        let p = backend.get_pressure_field();
        assert_eq!(p.shape(), &[32, 32, 32]);

        // Test velocity fields
        let (vx, vy, vz) = backend.get_velocity_fields();
        assert_eq!(vx.shape(), &[32, 32, 32]);
        assert_eq!(vy.shape(), &[32, 32, 32]);
        assert_eq!(vz.shape(), &[32, 32, 32]);
    }

    #[test]
    fn test_fdtd_backend_intensity_computation() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second).unwrap();

        // Compute intensity (should be zero for zero pressure)
        let intensity = backend
            .get_intensity_field()
            .expect("Intensity computation failed");
        assert_eq!(intensity.shape(), &[32, 32, 32]);

        // All values should be zero initially
        let max_intensity = intensity.iter().cloned().fold(0.0_f64, f64::max);
        assert_eq!(max_intensity, 0.0);
    }

    #[test]
    fn test_fdtd_backend_as_trait_object() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let backend = FdtdBackend::new(&grid, &medium, SpatialOrder::Second).unwrap();
        let mut solver: Box<dyn AcousticSolverBackend> = Box::new(backend);

        // Verify trait object works
        assert_eq!(solver.get_grid_dimensions(), (32, 32, 32));
        assert!(solver.get_dt() > 0.0);

        solver.step().expect("Step failed");
        assert!(solver.get_current_time() > 0.0);
    }

    #[test]
    fn test_stable_timestep_computation() {
        let dx = 0.0005; // 0.5 mm
        let c = 1500.0; // Water sound speed

        let dt = FdtdBackend::compute_stable_timestep(dx, c);

        // Verify CFL condition
        let cfl = c * dt / dx;
        assert!(cfl < 1.0 / 3.0_f64.sqrt());

        // Typical value check (should be ~1.9e-7 for these parameters)
        assert!(dt > 1e-8 && dt < 1e-6, "Unexpected time step: {}", dt);
    }

    #[test]
    fn test_max_sound_speed_estimation() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        let c_max = FdtdBackend::estimate_max_sound_speed(&medium, &grid).unwrap();

        // Should match homogeneous medium sound speed
        assert!((c_max - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_order_variants() {
        let grid = create_test_grid();
        let medium = create_test_medium(&grid);

        // Test all spatial orders
        let orders = [
            SpatialOrder::Second,
            SpatialOrder::Fourth,
            SpatialOrder::Sixth,
        ];

        for order in &orders {
            let backend = FdtdBackend::new(&grid, &medium, *order);
            assert!(
                backend.is_ok(),
                "Failed to create backend with order {:?}",
                order
            );
        }
    }
}
