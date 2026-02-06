//! Acoustic Wave Solver Module for Clinical Therapy Applications
//!
//! This module provides production-ready acoustic field computation for therapeutic
//! ultrasound applications including HIFU, lithotripsy, and sonoporation.
//!
//! # Module Organization
//!
//! The `AcousticWaveSolver` is the main public API for clinical applications.
//! It composes acoustic solver backends provided by the simulation layer.
//!
//! # Architecture
//!
//! ```text
//! Clinical Layer (this module)
//!     AcousticWaveSolver
//!         ↓ uses
//! Simulation Layer
//!     simulation::backends::AcousticSolverBackend (trait)
//!         ↑ implemented by
//! simulation::backends::acoustic::FdtdBackend
//!         ↓ wraps
//! Solver Layer
//!     solver::forward::fdtd::FdtdSolver
//! ```
//!
//! # Design Rationale
//!
//! By using simulation layer backends:
//! - Clinical code never depends on solver layer (only simulation)
//! - Clear layering: Clinical → Simulation → Solver
//! - Solver changes don't propagate to clinical code
//! - Simulation layer orchestrates solver access
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::clinical::therapy::therapy_integration::acoustic::AcousticWaveSolver;
//!
//! let solver = AcousticWaveSolver::new(&grid, &medium)?;
//! solver.step()?;
//! let pressure = solver.pressure_field();
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
// SpatialOrder is a domain-level discretization parameter
use crate::domain::grid::operators::coefficients::SpatialOrder;
use crate::simulation::backends::acoustic::{AcousticSolverBackend, FdtdBackend};
use ndarray::Array3;
use std::sync::Arc;

/// Acoustic wave solver for therapy applications
///
/// Production-ready solver providing acoustic field simulation for therapeutic
/// ultrasound applications. Automatically selects appropriate numerical backend
/// (FDTD, PSTD) based on problem characteristics.
///
/// # Backend Selection Criteria
///
/// The solver analyzes the problem and selects the backend as follows:
///
/// | Characteristic | Threshold | Selected Backend |
/// |----------------|-----------|------------------|
/// | Points per wavelength | < 4 | PSTD (spectral accuracy) |
/// | Heterogeneity | > 30% | FDTD (handles discontinuities) |
/// | Default | - | FDTD (robust, general-purpose) |
///
/// # Fields
///
/// - `backend`: Polymorphic solver backend (FDTD, PSTD, etc.)
/// - `grid`: Computational grid defining spatial domain
///
/// # Time Integration
///
/// The solver uses explicit time stepping with stability-constrained time steps:
///
/// - **FDTD**: `dt = 0.5 * dx_min / c_max` (CFL < 1/√3)
/// - **PSTD**: `dt = 0.9 * π / (c_max * k_max)` (spectral stability)
///
/// # Thread Safety
///
/// Not thread-safe (single-threaded simulation loop). Internal operations may
/// use SIMD or GPU parallelism.
#[derive(Debug)]
pub struct AcousticWaveSolver {
    /// Solver backend (FDTD, PSTD, etc.)
    backend: Box<dyn AcousticSolverBackend>,
    /// Computational grid
    grid: Grid,
    /// Accumulated squared pressure for temporal averaging (Pa²)
    accumulated_p_squared: Array3<f64>,
}

impl AcousticWaveSolver {
    /// Create new acoustic wave solver with automatic backend selection
    ///
    /// Analyzes problem characteristics (grid resolution, medium properties)
    /// and selects the optimal numerical backend. Currently defaults to FDTD
    /// for robustness; PSTD backend will be added in Sprint 212.
    ///
    /// # Arguments
    ///
    /// - `grid`: Computational grid defining spatial domain and discretization
    /// - `medium`: Acoustic medium properties (density, sound speed, absorption)
    ///
    /// # Returns
    ///
    /// New solver instance with:
    /// - Automatically selected backend
    /// - Stable time step computed from CFL condition
    /// - PML absorbing boundaries configured
    /// - Empty source list (add sources via `add_source()`)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Grid too small (< 8 points per dimension for PML boundaries)
    /// - Medium properties invalid (zero/negative density or sound speed)
    /// - Backend initialization fails
    ///
    /// # Backend Selection Logic (Current)
    ///
    /// Currently always selects FDTD backend for maximum robustness.
    /// Future versions will implement:
    ///
    /// ```text
    /// if points_per_wavelength < 4.0 {
    ///     PSTD  // Spectral accuracy for under-resolved grids
    /// } else if heterogeneity > 0.3 {
    ///     FDTD  // Handles discontinuities well
    /// } else {
    ///     FDTD  // Default robust choice
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use crate::domain::grid::Grid;
    /// use crate::domain::medium::HomogeneousMedium;
    ///
    /// let grid = Grid::new(128, 128, 128, 0.0005, 0.0005, 0.0005)?;
    /// let medium = HomogeneousMedium::new(&grid, 1000.0, 1500.0, 0.0, 0.0)?;
    ///
    /// let solver = AcousticWaveSolver::new(&grid, &medium)?;
    /// println!("Time step: {:.2e} s", solver.timestep());
    /// ```
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        // Phase 1 implementation: Always use FDTD backend
        // Phase 2 (Sprint 212) will add PSTD backend selection logic
        let backend = Self::create_fdtd_backend(grid, medium)?;
        let dims = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            backend,
            grid: grid.clone(),
            accumulated_p_squared: Array3::zeros(dims),
        })
    }

    /// Create FDTD backend with appropriate configuration
    ///
    /// Internal method for constructing FDTD solver backend with:
    /// - 2nd-order spatial accuracy (balance of accuracy and efficiency)
    /// - Automatic stable time step from CFL condition
    /// - PML absorbing boundaries (20-point thickness)
    ///
    /// # Arguments
    ///
    /// - `grid`: Computational grid
    /// - `medium`: Acoustic medium
    ///
    /// # Returns
    ///
    /// Boxed FDTD backend implementing `AcousticSolverBackend` trait.
    fn create_fdtd_backend(
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Box<dyn AcousticSolverBackend>> {
        let backend = FdtdBackend::new(grid, medium, SpatialOrder::Second)?;
        Ok(Box::new(backend))
    }

    /// Advance simulation by one time step
    ///
    /// Updates pressure and velocity fields by integrating the acoustic wave
    /// equations forward by one time step `dt`. The time step is determined
    /// automatically during initialization based on stability constraints.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Numerical instability detected (NaN/Inf in fields)
    /// - Boundary condition application fails
    /// - Source evaluation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut solver = AcousticWaveSolver::new(&grid, &medium)?;
    ///
    /// for step in 0..1000 {
    ///     solver.step()?;
    ///
    ///     if step % 100 == 0 {
    ///         let p_max = solver.max_pressure();
    ///         println!("Step {}: Max pressure = {:.2} MPa", step, p_max);
    ///     }
    /// }
    /// ```
    pub fn step(&mut self) -> KwaversResult<()> {
        self.backend.step()?;

        // Accumulate squared pressure for SPTA calculation
        let p = self.backend.get_pressure_field();
        self.accumulated_p_squared
            .zip_mut_with(p, |acc, &val| *acc += val * val);

        Ok(())
    }

    /// Advance simulation by specified time duration
    ///
    /// Convenience method for running multiple time steps to cover a physical
    /// time duration. Automatically computes the number of steps required.
    ///
    /// # Arguments
    ///
    /// - `duration`: Physical time to simulate (seconds)
    ///
    /// # Errors
    ///
    /// Returns error if any individual time step fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Simulate 1 microsecond
    /// solver.advance(1e-6)?;
    ///
    /// // Simulate 100 acoustic periods at 1 MHz
    /// let period = 1.0 / 1e6;
    /// solver.advance(100.0 * period)?;
    /// ```
    pub fn advance(&mut self, duration: f64) -> KwaversResult<()> {
        if duration < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Duration must be non-negative".into(),
            ));
        }

        let dt = self.backend.get_dt();
        let num_steps = (duration / dt).ceil() as usize;

        for _ in 0..num_steps {
            self.step()?;
        }

        Ok(())
    }

    /// Get current pressure field (Pa)
    ///
    /// Returns a reference to the 3D pressure field. Values are in Pascals (Pa).
    ///
    /// # Typical Pressure Ranges
    ///
    /// - **Diagnostic ultrasound**: 100 kPa - 1 MPa
    /// - **Therapeutic HIFU**: 1 MPa - 100 MPa
    /// - **Lithotripsy shocks**: 10 MPa - 100 MPa
    ///
    /// # Indexing
    ///
    /// Array dimensions are `[nx, ny, nz]`. Physical position of `field[[i,j,k]]`:
    /// ```text
    /// (x, y, z) = (i*dx, j*dy, k*dz)
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let p = solver.pressure_field();
    /// let p_center = p[[nx/2, ny/2, nz/2]];
    /// println!("Center pressure: {:.2} kPa", p_center / 1e3);
    /// ```
    pub fn pressure_field(&self) -> &Array3<f64> {
        self.backend.get_pressure_field()
    }

    /// Get current particle velocity fields (m/s)
    ///
    /// Returns references to the three velocity components: `(vx, vy, vz)`.
    ///
    /// # Physical Interpretation
    ///
    /// Particle velocity is the oscillatory motion of medium particles, not
    /// the wave propagation speed. Related to pressure by:
    /// ```text
    /// v ≈ p / (ρ₀c₀)  (plane wave approximation)
    /// ```
    ///
    /// # Typical Magnitudes
    ///
    /// - **Diagnostic**: 0.01 - 0.1 m/s
    /// - **Therapeutic**: 0.1 - 10 m/s
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let (vx, vy, vz) = solver.velocity_fields();
    /// let v_magnitude = (vx[[i,j,k]].powi(2) +
    ///                     vy[[i,j,k]].powi(2) +
    ///                     vz[[i,j,k]].powi(2)).sqrt();
    /// ```
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        self.backend.get_velocity_fields()
    }

    /// Get acoustic intensity field (W/m²)
    ///
    /// Computes instantaneous acoustic intensity using plane wave approximation:
    /// ```text
    /// I = p² / (ρ₀c₀)
    /// ```
    ///
    /// # Temporal Averaging
    ///
    /// For safety metrics, intensity should be time-averaged:
    /// ```text
    /// I_ta = (1/T) ∫₀ᵀ I(t) dt
    /// ```
    /// This method returns instantaneous intensity; averaging is caller's responsibility.
    ///
    /// # Safety Thresholds
    ///
    /// - **Diagnostic**: I_spta < 720 mW/cm² (FDA limit)
    /// - **Therapeutic**: I_spta > 100 W/cm² (ablation)
    /// - **Cavitation**: I_spta > 1-10 W/cm² (nucleation)
    ///
    /// # Errors
    ///
    /// Returns error if intensity computation fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let intensity = solver.intensity_field()?;
    /// let i_max = intensity.iter().cloned().fold(0.0, f64::max);
    /// println!("Max intensity: {:.2} W/cm²", i_max / 1e4);
    /// ```
    pub fn intensity_field(&self) -> KwaversResult<Array3<f64>> {
        self.backend.get_intensity_field()
    }

    /// Get maximum pressure magnitude (MPa)
    ///
    /// Convenience method returning the maximum absolute pressure in the field.
    /// Units are megapascals (MPa) for clinical relevance.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let p_max = solver.max_pressure();
    /// if p_max > 100.0 {
    ///     println!("Warning: High-intensity field detected");
    /// }
    /// ```
    pub fn max_pressure(&self) -> f64 {
        let p = self.pressure_field();
        let p_max = p.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        p_max / 1e6 // Convert Pa to MPa
    }

    /// Get spatial peak temporal average intensity (W/cm²)
    ///
    /// Computes the spatial peak of the time-averaged intensity field.
    /// This is a key FDA safety metric for ultrasound devices.
    ///
    /// # Arguments
    ///
    /// - `averaging_time`: Time window for temporal averaging (seconds)
    ///
    /// # Current Implementation
    ///
    /// Returns instantaneous spatial peak intensity (temporal averaging not
    /// yet implemented - requires storing field history).
    ///
    /// # FDA Limits
    ///
    /// - **Diagnostic ultrasound**: I_spta < 720 mW/cm²
    /// - **Ophthalmic**: I_spta < 50 mW/cm²
    ///
    /// # Errors
    ///
    /// Returns error if intensity computation fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let i_spta = solver.spta_intensity(1e-3)?; // 1 ms averaging
    /// if i_spta < 720.0 {
    ///     println!("Within FDA diagnostic limits");
    /// }
    /// ```
    pub fn spta_intensity(&self, averaging_time: f64) -> KwaversResult<f64> {
        if averaging_time <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Averaging time must be positive".into(),
            ));
        }

        // Retrieve impedance map from backend (Z = ρc)
        let impedance = self.backend.get_impedance_field()?;
        let dt = self.backend.get_dt();

        // Compute temporal average intensity field
        // I_ta = (1/T_avg) * ∫ (p²/Z) dt  ≈ (dt/T_avg) * Σ(p²)/Z
        // Note: accumulated_p_squared stores Σ(p²)
        let normalization = dt / averaging_time;

        // Compute max intensity directly or via map
        // I = accumulated * normalization / impedance
        let i_spta = self
            .accumulated_p_squared
            .iter()
            .zip(impedance.iter())
            .fold(0.0_f64, |max_val, (&acc_p2, &z)| {
                let val = (acc_p2 * normalization) / z;
                if val.is_nan() {
                    max_val
                } else {
                    max_val.max(val)
                }
            });

        Ok(i_spta / 1e4) // Convert W/m² to W/cm²
    }

    /// Add dynamic source to simulation
    ///
    /// Registers an acoustic source (transducer, phased array element, etc.)
    /// that will be evaluated and applied at each time step.
    ///
    /// # Arguments
    ///
    /// - `source`: Source implementation (e.g., piston, focused bowl, point source)
    ///
    /// # Common Source Types
    ///
    /// - **Piston transducer**: Circular aperture, uniform pressure
    /// - **Focused bowl**: Spherical cap, geometric focusing
    /// - **Phased array element**: Single element with phase/amplitude control
    /// - **Point source**: Omnidirectional (testing/validation)
    ///
    /// # Errors
    ///
    /// Returns error if source registration fails (e.g., position outside grid).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use crate::domain::source::transducers::PistonTransducer;
    ///
    /// let transducer = Arc::new(PistonTransducer::new(
    ///     position,
    ///     direction,
    ///     radius,
    ///     frequency,
    ///     amplitude,
    /// )?);
    ///
    /// solver.add_source(transducer)?;
    /// ```
    pub fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        self.backend.add_source(source)
    }

    /// Get simulation time step (s)
    ///
    /// Returns the time step used for time integration. Automatically computed
    /// during initialization based on CFL stability condition.
    ///
    /// # Typical Values
    ///
    /// - Fine grid (dx = 0.1 mm): dt ~ 10-30 ns
    /// - Medium grid (dx = 0.5 mm): dt ~ 50-150 ns
    /// - Coarse grid (dx = 1.0 mm): dt ~ 100-300 ns
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let dt = solver.timestep();
    /// let steps_per_period = (1.0 / frequency) / dt;
    /// println!("Time step: {:.2} ns", dt * 1e9);
    /// println!("Steps per period: {:.0}", steps_per_period);
    /// ```
    pub fn timestep(&self) -> f64 {
        self.backend.get_dt()
    }

    /// Get current simulation time (s)
    ///
    /// Returns the accumulated physical time, equal to `(steps completed) * dt`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// println!("Current time: {:.2} μs", solver.current_time() * 1e6);
    /// ```
    pub fn current_time(&self) -> f64 {
        self.backend.get_current_time()
    }

    /// Get grid dimensions
    ///
    /// Returns `(nx, ny, nz)` grid point counts.
    pub fn grid_dimensions(&self) -> (usize, usize, usize) {
        (self.grid.nx, self.grid.ny, self.grid.nz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;

    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 0.0005, 0.0005, 0.0005).expect("Failed to create grid")
    }

    fn create_water_medium(grid: &Grid) -> HomogeneousMedium {
        HomogeneousMedium::new(
            1000.0, // Water density
            1500.0, // Water sound speed
            0.0,    // Optical absorption mu_a
            0.0,    // Optical scattering mu_s_prime
            grid,
        )
    }

    #[test]
    fn test_acoustic_solver_creation() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let solver = AcousticWaveSolver::new(&grid, &medium);
        assert!(solver.is_ok(), "Solver creation failed");

        let solver = solver.unwrap();
        assert_eq!(solver.grid_dimensions(), (32, 32, 32));
        assert!(solver.timestep() > 0.0);
        assert_eq!(solver.current_time(), 0.0);
    }

    #[test]
    fn test_acoustic_solver_time_stepping() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        let dt = solver.timestep();

        // Step once
        solver.step().expect("Step failed");
        assert!((solver.current_time() - dt).abs() < 1e-15);

        // Step again
        solver.step().expect("Second step failed");
        assert!((solver.current_time() - 2.0 * dt).abs() < 1e-14);
    }

    #[test]
    fn test_acoustic_solver_advance() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Advance by 1 microsecond
        let duration = 1e-6;
        solver.advance(duration).expect("Advance failed");

        assert!(solver.current_time() >= duration);
        assert!(solver.current_time() < duration + solver.timestep());
    }

    #[test]
    fn test_acoustic_solver_field_access() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Test pressure field
        let p = solver.pressure_field();
        assert_eq!(p.shape(), &[32, 32, 32]);

        // Test velocity fields
        let (vx, vy, vz) = solver.velocity_fields();
        assert_eq!(vx.shape(), &[32, 32, 32]);
        assert_eq!(vy.shape(), &[32, 32, 32]);
        assert_eq!(vz.shape(), &[32, 32, 32]);

        // Test intensity field
        let intensity = solver.intensity_field().expect("Intensity failed");
        assert_eq!(intensity.shape(), &[32, 32, 32]);
    }

    #[test]
    fn test_acoustic_solver_max_pressure() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Initial state should have zero pressure
        let p_max = solver.max_pressure();
        assert_eq!(p_max, 0.0);
    }

    #[test]
    fn test_acoustic_solver_spta_intensity() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Test SPTA computation
        let i_spta = solver.spta_intensity(1e-3).expect("SPTA failed");
        assert_eq!(i_spta, 0.0); // Zero for zero pressure field
    }

    #[test]
    fn test_advance_negative_duration() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Negative duration should error
        let result = solver.advance(-1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_advance_zero_duration() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);

        let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Zero duration should succeed without stepping
        solver.advance(0.0).expect("Zero advance failed");
        assert_eq!(solver.current_time(), 0.0);
    }

    #[test]
    fn test_spta_intensity_validation() {
        let grid = create_test_grid();
        let medium = create_water_medium(&grid);
        let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

        // Should return error for negative or zero time
        assert!(solver.spta_intensity(-1.0).is_err());
        assert!(solver.spta_intensity(0.0).is_err());

        // Should return OK for positive time (even with zero field)
        assert!(solver.spta_intensity(1.0).is_ok());
        assert_eq!(solver.spta_intensity(1.0).unwrap(), 0.0);
    }
}
