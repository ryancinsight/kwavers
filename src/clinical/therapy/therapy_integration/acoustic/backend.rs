//! Backend Abstraction for Acoustic Wave Solvers
//!
//! This module defines the trait interface for acoustic solver backends,
//! enabling the clinical therapy acoustic solver to delegate to different
//! numerical methods (FDTD, PSTD, etc.) while maintaining a unified API.
//!
//! # Design Pattern
//!
//! The `AcousticSolverBackend` trait follows the **Strategy Pattern**, allowing
//! runtime selection of the appropriate numerical solver based on problem
//! characteristics:
//!
//! - **FDTD** (Finite-Difference Time-Domain): Robust for heterogeneous media,
//!   handles discontinuities well, straightforward implementation
//! - **PSTD** (Pseudospectral Time-Domain): Spectral accuracy for smooth media,
//!   4-8x fewer grid points, efficient for homogeneous cases
//! - **Nonlinear** (Westervelt/KZK): High-intensity therapeutic ultrasound,
//!   shock formation, harmonic generation
//!
//! # Backend Selection Criteria
//!
//! | Criterion | FDTD | PSTD |
//! |-----------|------|------|
//! | Heterogeneity | High (>30%) | Low (<30%) |
//! | Points per wavelength | 4-10 | 2-4 |
//! | Discontinuities | Sharp interfaces | Smooth variations |
//! | Efficiency | O(N log N) | O(N log N) |
//!
//! # Trait Design Principles
//!
//! - **Minimal Interface**: Only essential operations for therapy applications
//! - **Zero-Cost Abstraction**: Trait object overhead acceptable for long-running simulations
//! - **Thread Safety**: Not required (single-threaded simulation loop)
//! - **Error Handling**: All fallible operations return `KwaversResult`
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use crate::clinical::therapy::therapy_integration::acoustic::backend::AcousticSolverBackend;
//!
//! fn run_simulation(mut backend: Box<dyn AcousticSolverBackend>) -> KwaversResult<()> {
//!     // Time stepping loop
//!     for _ in 0..1000 {
//!         backend.step()?;
//!
//!         // Monitor pressure field
//!         let p = backend.get_pressure_field();
//!         let p_max = p.iter().cloned().fold(0.0, f64::max);
//!         println!("Max pressure: {:.2} kPa", p_max / 1e3);
//!     }
//!
//!     // Compute intensity for safety validation
//!     let intensity = backend.get_intensity_field()?;
//!     Ok(())
//! }
//! ```

use crate::core::error::KwaversResult;
use crate::domain::source::Source;
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// Backend trait for acoustic wave solvers
///
/// Defines the interface that all acoustic solver backends must implement
/// to be usable by the clinical therapy acoustic solver. This trait enables
/// polymorphic solver selection based on problem characteristics.
///
/// # Implementors
///
/// - `FdtdBackend`: Adapts `crate::solver::forward::fdtd::FdtdSolver`
/// - `PstdBackend`: Adapts `crate::solver::forward::pstd::PSTDSolver`
/// - `NonlinearBackend`: Future nonlinear acoustics support
///
/// # Mathematical Foundations
///
/// All backends must solve the acoustic wave equations in either first-order
/// (pressure-velocity) or second-order (pressure only) form:
///
/// **First-Order System** (FDTD):
/// ```text
/// ∂v/∂t = -(1/ρ₀)∇p        (momentum conservation)
/// ∂p/∂t = -ρ₀c₀²∇·v        (mass conservation)
/// ```
///
/// **Second-Order Form** (PSTD):
/// ```text
/// ∇²p - (1/c₀²)∂²p/∂t² = 0
/// ```
///
/// # Stability Requirements
///
/// Implementations must enforce stability constraints:
///
/// - **FDTD CFL**: `c_max·Δt/Δx ≤ 1/√3` (3D)
/// - **PSTD**: `c_max·Δt·k_max ≤ π` where `k_max = π/Δx`
///
/// Violating these constraints will cause exponential growth of numerical errors.
///
/// # Thread Safety
///
/// This trait is **not** required to be `Send` or `Sync`. Simulations run
/// in a single-threaded loop with potential internal parallelism via SIMD
/// or GPU acceleration.
pub trait AcousticSolverBackend: Debug {
    /// Advance simulation by one time step
    ///
    /// Updates the pressure and velocity fields by integrating the acoustic
    /// wave equations forward by the backend's internal time step `dt`.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Numerical instability detected (NaN/Inf in fields)
    /// - Boundary condition application fails
    /// - Source evaluation fails
    /// - Memory allocation fails
    ///
    /// # Implementation Notes
    ///
    /// - Must update pressure field accessible via `get_pressure_field()`
    /// - Must update velocity fields accessible via `get_velocity_fields()`
    /// - Should apply boundary conditions (PML, rigid, etc.)
    /// - Should evaluate and apply sources for current time step
    ///
    /// # Stability
    ///
    /// Implementations must ensure numerical stability by enforcing appropriate
    /// CFL conditions during initialization. If stability is violated, this
    /// method should detect the instability and return an error rather than
    /// producing invalid results.
    fn step(&mut self) -> KwaversResult<()>;

    /// Get current pressure field (Pa)
    ///
    /// Returns a reference to the 3D pressure field array. Pressure values
    /// are in Pascals (Pa), with:
    ///
    /// - Diagnostic ultrasound: p ~ 100 kPa - 1 MPa
    /// - Therapeutic HIFU: p ~ 1 MPa - 100 MPa
    /// - Lithotripsy: p ~ 10 MPa - 100 MPa (shock waves)
    ///
    /// # Indexing Convention
    ///
    /// The array dimensions are `[nx, ny, nz]` corresponding to spatial indices
    /// `(i, j, k)`. The physical position of `field[[i, j, k]]` is:
    /// ```text
    /// x = i * dx
    /// y = j * dy
    /// z = k * dz
    /// ```
    /// where `(dx, dy, dz)` are the grid spacings.
    ///
    /// # Return Value
    ///
    /// Reference to the internal pressure field. Lifetime is tied to the backend.
    /// The field is valid until the next call to `step()`.
    fn get_pressure_field(&self) -> &Array3<f64>;

    /// Get current particle velocity fields (m/s)
    ///
    /// Returns references to the three components of the particle velocity
    /// vector field: `(vx, vy, vz)`.
    ///
    /// # Physical Interpretation
    ///
    /// Particle velocity represents the oscillatory motion of the medium particles
    /// (not the wave propagation speed). Related to pressure by:
    /// ```text
    /// v = -(1/(ρ₀c₀))∫p dt    (plane wave approximation)
    /// ```
    ///
    /// Typical magnitudes:
    /// - Diagnostic: v ~ 0.01 - 0.1 m/s
    /// - Therapeutic: v ~ 0.1 - 10 m/s
    ///
    /// # Staggered Grids
    ///
    /// Some backends (FDTD) may use staggered grids where velocity components
    /// are defined at half-grid offsets from pressure. Implementations should
    /// return co-located (interpolated) values for API consistency.
    ///
    /// # Return Value
    ///
    /// Tuple of `(vx, vy, vz)` references. Lifetimes tied to backend.
    fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>);

    /// Get acoustic intensity field (W/m²)
    ///
    /// Computes the instantaneous acoustic intensity, which represents the
    /// energy flux (power per unit area) carried by the acoustic wave.
    ///
    /// # Calculation Methods
    ///
    /// **Instantaneous Intensity** (rigorous):
    /// ```text
    /// I = p · v    (pressure times particle velocity)
    /// ```
    ///
    /// **Plane Wave Approximation** (commonly used):
    /// ```text
    /// I = p² / (ρ₀c₀)
    /// ```
    /// This is accurate for plane waves and far-field conditions.
    ///
    /// # Temporal Averaging
    ///
    /// For safety metrics, intensity is typically time-averaged:
    /// ```text
    /// I_ta = (1/T) ∫₀ᵀ I(t) dt
    /// ```
    /// Implementations return instantaneous intensity; averaging is caller's responsibility.
    ///
    /// # Safety Thresholds
    ///
    /// - **Diagnostic ultrasound**: I_spta < 720 mW/cm² (FDA limit)
    /// - **Therapeutic HIFU**: I_spta > 100 W/cm² (ablation threshold)
    /// - **Cavitation**: I_spta > 1-10 W/cm² (nucleation threshold)
    ///
    /// # Errors
    ///
    /// Returns error if intensity computation fails (e.g., zero impedance).
    ///
    /// # Return Value
    ///
    /// New `Array3<f64>` with intensity in W/m². Convert to W/cm² by dividing by 10⁴.
    fn get_intensity_field(&self) -> KwaversResult<Array3<f64>>;

    /// Get simulation time step (s)
    ///
    /// Returns the time step `dt` used by the backend for time integration.
    /// This value is determined during backend initialization based on:
    ///
    /// - Grid spacing `(dx, dy, dz)`
    /// - Maximum sound speed `c_max` in the medium
    /// - Stability constraint (CFL condition)
    ///
    /// # Typical Values
    ///
    /// - Fine grid (dx = 0.1 mm): dt ~ 10-30 ns
    /// - Medium grid (dx = 0.5 mm): dt ~ 50-150 ns
    /// - Coarse grid (dx = 1.0 mm): dt ~ 100-300 ns
    ///
    /// # Stability
    ///
    /// The time step is chosen to satisfy:
    /// ```text
    /// dt ≤ (stability_factor) * dx / c_max
    /// ```
    /// where `stability_factor < 1/√3` for FDTD or `< π/k_max` for PSTD.
    fn get_dt(&self) -> f64;

    /// Add dynamic source to simulation
    ///
    /// Registers a new acoustic source (transducer, phased array element, etc.)
    /// that will be evaluated and applied at each time step.
    ///
    /// # Arguments
    ///
    /// - `source`: Source implementation (point source, piston, focused bowl, etc.)
    ///
    /// # Source Types
    ///
    /// Common sources for therapy applications:
    /// - **Piston transducer**: Circular aperture, uniform pressure
    /// - **Focused bowl**: Spherical cap, geometric focusing
    /// - **Phased array**: Multiple elements with delays for beam steering
    /// - **Point source**: Omnidirectional, testing/validation
    ///
    /// # Implementation Notes
    ///
    /// Sources should be evaluated once per time step and added to the appropriate
    /// field (pressure or velocity depending on source type). Multiple sources
    /// are superimposed linearly.
    ///
    /// # Errors
    ///
    /// Returns error if source registration fails (e.g., source position outside grid).
    fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()>;

    /// Get current simulation time (s)
    ///
    /// Returns the accumulated simulation time, which equals:
    /// ```text
    /// t = (number of steps completed) * dt
    /// ```
    ///
    /// # Usage
    ///
    /// Used for:
    /// - Time-dependent source evaluation
    /// - Progress reporting
    /// - Synchronization with other physics (thermal, cavitation)
    fn get_current_time(&self) -> f64;

    /// Get grid dimensions
    ///
    /// Returns the number of grid points in each dimension: `(nx, ny, nz)`.
    ///
    /// # Usage
    ///
    /// Required for:
    /// - Array allocation
    /// - Iteration bounds
    /// - Interfacing with external analysis tools
    fn get_grid_dimensions(&self) -> (usize, usize, usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock backend for testing trait interface
    #[derive(Debug)]
    struct MockBackend {
        nx: usize,
        ny: usize,
        nz: usize,
        dt: f64,
        time: f64,
        pressure: Array3<f64>,
        vx: Array3<f64>,
        vy: Array3<f64>,
        vz: Array3<f64>,
    }

    impl MockBackend {
        fn new(nx: usize, ny: usize, nz: usize, dt: f64) -> Self {
            Self {
                nx,
                ny,
                nz,
                dt,
                time: 0.0,
                pressure: Array3::zeros((nx, ny, nz)),
                vx: Array3::zeros((nx, ny, nz)),
                vy: Array3::zeros((nx, ny, nz)),
                vz: Array3::zeros((nx, ny, nz)),
            }
        }
    }

    impl AcousticSolverBackend for MockBackend {
        fn step(&mut self) -> KwaversResult<()> {
            self.time += self.dt;
            Ok(())
        }

        fn get_pressure_field(&self) -> &Array3<f64> {
            &self.pressure
        }

        fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
            (&self.vx, &self.vy, &self.vz)
        }

        fn get_intensity_field(&self) -> KwaversResult<Array3<f64>> {
            // I = p²/(ρc) with ρc = 1.5 MRayl for water
            let rho_c = 1.5e6;
            Ok(self.pressure.mapv(|p| p * p / rho_c))
        }

        fn get_dt(&self) -> f64 {
            self.dt
        }

        fn add_source(&mut self, _source: Arc<dyn Source>) -> KwaversResult<()> {
            Ok(())
        }

        fn get_current_time(&self) -> f64 {
            self.time
        }

        fn get_grid_dimensions(&self) -> (usize, usize, usize) {
            (self.nx, self.ny, self.nz)
        }
    }

    #[test]
    fn test_backend_trait_basic_operations() {
        let mut backend = MockBackend::new(10, 10, 10, 1e-7);

        // Test initial state
        assert_eq!(backend.get_current_time(), 0.0);
        assert_eq!(backend.get_dt(), 1e-7);
        assert_eq!(backend.get_grid_dimensions(), (10, 10, 10));

        // Test stepping
        backend.step().unwrap();
        assert_eq!(backend.get_current_time(), 1e-7);

        backend.step().unwrap();
        assert_eq!(backend.get_current_time(), 2e-7);
    }

    #[test]
    fn test_backend_field_access() {
        let backend = MockBackend::new(5, 5, 5, 1e-7);

        // Test field dimensions
        let p = backend.get_pressure_field();
        assert_eq!(p.shape(), &[5, 5, 5]);

        let (vx, vy, vz) = backend.get_velocity_fields();
        assert_eq!(vx.shape(), &[5, 5, 5]);
        assert_eq!(vy.shape(), &[5, 5, 5]);
        assert_eq!(vz.shape(), &[5, 5, 5]);
    }

    #[test]
    fn test_backend_intensity_computation() {
        let mut backend = MockBackend::new(3, 3, 3, 1e-7);

        // Set non-zero pressure
        backend.pressure[[1, 1, 1]] = 1e6; // 1 MPa

        // Compute intensity
        let intensity = backend.get_intensity_field().unwrap();

        // Expected: I = p²/(ρc) = (1e6)² / (1.5e6) ≈ 666.7 kW/m²
        let expected = (1e6 * 1e6) / 1.5e6;
        assert!((intensity[[1, 1, 1]] - expected).abs() < 1.0);
    }

    #[test]
    fn test_backend_as_trait_object() {
        // Verify trait object usage
        let backend: Box<dyn AcousticSolverBackend> = Box::new(MockBackend::new(8, 8, 8, 1e-7));

        assert_eq!(backend.get_grid_dimensions(), (8, 8, 8));
        assert_eq!(backend.get_dt(), 1e-7);
    }
}
