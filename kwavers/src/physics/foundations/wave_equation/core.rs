//! Core wave equation traits

use super::domain::{Domain, TimeIntegration};
use ndarray::ArrayD;

/// Abstract wave equation trait for traditional numerical solvers
///
/// This trait defines the mathematical structure common to all wave equations
/// implemented with traditional numerical methods (finite difference, finite element,
/// spectral methods, etc.).
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` to enable parallel validation and testing.
/// Traditional solvers using ndarray and standard Rust data structures satisfy this.
///
/// # See Also
///
/// For neural network and autodiff-based solvers that cannot satisfy `Sync` due to
/// internal framework constraints, see [`AutodiffWaveEquation`].
pub trait WaveEquation: Send + Sync {
    /// Get spatial domain specification
    fn domain(&self) -> &Domain;

    /// Get time integration scheme
    fn time_integration(&self) -> TimeIntegration;

    /// Compute the CFL stability limit (timestep in seconds)
    ///
    /// For explicit schemes: Δt ≤ CFL_factor * min(Δx, Δy, Δz) / c_max
    /// where c_max is the maximum wave speed in the domain.
    fn cfl_timestep(&self) -> f64;

    /// Evaluate the spatial differential operator L[u] at current state
    ///
    /// Returns the right-hand side of the wave equation:
    /// ∂²u/∂t² = spatial_operator(u) + source
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;

    /// Apply boundary conditions to field
    fn apply_boundary_conditions(&mut self, field: &mut ArrayD<f64>);

    /// Check if the current state satisfies physics constraints
    ///
    /// Returns Ok(()) if constraints are satisfied, Err with violation description otherwise.
    fn check_constraints(&self, field: &ArrayD<f64>) -> Result<(), String>;
}

/// Abstract wave equation trait for autodiff-based solvers
///
/// This trait mirrors [`WaveEquation`] but relaxes the `Sync` constraint to accommodate
/// neural network frameworks (e.g., Burn) that use internal cell types for lazy
/// initialization and gradient tracking.
///
/// # Design Rationale
///
/// Automatic differentiation frameworks like Burn use `std::cell::OnceCell` and similar
/// constructs internally for:
/// - Lazy tensor initialization
/// - Gradient accumulation
/// - Computation graph construction
///
/// These types are `!Sync` by design, preventing implementations from satisfying the
/// `Sync` bound required by traditional numerical solvers.
///
/// Rather than compromising either API, we maintain separate trait hierarchies:
/// - `WaveEquation` (with `Sync`) for traditional methods
/// - `AutodiffWaveEquation` (without `Sync`) for neural networks
///
/// This preserves thread safety guarantees where possible while enabling autodiff
/// integration where necessary.
///
/// # Thread Safety
///
/// This trait requires only `Send`, allowing solvers to be moved between threads but
/// not shared. Most PINN training is single-threaded per model instance, making this
/// sufficient.
pub trait AutodiffWaveEquation: Send {
    /// Get spatial domain specification
    fn domain(&self) -> &Domain;

    /// Get time integration scheme
    fn time_integration(&self) -> TimeIntegration;

    /// Compute the CFL stability limit (timestep in seconds)
    ///
    /// For explicit schemes: Δt ≤ CFL_factor * min(Δx, Δy, Δz) / c_max
    /// where c_max is the maximum wave speed in the domain.
    fn cfl_timestep(&self) -> f64;

    /// Evaluate the spatial differential operator L[u] at current state
    ///
    /// Returns the right-hand side of the wave equation:
    /// ∂²u/∂t² = spatial_operator(u) + source
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;

    /// Apply boundary conditions to field
    fn apply_boundary_conditions(&mut self, field: &mut ArrayD<f64>);

    /// Check if the current state satisfies physics constraints
    ///
    /// Returns Ok(()) if constraints are satisfied, Err with violation description otherwise.
    fn check_constraints(&self, field: &ArrayD<f64>) -> Result<(), String>;
}
