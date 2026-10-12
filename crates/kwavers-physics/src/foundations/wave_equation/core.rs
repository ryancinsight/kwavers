//! Core wave equation traits

use super::domain::{Domain, TimeIntegration};
use leto::Array3;

/// Common wave equation methods, independent of thread-safety bounds.
///
/// All wave equation solvers — traditional (finite-difference, spectral) and
/// autodiff-based (PINN, neural operators) — must satisfy this contract.
/// The `Send`/`Sync` refinements are layered on top via [`WaveEquation`] and
/// [`AutodiffWaveEquation`].
pub trait WaveEquationOps {
    /// Get spatial domain specification.
    fn domain(&self) -> &Domain;

    /// Get time integration scheme.
    fn time_integration(&self) -> TimeIntegration;

    /// Compute the CFL stability limit (timestep in seconds).
    ///
    /// For explicit schemes: Δt ≤ CFL_factor · min(Δx, Δy, Δz) / c_max
    /// where c_max is the maximum wave speed in the domain.
    fn cfl_timestep(&self) -> f64;

    /// Evaluate the spatial differential operator `L[u]` at the current state.
    ///
    /// Returns the right-hand side of the wave equation:
    /// ∂²u/∂t² = spatial_operator(u) + source
    fn spatial_operator(&self, field: &Array3<f64>) -> Array3<f64>;

    /// Apply boundary conditions to the field.
    fn apply_boundary_conditions(&mut self, field: &mut Array3<f64>);

    /// Check whether the current state satisfies physics constraints.
    ///
    /// # Errors
    /// Returns `Err` with a description when a constraint is violated.
    fn check_constraints(&self, field: &Array3<f64>) -> Result<(), String>;
}

/// Thread-safe wave equation solver for traditional numerical methods.
///
/// Extends [`WaveEquationOps`] with `Send + Sync` to enable parallel
/// validation, testing, and multi-threaded time-stepping.
///
/// # See Also
///
/// For autodiff/PINN solvers whose internal framework types are `!Sync`,
/// implement [`AutodiffWaveEquation`] instead.
pub trait WaveEquation: WaveEquationOps + Send + Sync {}

/// Wave equation solver for autodiff-based solvers (PINN, neural operators).
///
/// Extends [`WaveEquationOps`] with only `Send`, relaxing the `Sync`
/// constraint required by [`WaveEquation`].  Autodiff frameworks
/// (e.g., Burn, coeus-autograd) use `!Sync` cell types internally for
/// lazy tensor initialisation and gradient accumulation.
///
/// # Thread Safety
///
/// Requires `Send` so solvers can be moved across threads, but not shared.
/// PINN training is typically single-threaded per model instance.
pub trait AutodiffWaveEquation: WaveEquationOps + Send {}
