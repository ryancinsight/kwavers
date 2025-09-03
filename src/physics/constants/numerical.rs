//! Numerical constants for simulations

/// Default CFL number for stability
pub const CFL_DEFAULT: f64 = 0.3;

/// Maximum recommended CFL number
pub const CFL_MAX: f64 = 0.5;

/// Minimum grid points per wavelength
pub const MIN_POINTS_PER_WAVELENGTH: usize = 10;

/// Default tolerance for iterative solvers
pub const SOLVER_TOLERANCE: f64 = 1e-6;

/// Maximum iterations for iterative solvers
pub const MAX_ITERATIONS: usize = 1000;

/// Machine epsilon for f64
pub const MACHINE_EPSILON: f64 = f64::EPSILON;

/// Small value to prevent division by zero
pub const SMALL_VALUE: f64 = 1e-12;

/// Large value for boundary conditions
pub const LARGE_VALUE: f64 = 1e12;

/// Default smoothing parameter
pub const SMOOTHING_PARAMETER: f64 = 0.01;

/// Convergence criterion for Newton-Raphson
pub const NEWTON_TOLERANCE: f64 = 1e-10;

/// Maximum Newton-Raphson iterations
pub const NEWTON_MAX_ITER: usize = 50;