//! Numerical constants and coefficients

use std::f64::consts::PI;

/// Default CFL safety factor for stability
/// For 3D FDTD: max stable value is 1/sqrt(3) ≈ 0.577
/// Using 0.5 for safety margin (Taflove & Hagness, 2005)
pub const CFL_SAFETY_FACTOR: f64 = 0.5;

/// Default grid resolution points
pub const DEFAULT_GRID_POINTS: usize = 100;

/// Minimum grid points for valid simulation
pub const MIN_GRID_POINTS: usize = 10;

/// Maximum grid points for memory safety
pub const MAX_GRID_POINTS: usize = 1000;

/// Default spatial resolution in meters
pub const DEFAULT_SPATIAL_RESOLUTION: f64 = 1e-3;

/// Machine epsilon for numerical comparisons
pub const EPSILON: f64 = 1e-10;

/// Maximum iterations for iterative solvers
pub const MAX_ITERATIONS: usize = 1000;

/// Convergence tolerance
pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;

/// Finite difference coefficients - second order central
pub const FD2_CENTRAL_COEFF: f64 = 0.5;

/// Finite difference coefficients - second order forward
pub const FD2_FORWARD_COEFF: [f64; 3] = [-1.5, 2.0, -0.5];

/// Finite difference coefficients - second order backward
pub const FD2_BACKWARD_COEFF: [f64; 3] = [0.5, -2.0, 1.5];

/// Fourth-order central difference coefficients
pub const FD4_CENTRAL_COEFF: [f64; 5] = [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0];

/// Fourth-order Laplacian coefficients
pub const FD4_LAPLACIAN_COEFF: [f64; 5] = [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0];

/// FFT k-space scaling factor
pub const FFT_K_SCALING: f64 = 2.0 * PI;

/// PML absorption parameters
pub const PML_ALPHA_MAX: f64 = 0.3;
pub const PML_EXPONENT: f64 = 2.0;
pub const PML_THICKNESS: usize = 20;

/// Second order differential coefficient
pub const SECOND_ORDER_DIFF_COEFF: f64 = 1.0;

/// Third order differential coefficient  
pub const THIRD_ORDER_DIFF_COEFF: f64 = 1.0;