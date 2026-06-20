//! Solver-specific constants and parameters

// ============================================================================
// CFL Stability Constants
// ============================================================================

/// CFL number for explicit time stepping (dimensionless)
/// Conservative value for stability across different schemes
pub const CFL_NUMBER: f64 = 0.3;

/// Maximum CFL number for FDTD schemes
/// Reference: Taflove & Hagness (2005) "Computational Electrodynamics"
pub const CFL_MAX_FDTD: f64 = 0.5;

/// CFL number for spectral methods
/// More restrictive due to higher-order accuracy
pub const CFL_SPECTRAL: f64 = 0.2;

// ============================================================================
// Convergence Tolerances
// ============================================================================

/// Default tolerance for iterative solver convergence
pub const SOLVER_TOLERANCE: f64 = 1e-10;

/// Conservation tolerance for mass/energy checks
pub const CONSERVATION_TOLERANCE: f64 = 1e-10;

/// Relative tolerance for convergence checks
pub const RELATIVE_TOLERANCE: f64 = 1e-8;

/// Absolute tolerance for small value comparisons
pub const ABSOLUTE_TOLERANCE: f64 = 1e-12;

// ============================================================================
// Discontinuity Detection
// ============================================================================

/// Default threshold for discontinuity detection
/// Based on gradient magnitude relative to field values
pub const DISCONTINUITY_THRESHOLD: f64 = 0.1;

/// Minimum gradient for shock detection
pub const SHOCK_GRADIENT_MIN: f64 = 0.5;

// ============================================================================
// Grid Spacing Defaults
// ============================================================================

/// Default spatial resolution for medical ultrasound (meters)
/// 0.1 mm provides λ/10 sampling at 1.5 `MHz`
pub const DEFAULT_DX: f64 = 1e-4;

/// Fine grid spacing for high-frequency simulations
pub const FINE_DX: f64 = 1e-5;

/// Coarse grid spacing for low-frequency simulations  
pub const COARSE_DX: f64 = 1e-3;

// ============================================================================
// Numerical Method Parameters
// ============================================================================

/// Stencil coefficients for 2nd order finite difference
pub const STENCIL_2ND_ORDER: [f64; 3] = [-1.0, 2.0, -1.0];

/// Stencil coefficients for 4th order finite difference
pub const STENCIL_4TH_ORDER: [f64; 5] =
    [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0];

/// Default polynomial order for spectral methods
pub const DEFAULT_POLYNOMIAL_ORDER: usize = 8;

/// Maximum iterations for iterative solvers
pub const MAX_SOLVER_ITERATIONS: usize = 1000;

// ============================================================================
// Benchmark Tolerances
// ============================================================================
//
// These are acceptance tolerances for the analytical-reference benchmark suite,
// not physical constants. Their relative magnitudes follow the discretization
// error of each test, not an external standard:
//   - plane wave (5%): the cleanest case — a single mode on a uniform grid; error
//     is dominated by numerical dispersion ~O((kΔx)²), small at the benchmark
//     resolution.
//   - point source (10%): loosest — the source singularity is unresolved on the
//     grid, so near-field amplitude carries an irreducible discretization error
//     larger than the plane-wave case.
//   - dispersion (2%): tightest — phase velocity is measured from a fitted slope
//     over many wavelengths, which averages out per-sample noise.
// Tightening any of these is a signal to refine the solver, never to relax here.

/// Maximum acceptable amplitude error for plane-wave benchmarks (numerical
/// dispersion dominated; see module note).
pub const PLANE_WAVE_ERROR_TOLERANCE: f64 = 0.05; // 5%

/// Maximum acceptable amplitude error for point-source benchmarks (source-
/// singularity discretization dominated; see module note).
pub const POINT_SOURCE_ERROR_TOLERANCE: f64 = 0.1; // 10%

/// Maximum acceptable error for dispersion (phase-velocity) benchmarks
/// (slope-fit averaged; see module note).
pub const DISPERSION_ERROR_TOLERANCE: f64 = 0.02; // 2%
