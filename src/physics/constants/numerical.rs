//! Numerical constants for simulations

/// Default CFL number for stability
pub const CFL_DEFAULT: f64 = 0.3;

/// CFL safety factor for enhanced stability
/// For 3D FDTD: max stable value is 1/sqrt(3) ≈ 0.577
/// Using 0.5 for safety margin
pub const CFL_SAFETY_FACTOR: f64 = 0.5;

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

// ============================================================================
// Finite Difference Coefficients
// ============================================================================

/// Second-order central difference coefficient
pub const FD2_CENTRAL_COEFF: f64 = 0.5;

/// Second-order forward difference coefficients
pub const FD2_FORWARD_COEFF: [f64; 3] = [-1.5, 2.0, -0.5];

/// Second-order backward difference coefficients
pub const FD2_BACKWARD_COEFF: [f64; 3] = [0.5, -2.0, 1.5];

/// Fourth-order central difference coefficients
pub const FD4_CENTRAL_COEFF: [f64; 5] = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];

/// Fourth-order Laplacian coefficients
pub const FD4_LAPLACIAN_COEFF: [f64; 5] =
    [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0];

// ============================================================================
// PML (Perfectly Matched Layer) Parameters
// ============================================================================

/// Maximum PML absorption coefficient
pub const PML_ALPHA_MAX: f64 = 0.3;

/// PML polynomial exponent
pub const PML_EXPONENT: f64 = 2.0;

/// Default PML thickness in grid points
pub const PML_THICKNESS: usize = 20;

// ============================================================================
// WENO Scheme Constants
// ============================================================================

/// WENO epsilon to avoid division by zero
pub const WENO_EPSILON: f64 = 1e-6;

/// WENO weight for stencil 0
pub const WENO_WEIGHT_0: f64 = 0.1;

/// WENO weight for stencil 1
pub const WENO_WEIGHT_1: f64 = 0.6;

/// WENO weight for stencil 2
pub const WENO_WEIGHT_2: f64 = 0.3;

// ============================================================================
// Viscosity and Stabilization
// ============================================================================

/// Linear artificial viscosity coefficient
pub const LINEAR_VISCOSITY_COEFF: f64 = 0.06;

/// Quadratic artificial viscosity coefficient
pub const QUADRATIC_VISCOSITY_COEFF: f64 = 1.5;

/// Von Neumann-Richtmyer viscosity coefficient
pub const VON_NEUMANN_RICHTMYER_COEFF: f64 = 2.0;

/// Maximum viscosity limit to prevent over-damping
pub const MAX_VISCOSITY_LIMIT: f64 = 0.1;

// ============================================================================
// Error Thresholds
// ============================================================================

/// Interpolation error threshold
pub const INTERPOLATION_ERROR_THRESHOLD: f64 = 1e-6;

/// Conservation error threshold for mass/energy
pub const CONSERVATION_ERROR_THRESHOLD: f64 = 1e-8;

/// Energy threshold for small value cutoff
pub const ENERGY_THRESHOLD: f64 = 1e-10;

/// Amplitude threshold for small value cutoff
pub const AMPLITUDE_THRESHOLD: f64 = 1e-12;

/// Maximum pressure clamp value [Pa] (100 MPa)
pub const MAX_PRESSURE_CLAMP: f64 = 1e8;

// ============================================================================
// FFT and Spectral Method Constants
// ============================================================================

/// FFT k-space scaling factor (2π)
pub const FFT_K_SCALING: f64 = 2.0 * std::f64::consts::PI;

// ============================================================================
// Differential Operator Coefficients
// ============================================================================

/// Second-order differential coefficient
pub const SECOND_ORDER_DIFF_COEFF: f64 = 1.0;

/// Third-order differential coefficient
pub const THIRD_ORDER_DIFF_COEFF: f64 = 1.0;

/// Stencil coefficient (1/4)
pub const STENCIL_COEFF_1_4: f64 = 0.25;

// ============================================================================
// Heterogeneous Media Parameters
// ============================================================================

/// Heterogeneous media smoothing factor
pub const HETEROGENEOUS_SMOOTHING_FACTOR: f64 = 0.1;

/// Symmetric correction factor for heterogeneous media
pub const SYMMETRIC_CORRECTION_FACTOR: f64 = 0.5;

// ============================================================================
// Shock Detection Parameters
// ============================================================================

/// Shock detection threshold for gradient-based detectors
pub const SHOCK_DETECTION_THRESHOLD: f64 = 0.1;

// ============================================================================
// Additional Numerical Constants
// ============================================================================

/// Machine epsilon alias
pub const EPSILON: f64 = f64::EPSILON;

// ============================================================================
// Performance Thresholds
// ============================================================================

/// Large grid threshold for chunked processing
pub const LARGE_GRID_THRESHOLD: usize = 1_000_000;

/// Medium grid threshold
pub const MEDIUM_GRID_THRESHOLD: usize = 100_000;

/// Threshold for enabling chunked processing
pub const CHUNKED_PROCESSING_THRESHOLD: usize = 50_000;

/// Chunk size for small grids
pub const CHUNK_SIZE_SMALL: usize = 1024;

/// Chunk size for medium grids
pub const CHUNK_SIZE_MEDIUM: usize = 4096;

/// Chunk size for large grids
pub const CHUNK_SIZE_LARGE: usize = 16384;

// ============================================================================
// Stability Limits
// ============================================================================

/// Maximum pressure limit for stability (Pa)
pub const PRESSURE_LIMIT: f64 = 1e9;

/// B/A divisor for nonlinearity calculations
pub const B_OVER_A_DIVISOR: f64 = 2.0;

/// Nonlinearity coefficient offset
pub const NONLINEARITY_COEFFICIENT_OFFSET: f64 = 1.0;

// ============================================================================
// Unit Conversion Constants
// ============================================================================

/// Conversion from MHz to Hz
pub const MHZ_TO_HZ: f64 = 1e6;

/// Conversion from cm to m
pub const CM_TO_M: f64 = 0.01;

/// Minimum points per wavelength for accurate simulation
pub const MIN_PPW: f64 = 6.0;

/// CFL safety factor alias
pub const CFL_SAFETY: f64 = CFL_SAFETY_FACTOR;

/// Mechanical index threshold for safety
pub const MI_THRESHOLD: f64 = 1.9;

/// Energy conservation tolerance
pub const ENERGY_CONSERVATION_TOLERANCE: f64 = 1e-6;

/// Default spatial resolution (m)
pub const DEFAULT_SPATIAL_RESOLUTION: f64 = 1e-3;

/// Default number of frequency steps for sweeps
pub const DEFAULT_FREQUENCY_STEPS: usize = 100;

/// Minimum grid spacing for stability (m)
pub const MIN_DX: f64 = 1e-6;

// ============================================================================
// Adaptive Integration Constants
// ============================================================================

/// Default absolute tolerance for adaptive integration
pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-10;

/// Default relative tolerance for adaptive integration
pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-6;

/// Error control exponent for step size adjustment
pub const ERROR_CONTROL_EXPONENT: f64 = 0.2;

/// Factor for half-step calculations
pub const HALF_STEP_FACTOR: f64 = 0.5;

/// Initial time step as fraction of total time
pub const INITIAL_TIME_STEP_FRACTION: f64 = 1e-6;

/// Maximum radius safety factor
pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 100.0;

/// Maximum substeps allowed
pub const MAX_SUBSTEPS: usize = 1000;

/// Maximum temperature limit (K)
pub const MAX_TEMPERATURE: f64 = 1e6;

/// Maximum time step (s)
pub const MAX_TIME_STEP: f64 = 1e-6;

/// Maximum time step decrease factor
pub const MAX_TIME_STEP_DECREASE: f64 = 0.1;

/// Maximum time step increase factor
pub const MAX_TIME_STEP_INCREASE: f64 = 5.0;

/// Maximum velocity as fraction of sound speed
pub const MAX_VELOCITY_FRACTION: f64 = 0.1;

/// Minimum radius safety factor
pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.1;

/// Minimum temperature limit (K)
pub const MIN_TEMPERATURE: f64 = 100.0;

/// Minimum time step (s)
pub const MIN_TIME_STEP: f64 = 1e-15;

/// Safety factor for adaptive stepping
pub const SAFETY_FACTOR: f64 = 0.9;
