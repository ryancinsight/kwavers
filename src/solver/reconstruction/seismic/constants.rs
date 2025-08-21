//! Seismic reconstruction constants
//!
//! Named constants for seismic imaging algorithms to maintain SSOT principle

/// Default number of time steps for forward modeling
pub const DEFAULT_TIME_STEPS: usize = 2000;

/// Default time step size (seconds) - must satisfy CFL condition
pub const DEFAULT_TIME_STEP: f64 = 5e-4; // 0.5 ms

/// Default dominant frequency for Ricker wavelet (Hz)
pub const DEFAULT_RICKER_FREQUENCY: f64 = 15.0;

/// Minimum velocity for physical bounds (m/s)
pub const MIN_VELOCITY: f64 = 1000.0;

/// Maximum velocity for physical bounds (m/s)
pub const MAX_VELOCITY: f64 = 8000.0;

/// CFL stability factor for acoustic wave equation
pub const CFL_STABILITY_FACTOR: f64 = 0.5;

/// Default convergence tolerance for FWI
pub const DEFAULT_FWI_TOLERANCE: f64 = 1e-6;

/// Default maximum FWI iterations
pub const DEFAULT_FWI_ITERATIONS: usize = 100;

/// Default regularization parameter for Tikhonov regularization
pub const DEFAULT_REGULARIZATION_LAMBDA: f64 = 1e-4;

/// Default step length for line search
pub const DEFAULT_STEP_LENGTH: f64 = 1e-3;

/// Minimum step length for line search
pub const MIN_STEP_LENGTH: f64 = 1e-8;

/// Maximum step length for line search
pub const MAX_STEP_LENGTH: f64 = 1.0;

/// Default number of line search iterations
pub const DEFAULT_LINE_SEARCH_ITERATIONS: usize = 20;

/// Default Armijo constant for line search
pub const ARMIJO_CONSTANT: f64 = 1e-4;

/// Default Wolfe constant for line search
pub const WOLFE_CONSTANT: f64 = 0.9;

/// RTM correlation window size (samples)
pub const RTM_CORRELATION_WINDOW: usize = 100;

/// RTM Laplacian filter coefficient
pub const RTM_LAPLACIAN_COEFF: f64 = 0.1;

/// Gradient smoothing radius (grid points)
pub const GRADIENT_SMOOTHING_RADIUS: usize = 3;

/// Gradient clipping threshold
pub const GRADIENT_CLIPPING_THRESHOLD: f64 = 1e3;

/// Water layer velocity (m/s) for marine seismic
pub const WATER_VELOCITY: f64 = 1500.0;

/// Typical sediment velocity (m/s)
pub const SEDIMENT_VELOCITY: f64 = 2000.0;

/// Typical basement velocity (m/s)
pub const BASEMENT_VELOCITY: f64 = 5000.0;

// RTM (Reverse Time Migration) Constants
/// Storage decimation factor for wavefield checkpointing
pub const RTM_STORAGE_DECIMATION: usize = 10;

/// Amplitude threshold for RTM imaging condition
pub const RTM_AMPLITUDE_THRESHOLD: f64 = 1e-10;

/// Scaling factor for Laplacian-based RTM imaging
pub const RTM_LAPLACIAN_SCALING: f64 = 0.01;

// Wavelet Constants
/// Ricker wavelet time shift factor
pub const RICKER_TIME_SHIFT: f64 = 1.5;

// FWI (Full Waveform Inversion) Constants
/// Gradient scaling factor for FWI updates
pub const GRADIENT_SCALING_FACTOR: f64 = 1e-6;

/// Minimum gradient norm threshold for convergence
pub const MIN_GRADIENT_NORM: f64 = 1e-12;

/// Maximum line search iterations
pub const MAX_LINE_SEARCH_ITERATIONS: usize = 20;

/// Armijo condition constant c1 for line search
pub const ARMIJO_C1: f64 = 1e-4;

/// Line search backtracking factor
pub const LINE_SEARCH_BACKTRACK: f64 = 0.5;
