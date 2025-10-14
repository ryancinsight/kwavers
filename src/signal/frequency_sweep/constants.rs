// frequency_sweep/constants.rs - Named constants for frequency sweeps

/// Small value threshold for avoiding division by zero
pub const EPSILON: f64 = 1e-10;

/// Singularity avoidance factor for hyperbolic sweep
pub const SINGULARITY_AVOIDANCE_FACTOR: f64 = 0.999;

/// Tolerance for frequency comparison in tests
pub const FREQUENCY_TOLERANCE: f64 = 1e-6;

/// Relative tolerance for frequency comparison
pub const RELATIVE_FREQUENCY_TOLERANCE: f64 = 0.01;

/// Minimum frequency to avoid numerical issues \[Hz\]
pub const MIN_FREQUENCY: f64 = 1.0;

/// Maximum frequency ratio for logarithmic sweeps
pub const MAX_FREQUENCY_RATIO: f64 = 1e6;

/// Default number of steps for stepped frequency sweep
pub const DEFAULT_FREQUENCY_STEPS: usize = 10;

/// Minimum sweep duration \[seconds\]
pub const MIN_SWEEP_DURATION: f64 = 1e-9;

/// Two pi constant for phase calculations
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
