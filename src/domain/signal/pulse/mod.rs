//! Pulse signal generation module
//!
//! Implements various pulsing techniques including:
//! - Gaussian pulse
//! - Rectangular pulse
//! - Tone burst
//! - Pulse train
//! - Ricker wavelet (Mexican hat)
//!
//! Literature references:
//! - Oppenheim & Schafer (2010): "Discrete-Time Signal Processing"
//! - Harris (1978): "On the use of windows for harmonic analysis"
//! - Ricker (1953): "The form and laws of propagation of seismic wavelets"

pub mod gaussian;
pub mod rectangular;
pub mod ricker;
pub mod tone_burst;
pub mod train;

// Re-export main types
pub use super::window::WindowType;
pub use gaussian::GaussianPulse;
pub use rectangular::RectangularPulse;
pub use ricker::RickerWavelet;
pub use tone_burst::ToneBurst;
pub use train::{PulseShape, PulseTrain};

// Physical constants for pulse signals
/// Default Q factor for Gaussian pulses (bandwidth control)
pub const DEFAULT_GAUSSIAN_Q: f64 = 5.0;

/// Minimum pulse width in seconds to ensure numerical stability
pub const MIN_PULSE_WIDTH: f64 = 1e-9;

/// Default rise time fraction for rectangular pulses (10% of pulse width)
pub const DEFAULT_RISE_TIME_FRACTION: f64 = 0.1;

/// Maximum number of cycles in a tone burst
pub const MAX_TONE_BURST_CYCLES: usize = 1000;

/// Default duty cycle for pulse trains (50%)
pub const DEFAULT_DUTY_CYCLE: f64 = 0.5;
