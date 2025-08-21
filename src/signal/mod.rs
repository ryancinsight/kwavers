// signal/mod.rs
//! Signal generation and processing module
//!
//! Comprehensive signal generation library including:
//! - Basic waveforms (sine, square, triangle)
//! - Pulse signals (Gaussian, rectangular, tone burst, Ricker)
//! - Frequency sweeps (linear, logarithmic, hyperbolic)
//! - Modulation techniques (AM, FM, PM, QAM, PWM)
//! - Windowing functions

use std::fmt::Debug;

pub mod amplitude;
pub mod chirp;
pub mod frequency;
pub mod frequency_sweep;
pub mod modulation;
pub mod phase;
pub mod pulse;
pub mod sine_wave;
pub mod sweep;

// Core Signal trait
pub trait Signal: Debug + Send + Sync {
    /// Get the signal amplitude at time t
    fn amplitude(&self, t: f64) -> f64;

    /// Get the signal duration (if finite)
    fn duration(&self) -> Option<f64> {
        None
    }

    /// Get the instantaneous frequency at time t
    fn frequency(&self, t: f64) -> f64;

    /// Get the instantaneous phase at time t
    fn phase(&self, t: f64) -> f64;

    /// Clone the signal into a boxed trait object
    fn clone_box(&self) -> Box<dyn Signal>;
}

impl Clone for Box<dyn Signal> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// Re-export commonly used signal types
pub use chirp::ChirpSignal;
pub use sine_wave::SineWave;
pub use sweep::SweepSignal;

// Re-export pulse signals
pub use pulse::{
    GaussianPulse, PulseShape, PulseTrain, RectangularPulse, RickerWavelet, ToneBurst, WindowType,
};

// Re-export frequency sweeps
pub use frequency_sweep::{
    HyperbolicFrequencySweep, LinearFrequencySweep, LogarithmicFrequencySweep,
    PolynomialFrequencySweep, SteppedFrequencySweep, TransitionType,
};

// Re-export modulation types
pub use modulation::{
    AmplitudeModulation, FrequencyModulation, PhaseModulation, PulseWidthModulation,
    QuadratureAmplitudeModulation,
};
