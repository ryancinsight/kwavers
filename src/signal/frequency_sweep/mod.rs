// signal/frequency_sweep/mod.rs - Unified frequency sweep module

pub mod chirp;
pub mod constants;
pub mod exponential;
pub mod hyperbolic;
pub mod logarithmic;
pub mod stepped;

use crate::signal::Signal;
use std::fmt::Debug;

// Re-export main types
pub use chirp::LinearChirp;
pub use constants::*;
pub use exponential::ExponentialSweep;
pub use hyperbolic::HyperbolicSweep;
pub use logarithmic::LogarithmicSweep;
pub use stepped::SteppedSweep;

/// Base trait for frequency sweep signals
pub trait FrequencySweep: Signal + Debug {
    /// Get instantaneous frequency at time t
    fn instantaneous_frequency(&self, t: f64) -> f64;

    /// Get phase at time t
    fn phase(&self, t: f64) -> f64;

    /// Get sweep rate at time t
    fn sweep_rate(&self, t: f64) -> f64;

    /// Get start frequency
    fn start_frequency(&self) -> f64;

    /// Get stop frequency
    fn stop_frequency(&self) -> f64;

    /// Get sweep duration
    fn duration(&self) -> f64;
}

/// Sweep direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SweepDirection {
    Upward,
    Downward,
}

/// Sweep type for configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SweepType {
    Linear,
    Logarithmic,
    Exponential,
    Hyperbolic,
    Stepped,
}

/// Configuration for frequency sweeps
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub start_frequency: f64,
    pub stop_frequency: f64,
    pub duration: f64,
    pub amplitude: f64,
    pub phase_offset: f64,
    pub sweep_type: SweepType,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            start_frequency: 1000.0,
            stop_frequency: 10000.0,
            duration: 0.001,
            amplitude: 1.0,
            phase_offset: 0.0,
            sweep_type: SweepType::Linear,
        }
    }
}

/// Factory for creating frequency sweeps
#[derive(Debug)]
pub struct SweepFactory;

impl SweepFactory {
    /// Create sweep from configuration
    pub fn create(config: &SweepConfig) -> Box<dyn FrequencySweep> {
        match config.sweep_type {
            SweepType::Linear => Box::new(LinearChirp::new(
                config.start_frequency,
                config.stop_frequency,
                config.duration,
                config.amplitude,
            )),
            SweepType::Logarithmic => Box::new(LogarithmicSweep::new(
                config.start_frequency,
                config.stop_frequency,
                config.duration,
                config.amplitude,
            )),
            SweepType::Exponential => Box::new(ExponentialSweep::new(
                config.start_frequency,
                config.stop_frequency,
                config.duration,
                config.amplitude,
            )),
            SweepType::Hyperbolic => Box::new(HyperbolicSweep::new(
                config.start_frequency,
                config.stop_frequency,
                config.duration,
                config.amplitude,
            )),
            SweepType::Stepped => Box::new(SteppedSweep::new(
                config.start_frequency,
                config.stop_frequency,
                config.duration,
                DEFAULT_FREQUENCY_STEPS,
                config.amplitude,
            )),
        }
    }
}
