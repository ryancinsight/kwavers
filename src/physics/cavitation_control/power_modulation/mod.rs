//! Power Modulation for Cavitation Control
//!
//! Implements amplitude and power modulation schemes for controlling cavitation activity.
//!
//! References:
//! - Hockham et al. (2010): "A real-time controller for sustaining thermally relevant acoustic cavitation"
//! - O'Reilly & Hynynen (2012): "Blood-brain barrier: real-time feedback-controlled focused ultrasound"
//! - Tsai et al. (2016): "Real-time monitoring of focused ultrasound blood-brain barrier opening"

mod amplitude;
mod constants;
mod duty_cycle;
mod filters;
mod modulator;
mod pulse_sequence;
mod safety;
mod schemes;

pub use amplitude::AmplitudeController;
pub use crate::physics::constants::*;
pub use duty_cycle::DutyCycleController;
pub use filters::ExponentialFilter;
pub use modulator::PowerModulator;
pub use pulse_sequence::{PulseDescriptor, PulseSequenceGenerator};
pub use safety::SafetyLimiter;
pub use schemes::{ModulationScheme, PowerControl};

#[cfg(test)]
mod tests;
