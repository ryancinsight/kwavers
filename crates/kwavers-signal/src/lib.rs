#![doc = include_str!("../README.md")]

pub mod amplitude;
pub mod analytic;
pub mod filter;
pub mod frequency;
pub mod frequency_sweep;
pub mod functions;
pub mod modulation;
pub mod phase;
pub mod pulse;
pub mod special;
#[cfg(test)]
mod tests;
pub mod traits;
pub mod waveform;
pub mod window;

pub use filter::{Filter, FrequencyFilter};
pub use functions::{
    add_noise, create_cw_signal, create_cw_signals, next_pow2, pad_zeros, sample_signal,
    tone_burst_series, ToneBurstSpec,
};
pub use special::{NullSignal, TimeVaryingSignal};
pub use traits::Signal;
pub use waveform::{SineWave, SquareWave, TriangleWave};
pub use window::{window_value, SignalWindowType};

// Re-export pulse signals
pub use pulse::{
    DomainRickerWavelet, GaussianPulse, PulseShape, PulseTrain, RectangularPulse, ToneBurst,
};

// Re-export frequency sweeps
pub use frequency_sweep::{
    ExponentialSweep, FrequencySweep, HyperbolicSweep, LinearChirp, LogarithmicSweep, SteppedSweep,
    SweepConfig, SweepDirection, SweepType,
};

// Re-export modulation types
pub use modulation::{
    AmplitudeModulation, FrequencyModulation, PhaseModulation, PulseWidthModulation,
    QuadratureAmplitudeModulation,
};
