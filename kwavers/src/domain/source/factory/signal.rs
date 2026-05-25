//! Signal construction for SourceFactory.

use crate::core::error::{ConfigError, KwaversResult};
use crate::domain::signal::{Signal, SignalWindowType, SineWave, ToneBurst};
use crate::domain::source::{EnvelopeType, PulseType};
use std::sync::Arc;

pub(super) fn create_signal(
    pulse: &crate::domain::source::config::PulseParameters,
    frequency: f64,
    amplitude: f64,
    phase: f64,
    delay: f64,
) -> KwaversResult<Arc<dyn Signal>> {
    match pulse.pulse_type {
        PulseType::ContinuousWave | PulseType::Sine => {
            Ok(Arc::new(SineWave::new(frequency, amplitude, phase)))
        }
        PulseType::ToneBurst => {
            let window = match pulse.envelope {
                EnvelopeType::Hann | EnvelopeType::Hanning => SignalWindowType::Hann,
                EnvelopeType::Rectangular => SignalWindowType::Rectangular,
                EnvelopeType::Gaussian => SignalWindowType::Gaussian,
                EnvelopeType::Tukey => SignalWindowType::Tukey { alpha: 0.5 },
                EnvelopeType::Blackman => SignalWindowType::Blackman,
                EnvelopeType::Hamming => SignalWindowType::Hamming,
            };

            Ok(Arc::new(
                ToneBurst::try_new(frequency, pulse.cycles, delay, amplitude)?
                    .with_window(window)
                    .with_phase(phase),
            ))
        }
        _ => Err(ConfigError::InvalidValue {
            parameter: "pulse_type".to_owned(),
            value: format!("{:?}", pulse.pulse_type),
            constraint: "Pulse type not currently supported by factory".to_owned(),
        }
        .into()),
    }
}
