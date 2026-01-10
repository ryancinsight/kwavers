//! Pulse Width Modulation (PWM)

use super::{Modulation, ModulationParams};
use crate::domain::core::error::KwaversResult;

/// PWM implementation
#[derive(Debug, Clone)]
pub struct PulseWidthModulation {
    params: ModulationParams,
}

impl PulseWidthModulation {
    #[must_use]
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for PulseWidthModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        let period = 1.0 / self.params.carrier_freq;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &msg)| {
                let phase = (ti % period) / period;
                let duty_cycle = 0.5 + 0.5 * msg; // Map [-1,1] to [0,1]
                if phase < duty_cycle {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect())
    }

    fn demodulate(&self, signal: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        // PWM demodulation via low-pass filtering (standard technique per Black 1953)
        // Extracts pulse width information by averaging over carrier period
        // **Reference**: Black (1953) "Pulse Code Modulation" Bell System Technical Journal
        if signal.is_empty() {
            return Ok(Vec::new());
        }

        let period = 1.0 / self.params.carrier_freq;
        let samples_per_period = (self.params.sample_rate * period) as usize;

        let mut demodulated = Vec::with_capacity(signal.len());

        // Simple moving average as low-pass filter
        for i in 0..signal.len() {
            let start = i.saturating_sub(samples_per_period / 2);
            let end = (i + samples_per_period / 2).min(signal.len());

            let avg: f64 = signal[start..end].iter().sum::<f64>() / (end - start) as f64;
            demodulated.push(avg);
        }

        Ok(demodulated)
    }
}
