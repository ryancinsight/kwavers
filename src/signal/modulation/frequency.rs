//! Frequency Modulation (FM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// Frequency modulation implementation
#[derive(Debug, Clone)]
pub struct FrequencyModulation {
    params: ModulationParams,
}

impl FrequencyModulation {
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for FrequencyModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // FM modulation: y(t) = A*cos(2π*fc*t + β*∫m(τ)dτ)
        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let beta = self.params.modulation_index;

        let mut phase = 0.0;
        let dt = 1.0 / self.params.sample_rate;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &msg)| {
                phase += msg * dt;
                (omega_c * ti + beta * phase).cos()
            })
            .collect())
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // FM demodulation using instantaneous frequency estimation
        if signal.len() < 2 {
            return Ok(signal.to_vec());
        }

        let dt = if t.len() > 1 {
            t[1] - t[0]
        } else {
            1.0 / self.params.sample_rate
        };
        let mut demodulated = Vec::with_capacity(signal.len());

        // Use phase difference method
        let mut prev_phase = 0.0;
        for (i, &sample) in signal.iter().enumerate() {
            let phase = sample.atan2(0.0); // Simplified - real implementation needs Hilbert transform
            let phase_diff = phase - prev_phase;

            // Unwrap phase
            let phase_diff = if phase_diff > std::f64::consts::PI {
                phase_diff - 2.0 * std::f64::consts::PI
            } else if phase_diff < -std::f64::consts::PI {
                phase_diff + 2.0 * std::f64::consts::PI
            } else {
                phase_diff
            };

            // Convert phase difference to frequency deviation
            let freq_deviation = phase_diff / (2.0 * std::f64::consts::PI * dt);
            demodulated.push(freq_deviation / self.params.modulation_index);

            prev_phase = phase;
        }

        Ok(demodulated)
    }
}
