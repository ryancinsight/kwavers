//! Quadrature Amplitude Modulation (QAM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// QAM implementation
#[derive(Debug, Clone))]
pub struct QuadratureAmplitudeModulation {
    params: ModulationParams,
}

impl QuadratureAmplitudeModulation {
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for QuadratureAmplitudeModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // Simplified QAM - treats carrier as I component, generates Q internally
        if carrier.len() != t.len() {
            return Err(crate::error::KwaversError::InvalidInput(
                "Carrier and time arrays must have same length".to_string(),
            ));
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &i_component)| {
                // Generate Q component (90° phase shift)
                let q_component = i_component * 0.5; // Simplified

                // QAM signal = I*cos(ωt) - Q*sin(ωt)
                i_component * (omega_c * ti).cos() - q_component * (omega_c * ti).sin()
            })
            .collect())
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // Coherent QAM demodulation
        if signal.is_empty() {
            return Ok(Vec::new());
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let mut demodulated = Vec::with_capacity(signal.len());

        for (i, &sample) in signal.iter().enumerate() {
            let ti = if i < t.len() {
                t[i]
            } else {
                i as f64 / self.params.sample_rate
            };

            // Extract I component
            let i_component = 2.0 * sample * (omega_c * ti).cos();

            // Extract Q component
            let q_component = -2.0 * sample * (omega_c * ti).sin();

            // Combine (simplified - real QAM needs proper constellation mapping)
            demodulated.push((i_component.powi(2) + q_component.powi(2)).sqrt());
        }

        Ok(demodulated)
    }
}
