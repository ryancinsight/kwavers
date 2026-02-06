//! Phase Modulation (PM)

use super::{Modulation, ModulationParams};
use crate::core::error::{KwaversError, KwaversResult};

/// Phase modulation implementation
#[derive(Debug, Clone)]
pub struct PhaseModulation {
    params: ModulationParams,
}

impl PhaseModulation {
    #[must_use]
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for PhaseModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let beta = self.params.modulation_index;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &msg)| (omega_c * ti + beta * msg).cos())
            .collect())
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        use crate::domain::signal::analytic::instantaneous_phase;
        use ndarray::Array1;

        if signal.is_empty() {
            return Ok(Vec::new());
        }

        if signal.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "Signal and time arrays must have same length".to_string(),
            ));
        }

        if self.params.modulation_index == 0.0 {
            return Err(KwaversError::InvalidInput(
                "Modulation index must be non-zero".to_string(),
            ));
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let beta = self.params.modulation_index;

        let phase_wrapped = instantaneous_phase(&Array1::from_vec(signal.to_vec()));
        let mut phase_unwrapped = Vec::with_capacity(phase_wrapped.len());

        if let Some(&first) = phase_wrapped.first() {
            phase_unwrapped.push(first);
        }

        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 1..phase_wrapped.len() {
            let prev_wrapped = phase_wrapped[i - 1];
            let curr_wrapped = phase_wrapped[i];
            let mut dphi = curr_wrapped - prev_wrapped;
            while dphi > std::f64::consts::PI {
                dphi -= two_pi;
            }
            while dphi < -std::f64::consts::PI {
                dphi += two_pi;
            }
            let prev_unwrapped = phase_unwrapped[i - 1];
            phase_unwrapped.push(prev_unwrapped + dphi);
        }

        Ok(phase_unwrapped
            .iter()
            .zip(t.iter())
            .map(|(&phi, &ti)| (phi - omega_c * ti) / beta)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pm_modulation_demodulation_recovers_message_interior() {
        let params = ModulationParams {
            carrier_freq: 1000.0,
            sample_rate: 10000.0,
            modulation_index: 0.7,
        };

        let sample_rate = params.sample_rate;
        let pm = PhaseModulation::new(params);

        let n = 1024;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / sample_rate).collect();
        let message: Vec<f64> = t
            .iter()
            .map(|&ti| 0.25 * (2.0 * std::f64::consts::PI * 5.0 * ti).sin())
            .collect();

        let modulated = pm.modulate(&message, &t).unwrap();
        let demodulated = pm.demodulate(&modulated, &t).unwrap();

        let start = 32;
        let end = n - 32;
        let mae = demodulated[start..end]
            .iter()
            .zip(message[start..end].iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / (end - start) as f64;

        assert!(mae < 0.15, "Mean absolute error too large: {mae}");
    }
}
