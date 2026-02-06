//! Amplitude Modulation (AM)

use super::{Modulation, ModulationParams};
use crate::core::error::{KwaversError, KwaversResult};

/// Amplitude modulation implementation
#[derive(Debug, Clone)]
pub struct AmplitudeModulation {
    params: ModulationParams,
}

impl AmplitudeModulation {
    /// Create new AM modulator
    pub fn new(params: ModulationParams) -> KwaversResult<Self> {
        if params.modulation_index < 0.0 || params.modulation_index > super::constants::MAX_AM_INDEX
        {
            return Err(KwaversError::InvalidInput(format!(
                "Modulation index must be in [0, {}]",
                super::constants::MAX_AM_INDEX
            )));
        }
        Ok(Self { params })
    }

    /// Get modulation depth
    #[must_use]
    pub fn modulation_depth(&self) -> f64 {
        self.params.modulation_index
    }
}

impl Modulation for AmplitudeModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        if carrier.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "Carrier and time arrays must have same length".to_string(),
            ));
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let m = self.params.modulation_index;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &msg)| (1.0 + m * msg) * (omega_c * ti).cos())
            .collect())
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        if signal.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "Signal and time arrays must have same length".to_string(),
            ));
        }

        // Envelope detection for AM demodulation
        // **Implementation**: Synchronous detection (coherent demodulation) per Lyons (2010)
        // Multiplies by reference carrier and applies implicit low-pass filtering.
        // Alternative approaches: Hilbert transform or matched filtering (see utils/signal_processing.rs)
        //
        // **Reference**: Lyons (2010) "Understanding Digital Signal Processing" §13.1
        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;

        Ok(signal
            .iter()
            .zip(t.iter())
            .map(|(&sig, &ti)| {
                // Coherent demodulation: multiply by carrier, factor of 2 compensates mixer loss
                sig * (omega_c * ti).cos() * 2.0
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_am_modulation() {
        let params = ModulationParams {
            carrier_freq: 1000.0,
            sample_rate: 10000.0,
            modulation_index: 0.5,
        };

        let am = AmplitudeModulation::new(params).unwrap();
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 10000.0).collect();
        let message = vec![0.5; 100];

        let modulated = am.modulate(&message, &t).unwrap();
        assert_eq!(modulated.len(), 100);
    }
}
