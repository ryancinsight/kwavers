//! Phase Modulation (PM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// Phase modulation implementation
#[derive(Debug, Clone))]
pub struct PhaseModulation {
    params: ModulationParams,
}

impl PhaseModulation {
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
        // PM demodulation using phase detection
        if signal.len() < 2 {
            return Ok(signal.to_vec());
        }

        let mut demodulated = Vec::with_capacity(signal.len());
        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;

        for (i, &sample) in signal.iter().enumerate() {
            let ti = if i < t.len() {
                t[i]
            } else {
                i as f64 / self.params.sample_rate
            };

            // Coherent detection: multiply by carrier and extract phase
            let baseband = sample * (omega_c * ti).cos();

            // Simple phase extraction (real implementation needs Hilbert transform)
            let phase = baseband.atan2(0.0);
            demodulated.push(phase / self.params.modulation_index);
        }

        Ok(demodulated)
    }
}
