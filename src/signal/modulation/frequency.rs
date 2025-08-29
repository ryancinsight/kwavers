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
    
    fn demodulate(&self, _signal: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        // FM demodulation requires differentiation
        todo!("FM demodulation implementation")
    }
}