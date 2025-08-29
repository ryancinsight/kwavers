//! Phase Modulation (PM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// Phase modulation implementation
#[derive(Debug, Clone)]
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
    
    fn demodulate(&self, _signal: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        todo!("PM demodulation implementation")
    }
}