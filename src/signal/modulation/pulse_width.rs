//! Pulse Width Modulation (PWM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// PWM implementation
#[derive(Debug, Clone)]
pub struct PulseWidthModulation {
    params: ModulationParams,
}

impl PulseWidthModulation {
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
                if phase < duty_cycle { 1.0 } else { -1.0 }
            })
            .collect())
    }
    
    fn demodulate(&self, _signal: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        todo!("PWM demodulation implementation")
    }
}