//! Quadrature Amplitude Modulation (QAM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// QAM implementation
#[derive(Debug, Clone)]
pub struct QuadratureAmplitudeModulation {
    params: ModulationParams,
}

impl QuadratureAmplitudeModulation {
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for QuadratureAmplitudeModulation {
    fn modulate(&self, _carrier: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        todo!("QAM modulation requires I/Q components")
    }
    
    fn demodulate(&self, _signal: &[f64], _t: &[f64]) -> KwaversResult<Vec<f64>> {
        todo!("QAM demodulation implementation")
    }
}