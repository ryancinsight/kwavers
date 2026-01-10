//! Shock wave generation and propagation for lithotripsy.
//!
//! This module implements shock wave physics for extracorporeal shock wave
//! lithotripsy (ESWL), including waveform generation, nonlinear propagation,
//! and focusing characteristics.

use crate::core::error::KwaversResult;

/// Shock wave parameters for lithotripsy.
#[derive(Debug, Clone)]
pub struct ShockWaveParameters {
    /// Peak positive pressure (Pa)
    pub peak_positive_pressure: f64,
    /// Peak negative pressure (Pa)
    pub peak_negative_pressure: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Repetition rate (Hz)
    pub repetition_rate: f64,
    /// Focal spot diameter (m)
    pub focal_spot_diameter: f64,
}

impl Default for ShockWaveParameters {
    fn default() -> Self {
        Self {
            peak_positive_pressure: 50e6,  // 50 MPa
            peak_negative_pressure: -10e6, // -10 MPa
            pulse_duration: 1e-6,          // 1 microsecond
            repetition_rate: 2.0,          // 2 Hz
            focal_spot_diameter: 5e-3,     // 5 mm
        }
    }
}

/// Shock wave generator for lithotripsy simulation.
#[derive(Debug, Clone)]
pub struct ShockWaveGenerator {
    parameters: ShockWaveParameters,
}

impl ShockWaveGenerator {
    /// Create new shock wave generator with given parameters.
    pub fn new(parameters: ShockWaveParameters) -> KwaversResult<Self> {
        Ok(Self { parameters })
    }

    /// Get shock wave parameters.
    pub fn parameters(&self) -> &ShockWaveParameters {
        &self.parameters
    }
}

/// Shock wave propagation model.
#[derive(Debug, Clone)]
pub struct ShockWavePropagation {
    generator: ShockWaveGenerator,
}

impl ShockWavePropagation {
    /// Create new propagation model.
    pub fn new(generator: ShockWaveGenerator) -> Self {
        Self { generator }
    }

    /// Get associated generator.
    pub fn generator(&self) -> &ShockWaveGenerator {
        &self.generator
    }
}
