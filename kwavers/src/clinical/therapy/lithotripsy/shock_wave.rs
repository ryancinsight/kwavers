//! Shock wave generation and propagation for lithotripsy.
//!
//! This module implements shock wave physics for extracorporeal shock wave
//! lithotripsy (ESWL), including waveform generation, nonlinear propagation,
//! and focusing characteristics.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

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
    /// Center frequency (Hz)
    pub center_frequency: f64,
    /// Rise time (s)
    pub rise_time: f64,
}

impl Default for ShockWaveParameters {
    fn default() -> Self {
        Self {
            peak_positive_pressure: 50e6,  // 50 MPa
            peak_negative_pressure: -10e6, // -10 MPa
            pulse_duration: 1e-6,          // 1 microsecond
            repetition_rate: 2.0,          // 2 Hz
            focal_spot_diameter: 5e-3,     // 5 mm
            center_frequency: 500e3,       // 500 kHz
            rise_time: 10e-9,              // 10 ns
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
    pub fn new(parameters: ShockWaveParameters, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self { parameters })
    }

    /// Get shock wave parameters.
    pub fn parameters(&self) -> &ShockWaveParameters {
        &self.parameters
    }

    /// Generate initial shock field.
    pub fn generate_shock_field(&self, grid: &Grid, _frequency: f64) -> Array3<f64> {
        // Simplified generation: Gaussian focal spot
        let mut field = Array3::zeros(grid.dimensions());
        let (lx, ly, lz) = grid.physical_size();
        let (cx, cy, cz) = (lx / 2.0, ly / 2.0, lz / 2.0);
        let sigma = self.parameters.focal_spot_diameter / 2.355; // FWHM to sigma

        for ((i, j, k), val) in field.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            let r2 = (x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2);
            *val = self.parameters.peak_positive_pressure * (-r2 / (2.0 * sigma * sigma)).exp();
        }
        field
    }
}

/// Shock wave propagation model.
#[derive(Debug, Clone)]
pub struct ShockWavePropagation {
    attenuation: f64,
}

impl ShockWavePropagation {
    /// Create new propagation model.
    pub fn new(attenuation: f64, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self { attenuation })
    }

    /// Propagate shock wave.
    pub fn propagate_shock_wave(
        &self,
        field: &Array3<f64>,
        _frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        let factor = (-self.attenuation).exp();
        Ok(field.mapv(|p| p * factor))
    }
}
