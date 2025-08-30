// thermal/source.rs - Heat source models

use crate::grid::Grid;
use ndarray::Array3;

/// Heat source for thermal calculations
pub trait HeatSource {
    fn compute(&self, grid: &Grid, time: f64) -> Array3<f64>;
}

/// Thermal source from acoustic absorption
#[derive(Debug)]
pub struct ThermalSource {
    absorption_coefficient: f64,
    intensity: Array3<f64>,
}

impl ThermalSource {
    pub fn new(absorption: f64, intensity: Array3<f64>) -> Self {
        Self {
            absorption_coefficient: absorption,
            intensity,
        }
    }
}

impl HeatSource for ThermalSource {
    fn compute(&self, _grid: &Grid, _time: f64) -> Array3<f64> {
        &self.intensity * self.absorption_coefficient
    }
}
