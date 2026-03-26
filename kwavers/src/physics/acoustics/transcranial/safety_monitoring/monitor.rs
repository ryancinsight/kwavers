//! SafetyMonitor struct and field update orchestration

use super::types::{MechanicalIndex, SafetyThresholds, ThermalDose};
use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Safety monitoring system for tFUS
#[derive(Debug)]
pub struct SafetyMonitor {
    /// Temperature field (°C)
    pub(crate) temperature: Array3<f64>,
    /// Acoustic pressure field (Pa)
    pub(crate) pressure: Array3<f64>,
    /// Thermal dose accumulation
    pub(crate) thermal_dose: ThermalDose,
    /// Mechanical index
    pub(crate) mechanical_index: MechanicalIndex,
    /// Tissue perfusion rate (1/s)
    pub(crate) _perfusion_rate: f64,
    /// Acoustic frequency (Hz)
    pub(crate) frequency: f64,
    /// Safety thresholds
    pub(crate) thresholds: SafetyThresholds,
}

impl SafetyMonitor {
    /// Create new safety monitor
    pub fn new(grid_dims: (usize, usize, usize), perfusion_rate: f64, frequency: f64) -> Self {
        let temperature = Array3::from_elem(grid_dims, 37.0); // Body temperature
        let pressure = Array3::zeros(grid_dims);
        let thermal_dose = ThermalDose {
            current_dose: Array3::zeros(grid_dims),
            dose_rate: Array3::zeros(grid_dims),
            time_to_target: Array3::from_elem(grid_dims, f64::INFINITY),
            max_safe_time: Array3::zeros(grid_dims),
        };
        let mechanical_index = MechanicalIndex {
            current_mi: 0.0,
            peak_pressure: 0.0,
            limit: 1.9,
            safety_margin: 1.0,
        };

        Self {
            temperature,
            pressure,
            thermal_dose,
            mechanical_index,
            _perfusion_rate: perfusion_rate,
            frequency,
            thresholds: SafetyThresholds::default(),
        }
    }

    /// Update safety monitoring with new field data
    pub fn update_fields(
        &mut self,
        temperature: &Array3<f64>,
        pressure: &Array3<f64>,
        time_step: f64,
    ) -> KwaversResult<()> {
        self.temperature.assign(temperature);
        self.pressure.assign(pressure);
        self.update_thermal_dose(time_step);
        self.update_mechanical_index();
        self.check_safety_limits()?;
        Ok(())
    }
}
