//! TranscranialSafetyMonitor struct and field update orchestration

use super::types::{MechanicalIndex, SafetyThresholds, TranscranialSafetyDose};
use kwavers_core::constants::medical::MI_LIMIT_SOFT_TISSUE;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// Safety monitoring system for tFUS
#[derive(Debug)]
pub struct TranscranialSafetyMonitor {
    /// Temperature field (°C)
    pub(crate) temperature: Array3<f64>,
    /// Acoustic pressure field (Pa)
    pub(crate) pressure: Array3<f64>,
    /// Thermal dose accumulation
    pub(crate) thermal_dose: TranscranialSafetyDose,
    /// Reusable checked CEM43 increments.
    pub(crate) thermal_dose_increments: Array3<f64>,
    /// Mechanical index
    pub(crate) mechanical_index: MechanicalIndex,
    /// Tissue perfusion rate (1/s)
    pub(crate) _perfusion_rate: f64,
    /// Acoustic frequency (Hz)
    pub(crate) frequency: f64,
    /// Safety thresholds
    pub(crate) thresholds: SafetyThresholds,
}

impl TranscranialSafetyMonitor {
    /// Create new safety monitor
    #[must_use]
    pub fn new(grid_dims: (usize, usize, usize), perfusion_rate: f64, frequency: f64) -> Self {
        let shape = [grid_dims.0, grid_dims.1, grid_dims.2];
        let temperature = Array3::from_elem(shape, BODY_TEMPERATURE_C);
        let pressure = Array3::zeros(shape);
        let thermal_dose = TranscranialSafetyDose {
            current_dose: Array3::zeros(shape),
            dose_rate: Array3::zeros(shape),
            time_to_target: Array3::from_elem(shape, f64::INFINITY),
            max_safe_time: Array3::zeros(shape),
        };
        let mechanical_index = MechanicalIndex {
            current_mi: 0.0,
            peak_pressure: 0.0,
            limit: MI_LIMIT_SOFT_TISSUE,
            safety_margin: 1.0,
        };

        Self {
            temperature,
            pressure,
            thermal_dose,
            thermal_dose_increments: Array3::zeros(shape),
            mechanical_index,
            _perfusion_rate: perfusion_rate,
            frequency,
            thresholds: SafetyThresholds::default(),
        }
    }

    /// Update safety monitoring with new field data
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update_fields(
        &mut self,
        temperature: &Array3<f64>,
        pressure: &Array3<f64>,
        time_step: f64,
    ) -> KwaversResult<()> {
        if temperature.shape() != self.temperature.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "transcranial temperature shape {:?} does not match monitor shape {:?}",
                temperature.shape(),
                self.temperature.shape()
            )));
        }
        if pressure.shape() != self.pressure.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "transcranial pressure shape {:?} does not match monitor shape {:?}",
                pressure.shape(),
                self.pressure.shape()
            )));
        }
        self.update_thermal_dose(temperature, time_step)?;
        self.temperature.assign(temperature);
        self.pressure.assign(pressure);
        self.update_mechanical_index();
        self.check_safety_limits()?;
        Ok(())
    }
}
