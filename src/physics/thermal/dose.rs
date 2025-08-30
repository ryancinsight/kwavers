// thermal/dose.rs - Thermal dose calculation

use super::constants;
use ndarray::{Array3, Zip};

/// Thermal dose using CEM43 model
#[derive(Debug)]
pub struct ThermalDose;

impl ThermalDose {
    /// Calculate CEM43 thermal dose
    pub fn cem43(temperature: &Array3<f64>, time_minutes: f64) -> Array3<f64> {
        let mut dose = Array3::zeros(temperature.raw_dim());

        Zip::from(&mut dose).and(temperature).for_each(|d, &t| {
            let t_celsius = t - constants::CELSIUS_TO_KELVIN;

            if t_celsius > constants::CEM43_REFERENCE_TEMP {
                let r: f64 = if t_celsius > 43.0 { 0.5 } else { 0.25 };
                *d = time_minutes * r.powf(t_celsius - constants::CEM43_REFERENCE_TEMP);
            }
        });

        dose
    }
}

/// Thermal dose calculator with accumulation
#[derive(Debug)]
pub struct ThermalDoseCalculator {
    accumulated_dose: Array3<f64>,
}

impl ThermalDoseCalculator {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            accumulated_dose: Array3::zeros(shape),
        }
    }

    pub fn update(&mut self, temperature: &Array3<f64>, dt_minutes: f64) {
        let dose_increment = ThermalDose::cem43(temperature, dt_minutes);
        self.accumulated_dose += &dose_increment;
    }

    pub fn dose(&self) -> &Array3<f64> {
        &self.accumulated_dose
    }
}
