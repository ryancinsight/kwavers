use crate::core::constants::medical::MI_LIMIT_SOFT_TISSUE;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use ndarray::Array3;

#[derive(Debug, Clone, Default)]
pub struct DomainTreatmentMetrics {
    pub thermal_dose: f64,
    pub cavitation_dose: f64,
    pub peak_temperature: f64,
    pub safety_index: f64,
    pub efficiency: f64,
}

impl DomainTreatmentMetrics {
    #[must_use]
    pub fn calculate_thermal_dose(temperature: &Array3<f64>, dt: f64) -> f64 {
        let max_dose_rate = temperature.iter().fold(0.0f64, |acc, &t| {
            let rate = if t > 43.0 {
                (t - 43.0).exp2()
            } else if t > BODY_TEMPERATURE_C {
                4.0_f64.powf(t - 43.0)
            } else {
                0.0
            };
            acc.max(rate)
        });
        max_dose_rate * dt
    }

    #[must_use]
    pub fn calculate_cavitation_dose(cavitation_field: &Array3<f64>, dt: f64) -> f64 {
        cavitation_field.sum() * dt
    }

    pub fn update_peak_temperature(&mut self, temperature: &Array3<f64>) {
        let max_t = temperature.iter().copied().fold(0.0_f64, f64::max);
        if max_t > self.peak_temperature {
            self.peak_temperature = max_t;
        }
    }

    pub fn calculate_safety_index(&mut self) {
        if self.peak_temperature > 90.0 {
            self.safety_index = 0.0;
        } else {
            self.safety_index = 1.0;
        }
    }

    pub fn calculate_efficiency(&mut self, target_dose: f64) {
        if target_dose > 0.0 {
            self.efficiency = (self.thermal_dose / target_dose).min(1.0);
        }
    }

    #[must_use]
    pub fn is_successful(&self, target_dose: f64, threshold: f64) -> bool {
        self.thermal_dose >= target_dose * threshold
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Dose: {:.1} CEM43, Peak T: {:.1} C, Safety: {:.2}",
            self.thermal_dose, self.peak_temperature, self.safety_index
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainTherapyMechanism {
    Thermal,
    Mechanical,
    Combined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainTherapyModality {
    HIFU,
    LIFU,
    Histotripsy,
    BBBOpening,
    Sonodynamic,
    Sonoporation,
}

impl DomainTherapyModality {
    #[must_use]
    pub fn has_thermal_effects(&self) -> bool {
        matches!(self, Self::HIFU | Self::Sonodynamic)
    }

    #[must_use]
    pub fn has_cavitation(&self) -> bool {
        matches!(
            self,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation | Self::Sonodynamic
        )
    }

    #[must_use]
    pub fn primary_mechanism(&self) -> DomainTherapyMechanism {
        match self {
            Self::HIFU => DomainTherapyMechanism::Thermal,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation => {
                DomainTherapyMechanism::Mechanical
            }
            Self::LIFU | Self::Sonodynamic => DomainTherapyMechanism::Combined,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DomainTherapyParameters {
    pub frequency: f64,
    pub peak_negative_pressure: f64,
    pub treatment_duration: f64,
    pub mechanical_index: f64,
    pub duty_cycle: f64,
    pub prf: f64,
}

impl DomainTherapyParameters {
    #[must_use]
    pub fn new(frequency: f64, pressure: f64, duration: f64) -> Self {
        let mut params = Self {
            frequency,
            peak_negative_pressure: pressure,
            treatment_duration: duration,
            mechanical_index: 0.0,
            duty_cycle: 1.0,
            prf: 0.0,
        };
        params.calculate_mechanical_index();
        params
    }

    #[must_use]
    pub fn hifu() -> Self {
        Self::new(1.5 * MHZ_TO_HZ, 2.0 * MPA_TO_PA, 5.0)
    }

    pub fn calculate_mechanical_index(&mut self) {
        if self.frequency > 0.0 {
            self.mechanical_index =
                (self.peak_negative_pressure / MPA_TO_PA) / (self.frequency / MHZ_TO_HZ).sqrt();
        }
    }

    #[must_use]
    pub fn validate_safety(&self) -> bool {
        if self.mechanical_index > MI_LIMIT_SOFT_TISSUE {
            return false;
        }
        if self.treatment_duration > 3600.0 {
            return false;
        }
        true
    }
}
