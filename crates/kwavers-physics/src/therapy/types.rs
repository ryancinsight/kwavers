use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::response::thermal::Cem43;
use kwavers_core::constants::medical::MI_LIMIT_SOFT_TISSUE;
use kwavers_core::constants::numerical::{
    MHZ_TO_HZ, MPA_TO_PA, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
};
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_C, KELVIN_OFFSET_C};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

#[derive(Debug, Clone, Default)]
pub struct DomainTreatmentMetrics {
    pub thermal_dose: f64,
    pub cavitation_dose: f64,
    pub peak_temperature: f64,
    pub safety_index: f64,
    pub efficiency: f64,
}

impl DomainTreatmentMetrics {
    /// Return the maximum per-voxel CEM43 increment for one time step.
    ///
    /// Temperatures at or below the consumer-owned body-temperature gate do
    /// not contribute to this aggregate treatment metric.
    ///
    /// # Errors
    ///
    /// Returns an error when Asclepius rejects the time step or an absolute
    /// temperature.
    pub fn calculate_thermal_dose(temperature: &Array3<f64>, dt: f64) -> KwaversResult<f64> {
        let law = Cem43::canonical();
        let step = Time::from_base(dt);
        let mut maximum = 0.0_f64;
        for &temperature_c in temperature.iter() {
            let increment = law
                .increment(
                    ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C),
                    step,
                )
                .map_err(|source| {
                    KwaversError::InvalidInput(format!(
                        "treatment CEM43 observation is invalid: {source}"
                    ))
                })?
                .get()
                .into_base()
                / SECONDS_PER_MINUTE;
            if temperature_c > BODY_TEMPERATURE_C {
                maximum = maximum.max(increment);
            }
        }
        Ok(maximum)
    }

    #[must_use]
    pub fn calculate_cavitation_dose(cavitation_field: &Array3<f64>, dt: f64) -> f64 {
        cavitation_field.iter().copied().sum::<f64>() * dt
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
        if self.treatment_duration > SECONDS_PER_HOUR {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::DomainTreatmentMetrics;
    use leto::Array3;

    #[test]
    fn maximum_thermal_dose_uses_canonical_equivalent_minutes() {
        let temperature = Array3::from_shape_vec([3, 1, 1], vec![37.0, 43.0, 44.0])
            .expect("valid temperature field");
        let dose = DomainTreatmentMetrics::calculate_thermal_dose(&temperature, 60.0)
            .expect("valid CEM43 observation");
        assert_eq!(dose, 2.0);
    }

    #[test]
    fn maximum_thermal_dose_rejects_invalid_observations() {
        let invalid = Array3::from_elem([1, 1, 1], f64::NAN);
        assert!(DomainTreatmentMetrics::calculate_thermal_dose(&invalid, 60.0).is_err());

        let temperature = Array3::from_elem([1, 1, 1], 44.0);
        assert!(DomainTreatmentMetrics::calculate_thermal_dose(&temperature, 0.0).is_err());
    }
}
