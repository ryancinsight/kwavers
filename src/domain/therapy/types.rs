use ndarray::Array3;

#[derive(Debug, Clone, Default)]
pub struct TreatmentMetrics {
    pub thermal_dose: f64,
    pub cavitation_dose: f64,
    pub peak_temperature: f64,
    pub safety_index: f64,
    pub efficiency: f64,
}

impl TreatmentMetrics {
    pub fn calculate_thermal_dose(temperature: &Array3<f64>, dt: f64) -> f64 {
        let max_dose_rate = temperature.iter().fold(0.0f64, |acc, &t| {
            let rate = if t > 43.0 {
                2.0_f64.powf(t - 43.0)
            } else if t > 37.0 {
                4.0_f64.powf(t - 43.0)
            } else {
                0.0
            };
            acc.max(rate)
        });
        max_dose_rate * dt
    }

    pub fn calculate_cavitation_dose(cavitation_field: &Array3<f64>, dt: f64) -> f64 {
        cavitation_field.sum() * dt
    }

    pub fn update_peak_temperature(&mut self, temperature: &Array3<f64>) {
        let max_t = temperature.iter().cloned().fold(0.0_f64, f64::max);
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

    pub fn is_successful(&self, target_dose: f64, threshold: f64) -> bool {
        self.thermal_dose >= target_dose * threshold
    }

    pub fn summary(&self) -> String {
        format!(
            "Dose: {:.1} CEM43, Peak T: {:.1} C, Safety: {:.2}",
            self.thermal_dose, self.peak_temperature, self.safety_index
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyMechanism {
    Thermal,
    Mechanical,
    Combined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyModality {
    HIFU,
    LIFU,
    Histotripsy,
    BBBOpening,
    Sonodynamic,
    Sonoporation,
}

impl TherapyModality {
    pub fn has_thermal_effects(&self) -> bool {
        matches!(self, Self::HIFU | Self::Sonodynamic)
    }

    pub fn has_cavitation(&self) -> bool {
        matches!(
            self,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation | Self::Sonodynamic
        )
    }

    pub fn primary_mechanism(&self) -> TherapyMechanism {
        match self {
            Self::HIFU => TherapyMechanism::Thermal,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation => TherapyMechanism::Mechanical,
            Self::LIFU | Self::Sonodynamic => TherapyMechanism::Combined,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TherapyParameters {
    pub frequency: f64,
    pub peak_negative_pressure: f64,
    pub treatment_duration: f64,
    pub mechanical_index: f64,
    pub duty_cycle: f64,
    pub prf: f64,
}

impl TherapyParameters {
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

    pub fn hifu() -> Self {
        Self::new(1.5e6, 2.0e6, 5.0)
    }

    pub fn calculate_mechanical_index(&mut self) {
        if self.frequency > 0.0 {
            self.mechanical_index = (self.peak_negative_pressure / 1e6) / (self.frequency / 1e6).sqrt();
        }
    }

    pub fn validate_safety(&self) -> bool {
        if self.mechanical_index > 1.9 {
            return false;
        }
        if self.treatment_duration > 3600.0 {
            return false;
        }
        true
    }
}

