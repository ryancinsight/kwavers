//! Treatment metrics and outcome assessment
//!
//! Provides metrics for evaluating therapy effectiveness and safety.

use ndarray::Array3;

/// Treatment outcome metrics
#[derive(Debug, Clone)]
pub struct TreatmentMetrics {
    /// Thermal dose (CEM43) [equivalent minutes]
    pub thermal_dose: f64,
    /// Cavitation dose (cumulative)
    pub cavitation_dose: f64,
    /// Lesion volume [m³]
    pub lesion_volume: f64,
    /// Peak temperature reached [K]
    pub peak_temperature: f64,
    /// Treatment efficiency (0-1)
    pub efficiency: f64,
    /// Safety metric (0-1, higher is safer)
    pub safety_index: f64,
}

impl Default for TreatmentMetrics {
    fn default() -> Self {
        Self {
            thermal_dose: 0.0,
            cavitation_dose: 0.0,
            lesion_volume: 0.0,
            peak_temperature: 310.15, // 37°C baseline
            efficiency: 0.0,
            safety_index: 1.0,
        }
    }
}

impl TreatmentMetrics {
    /// Calculate thermal dose using CEM43 model
    /// CEM43 = Σ R^(43-T) * Δt
    /// where R = 0.5 for T > 43°C, R = 0.25 for T < 43°C
    #[must_use]
    pub fn calculate_thermal_dose(temperature: &Array3<f64>, dt: f64) -> f64 {
        const REFERENCE_TEMP: f64 = 316.15; // 43°C in Kelvin

        temperature
            .iter()
            .map(|&t| {
                if t > REFERENCE_TEMP {
                    0.5_f64.powf(REFERENCE_TEMP - t) * dt / 60.0 // Convert to minutes
                } else if t > 310.15 {
                    // Above body temperature
                    0.25_f64.powf(REFERENCE_TEMP - t) * dt / 60.0
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Calculate cavitation dose (cumulative cavitation activity)
    #[must_use]
    pub fn calculate_cavitation_dose(cavitation_field: &Array3<bool>, dt: f64) -> f64 {
        cavitation_field.iter().filter(|&&cav| cav).count() as f64 * dt
    }

    /// Calculate lesion volume from thermal dose field
    #[must_use]
    pub fn calculate_lesion_volume(
        thermal_dose_field: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> f64 {
        const LESION_THRESHOLD: f64 = 240.0; // 240 CEM43 minutes for complete necrosis

        let voxel_volume = dx * dy * dz;
        thermal_dose_field
            .iter()
            .filter(|&&dose| dose > LESION_THRESHOLD)
            .count() as f64
            * voxel_volume
    }

    /// Update peak temperature
    pub fn update_peak_temperature(&mut self, temperature: &Array3<f64>) {
        if let Some(&max_temp) = temperature.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if max_temp > self.peak_temperature {
                self.peak_temperature = max_temp;
            }
        }
    }

    /// Calculate treatment efficiency
    /// Efficiency = (achieved dose / target dose) * `safety_index`
    pub fn calculate_efficiency(&mut self, target_dose: f64) {
        if target_dose > 0.0 {
            self.efficiency = (self.thermal_dose / target_dose).min(1.0) * self.safety_index;
        } else {
            self.efficiency = 0.0;
        }
    }

    /// Calculate safety index based on temperature and cavitation
    pub fn calculate_safety_index(&mut self) {
        const MAX_SAFE_TEMP: f64 = 333.15; // 60°C
        const MAX_SAFE_CAVITATION: f64 = 100.0; // Arbitrary units

        let temp_safety = if self.peak_temperature < MAX_SAFE_TEMP {
            1.0
        } else {
            (MAX_SAFE_TEMP / self.peak_temperature).max(0.0)
        };

        let cav_safety = if self.cavitation_dose < MAX_SAFE_CAVITATION {
            1.0
        } else {
            (MAX_SAFE_CAVITATION / self.cavitation_dose).max(0.0)
        };

        self.safety_index = (temp_safety * cav_safety).sqrt();
    }

    /// Check if treatment goals are met
    #[must_use]
    pub fn is_successful(&self, target_dose: f64, min_efficiency: f64) -> bool {
        self.thermal_dose >= target_dose && self.efficiency >= min_efficiency
    }

    /// Get treatment summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Treatment Metrics:\n\
             - Thermal Dose: {:.1} CEM43 minutes\n\
             - Cavitation Dose: {:.1} s\n\
             - Lesion Volume: {:.2} mm³\n\
             - Peak Temperature: {:.1}°C\n\
             - Efficiency: {:.1}%\n\
             - Safety Index: {:.2}",
            self.thermal_dose,
            self.cavitation_dose,
            self.lesion_volume * 1e9,       // Convert to mm³
            self.peak_temperature - 273.15, // Convert to °C
            self.efficiency * 100.0,
            self.safety_index
        )
    }
}
