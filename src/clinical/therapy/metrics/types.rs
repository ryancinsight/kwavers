//! Treatment metrics

use ndarray::Array3;

/// Metrics for tracking treatment progress and outcome
#[derive(Debug, Clone, Default)]
pub struct TreatmentMetrics {
    /// Cumulative thermal dose (CEM43)
    pub thermal_dose: f64,
    /// Cumulative cavitation dose
    pub cavitation_dose: f64,
    /// Peak temperature reached (°C)
    pub peak_temperature: f64,
    /// Safety index (0-1)
    pub safety_index: f64,
    /// Treatment efficiency
    pub efficiency: f64,
}

impl TreatmentMetrics {
    /// Calculate thermal dose accumulation for a time step
    ///
    /// Uses CEM43 formula: R^(T - 43) * dt
    ///
    /// Returns the maximum thermal dose rate in the volume multiplied by dt.
    /// Standard CEM43: R=0.5 (2^(T-43)) for T>43°C, R=0.25 (4^(T-43)) for 37°C<T<43°C
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

    /// Calculate cavitation dose from field
    pub fn calculate_cavitation_dose(cavitation_field: &Array3<f64>, dt: f64) -> f64 {
        // Sum of cavitation activity
        cavitation_field.sum() * dt
    }

    /// Update peak temperature
    pub fn update_peak_temperature(&mut self, temperature: &Array3<f64>) {
        let max_t = temperature.iter().cloned().fold(0.0_f64, f64::max);
        if max_t > self.peak_temperature {
            self.peak_temperature = max_t;
        }
    }

    /// Calculate safety index
    pub fn calculate_safety_index(&mut self) {
        // Placeholder safety logic
        if self.peak_temperature > 90.0 {
            // Boiling
            self.safety_index = 0.0;
        } else {
            self.safety_index = 1.0;
        }
    }

    /// Calculate efficiency
    pub fn calculate_efficiency(&mut self, target_dose: f64) {
        if target_dose > 0.0 {
            self.efficiency = (self.thermal_dose / target_dose).min(1.0);
        }
    }

    /// Check if successful
    pub fn is_successful(&self, target_dose: f64, threshold: f64) -> bool {
        self.thermal_dose >= target_dose * threshold
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Dose: {:.1} CEM43, Peak T: {:.1} C, Safety: {:.2}",
            self.thermal_dose, self.peak_temperature, self.safety_index
        )
    }
}
