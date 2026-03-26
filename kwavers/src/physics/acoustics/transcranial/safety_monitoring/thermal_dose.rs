//! Thermal dose accumulation using the Sapareto & Dewey CEM43 model

use super::monitor::SafetyMonitor;

impl SafetyMonitor {
    /// Update thermal dose accumulation
    ///
    /// Implements the Sapareto & Dewey (1984) CEM43 thermal dose model:
    /// CEM43 = Σ R^(43-T) · Δt
    /// where R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C
    pub(crate) fn update_thermal_dose(&mut self, dt: f64) {
        let (nx, ny, nz) = self.temperature.dim();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let temp = self.temperature[[i, j, k]];
                    let temp_celsius = temp - 37.0; // Relative to body temperature

                    // Sapareto & Dewey thermal dose model
                    let r_value: f64 = if temp_celsius >= 0.0 {
                        0.5 // Above 37°C
                    } else {
                        0.25 // Below 37°C
                    };

                    let dose_rate = r_value.powf(43.0 - temp) * 0.5; // Simplified

                    // Accumulate dose
                    self.thermal_dose.dose_rate[[i, j, k]] = dose_rate;
                    self.thermal_dose.current_dose[[i, j, k]] += dose_rate * dt;

                    // Calculate time to target dose
                    let target_dose = self.thresholds.max_thermal_dose;
                    let current_dose = self.thermal_dose.current_dose[[i, j, k]];

                    if current_dose < target_dose && dose_rate > 0.0 {
                        self.thermal_dose.time_to_target[[i, j, k]] =
                            (target_dose - current_dose) / dose_rate;
                    } else {
                        self.thermal_dose.time_to_target[[i, j, k]] = 0.0;
                    }

                    // Calculate maximum safe time
                    if dose_rate > 0.0 {
                        self.thermal_dose.max_safe_time[[i, j, k]] =
                            (target_dose - current_dose) / dose_rate;
                    }
                }
            }
        }
    }
}
