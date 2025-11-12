//! Safety Monitoring for Transcranial Focused Ultrasound
//!
//! Real-time monitoring of thermal and mechanical indices during tFUS treatments
//! to ensure patient safety and treatment efficacy.

use crate::error::KwaversResult;
use ndarray::Array3;

/// Thermal dose accumulation (CEM43)
#[derive(Debug, Clone)]
pub struct ThermalDose {
    /// Current thermal dose (CEM43)
    pub current_dose: Array3<f64>,
    /// Dose rate (CEM43/s)
    pub dose_rate: Array3<f64>,
    /// Time to reach target dose (s)
    pub time_to_target: Array3<f64>,
    /// Maximum safe exposure time (s)
    pub max_safe_time: Array3<f64>,
}

/// Mechanical index monitoring
#[derive(Debug, Clone)]
pub struct MechanicalIndex {
    /// Current MI value
    pub current_mi: f64,
    /// Peak pressure (MPa)
    pub peak_pressure: f64,
    /// MI limit for tissue type
    pub limit: f64,
    /// Safety margin (1.0 = at limit, <1.0 = safe)
    pub safety_margin: f64,
}

/// Safety monitoring system for tFUS
pub struct SafetyMonitor {
    /// Temperature field (°C)
    temperature: Array3<f64>,
    /// Acoustic pressure field (Pa)
    pressure: Array3<f64>,
    /// Thermal dose accumulation
    thermal_dose: ThermalDose,
    /// Mechanical index
    mechanical_index: MechanicalIndex,
    /// Tissue perfusion rate (1/s)
    perfusion_rate: f64,
    /// Acoustic frequency (Hz)
    frequency: f64,
    /// Safety thresholds
    thresholds: SafetyThresholds,
}

#[derive(Debug, Clone)]
pub struct SafetyThresholds {
    pub max_temperature: f64,      // °C
    pub max_thermal_dose: f64,     // CEM43
    pub max_mechanical_index: f64, // MI
    pub max_power_density: f64,    // W/cm²
}

impl Default for SafetyThresholds {
    fn default() -> Self {
        Self {
            max_temperature: 43.0,    // Brain tissue limit
            max_thermal_dose: 240.0,   // CEM43 for brain
            max_mechanical_index: 1.9, // FDA limit
            max_power_density: 100.0,  // W/cm²
        }
    }
}

impl SafetyMonitor {
    /// Create new safety monitor
    pub fn new(
        grid_dims: (usize, usize, usize),
        perfusion_rate: f64,
        frequency: f64,
    ) -> Self {
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
            perfusion_rate,
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
        // Update temperature field
        self.temperature.assign(temperature);

        // Update pressure field
        self.pressure.assign(pressure);

        // Update thermal dose
        self.update_thermal_dose(time_step);

        // Update mechanical index
        self.update_mechanical_index();

        // Check safety limits
        self.check_safety_limits()?;

        Ok(())
    }

    /// Update thermal dose accumulation
    fn update_thermal_dose(&mut self, dt: f64) {
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

    /// Update mechanical index
    fn update_mechanical_index(&mut self) {
        // Find peak pressure
        let peak_pressure = self.pressure.iter()
            .map(|&p| p.abs())
            .fold(0.0_f64, f64::max);

        self.mechanical_index.peak_pressure = peak_pressure / 1e6; // Convert to MPa

        // Calculate MI: p_peak / sqrt(f)
        // Reference: AIUM/NEMA UD 3-2004
        let mi = if self.frequency > 0.0 {
            self.mechanical_index.peak_pressure / self.frequency.sqrt() * 1e3 // kHz to Hz conversion
        } else {
            0.0
        };

        self.mechanical_index.current_mi = mi;
        self.mechanical_index.safety_margin = self.thresholds.max_mechanical_index / mi;
    }

    /// Check safety limits and return warnings
    fn check_safety_limits(&self) -> KwaversResult<()> {
        // Immediate error on temperature limit exceedance (per tests and safety policy)
        let max_temp = self
            .temperature
            .iter()
            .fold(f64::MIN, |a, &b| a.max(b));
        if max_temp > self.thresholds.max_temperature {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Temperature {:.1}°C exceeds limit {:.1}°C",
                        max_temp, self.thresholds.max_temperature
                    ),
                },
            ));
        }

        // Check thermal dose limits (strict)
        let max_dose = self.thermal_dose.current_dose.iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        if max_dose > self.thresholds.max_thermal_dose {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::ConstraintViolation {
                    message: format!("Thermal dose {:.0} CEM43 exceeds limit {:.0} CEM43",
                                   max_dose, self.thresholds.max_thermal_dose)
                }
            ));
        }

        // Check mechanical index
        if self.mechanical_index.current_mi > self.thresholds.max_mechanical_index {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::ConstraintViolation {
                    message: format!("Mechanical index {:.2} exceeds limit {:.2}",
                                   self.mechanical_index.current_mi, self.thresholds.max_mechanical_index)
                }
            ));
        }

        Ok(())
    }

    /// Get current safety status
    pub fn safety_status(&self) -> SafetyStatus {
        let max_temp = self.temperature.iter().fold(0.0_f64, |a, &b| a.max(b));
        let max_dose = self.thermal_dose.current_dose.iter()
            .fold(0.0_f64, |a, &b| a.max(b));

        SafetyStatus {
            temperature_status: SafetyLevel::from_value(max_temp, self.thresholds.max_temperature),
            thermal_dose_status: SafetyLevel::from_value(max_dose, self.thresholds.max_thermal_dose),
            mechanical_index_status: SafetyLevel::from_value(
                self.mechanical_index.current_mi,
                self.thresholds.max_mechanical_index
            ),
            overall_safety: self.overall_safety_level(),
        }
    }

    /// Calculate overall safety level
    fn overall_safety_level(&self) -> SafetyLevel {
        let status = self.safety_status();

        // Use the most restrictive safety level
        let levels = [
            status.temperature_status,
            status.thermal_dose_status,
            status.mechanical_index_status,
        ];

        levels.iter().cloned().max().unwrap_or(SafetyLevel::Safe)
    }

    /// Get treatment progress towards target dose
    pub fn treatment_progress(&self, target_dose: f64) -> TreatmentProgress {
        let max_current_dose = self.thermal_dose.current_dose.iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        let progress = (max_current_dose / target_dose).min(1.0);

        let estimated_time_remaining = if progress < 1.0 {
            let max_time_to_target = self.thermal_dose.time_to_target.iter()
                .fold(0.0_f64, |a, &b| a.max(b));
            max_time_to_target
        } else {
            0.0
        };

        TreatmentProgress {
            dose_progress: progress,
            estimated_time_remaining,
            current_max_dose: max_current_dose,
            target_dose,
        }
    }

    /// Generate safety report
    pub fn safety_report(&self) -> SafetyReport {
        let status = self.safety_status();
        let progress = self.treatment_progress(self.thresholds.max_thermal_dose);

        SafetyReport {
            status,
            progress,
            thermal_dose: self.thermal_dose.clone(),
            mechanical_index: self.mechanical_index.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate safety recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let status = self.safety_status();

        match status.temperature_status {
            SafetyLevel::Critical => {
                recommendations.push("CRITICAL: Temperature exceeds safe limits. Stop treatment immediately.".to_string());
            }
            SafetyLevel::Warning => {
                recommendations.push("WARNING: Temperature approaching limit. Reduce power.".to_string());
            }
            _ => {}
        }

        match status.thermal_dose_status {
            SafetyLevel::Critical => {
                recommendations.push("CRITICAL: Thermal dose exceeds safe limits. Stop treatment.".to_string());
            }
            SafetyLevel::Warning => {
                recommendations.push("WARNING: Thermal dose approaching limit. Monitor closely.".to_string());
            }
            _ => {}
        }

        match status.mechanical_index_status {
            SafetyLevel::Critical => {
                recommendations.push("CRITICAL: Mechanical index exceeds safety limit. Reduce acoustic power.".to_string());
            }
            SafetyLevel::Warning => {
                recommendations.push("WARNING: Mechanical index approaching limit. Reduce pressure amplitude.".to_string());
            }
            _ => {}
        }

        if recommendations.is_empty() {
            recommendations.push("All parameters within safe limits. Treatment may continue.".to_string());
        }

        recommendations
    }
}

/// Safety level classification
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum SafetyLevel {
    Safe = 0,
    Monitor = 1,
    Warning = 2,
    Critical = 3,
}

impl SafetyLevel {
    fn from_value(current: f64, limit: f64) -> Self {
        let ratio = current / limit;
        if ratio >= 1.0 {
            SafetyLevel::Critical
        } else if ratio >= 0.9 {
            SafetyLevel::Warning
        } else if ratio >= 0.8 {
            SafetyLevel::Monitor
        } else {
            SafetyLevel::Safe
        }
    }
}

/// Overall safety status
#[derive(Debug, Clone)]
pub struct SafetyStatus {
    pub temperature_status: SafetyLevel,
    pub thermal_dose_status: SafetyLevel,
    pub mechanical_index_status: SafetyLevel,
    pub overall_safety: SafetyLevel,
}

/// Treatment progress information
#[derive(Debug, Clone)]
pub struct TreatmentProgress {
    pub dose_progress: f64,           // 0.0 to 1.0
    pub estimated_time_remaining: f64, // seconds
    pub current_max_dose: f64,        // CEM43
    pub target_dose: f64,             // CEM43
}

/// Comprehensive safety report
#[derive(Debug)]
pub struct SafetyReport {
    pub status: SafetyStatus,
    pub progress: TreatmentProgress,
    pub thermal_dose: ThermalDose,
    pub mechanical_index: MechanicalIndex,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_monitor_creation() {
        let monitor = SafetyMonitor::new((16, 16, 16), 0.01, 650e3);
        assert_eq!(monitor.temperature.dim(), (16, 16, 16));
    }

    #[test]
    fn test_safety_level_classification() {
        assert_eq!(SafetyLevel::from_value(0.5, 1.0), SafetyLevel::Safe);
        assert_eq!(SafetyLevel::from_value(0.85, 1.0), SafetyLevel::Monitor);
        assert_eq!(SafetyLevel::from_value(0.95, 1.0), SafetyLevel::Warning);
        assert_eq!(SafetyLevel::from_value(1.1, 1.0), SafetyLevel::Critical);
    }

    #[test]
    fn test_mechanical_index_calculation() {
        let mut monitor = SafetyMonitor::new((8, 8, 8), 0.01, 1e6);
        let temperature = Array3::from_elem((8, 8, 8), 37.0);
        let mut pressure = Array3::zeros((8, 8, 8));
        pressure[[4, 4, 4]] = 1e6; // 1 MPa

        let result = monitor.update_fields(&temperature, &pressure, 0.1);
        assert!(result.is_ok());

        // MI should be approximately 1.0 for 1 MPa at 1 MHz
        assert!(monitor.mechanical_index.current_mi > 0.0);
    }

    #[test]
    fn test_thermal_dose_accumulation() {
        let mut monitor = SafetyMonitor::new((4, 4, 4), 0.01, 650e3);
        let mut temperature = Array3::from_elem((4, 4, 4), 37.0);
        temperature[[2, 2, 2]] = 42.0; // Hot spot below safety limit (43°C)
        let pressure = Array3::zeros((4, 4, 4));

        let result = monitor.update_fields(&temperature, &pressure, 1.0);
        assert!(result.is_ok(), "Update should succeed with safe temperature");

        // Thermal dose should accumulate
        assert!(monitor.thermal_dose.current_dose[[2, 2, 2]] > 0.0);
    }

    #[test]
    fn test_safety_limit_checking() {
        let mut monitor = SafetyMonitor::new((4, 4, 4), 0.01, 650e3);
        let mut temperature = Array3::from_elem((4, 4, 4), 37.0);
        temperature[[2, 2, 2]] = 50.0; // Above limit
        let pressure = Array3::zeros((4, 4, 4));

        let result = monitor.update_fields(&temperature, &pressure, 1.0);
        assert!(result.is_err()); // Should fail safety check
    }
}
