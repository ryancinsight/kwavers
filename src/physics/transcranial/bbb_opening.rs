//! Blood-Brain Barrier Opening with Focused Ultrasound
//!
//! Models the mechanisms of BBB opening for targeted drug delivery using
//! low-intensity focused ultrasound with microbubbles.

use crate::error::KwaversResult;
use ndarray::Array3;

/// Permeability enhancement data
#[derive(Debug, Clone)]
pub struct PermeabilityEnhancement {
    /// Local permeability increase factor
    pub permeability_factor: Array3<f64>,
    /// Opening duration (seconds)
    pub opening_duration: Array3<f64>,
    /// BBB recovery time (hours)
    pub recovery_time: Array3<f64>,
    /// Microbubble concentration effect
    pub microbubble_effect: Array3<f64>,
}

/// BBB opening simulation
#[derive(Debug)]
pub struct BBBOpening {
    /// Acoustic pressure field (Pa)
    acoustic_pressure: Array3<f64>,
    /// Microbubble concentration (bubbles/mL)
    microbubble_concentration: Array3<f64>,
    /// Treatment parameters
    parameters: BBBParameters,
    /// Permeability enhancement results
    permeability: PermeabilityEnhancement,
}

#[derive(Debug, Clone)]
pub struct BBBParameters {
    /// Acoustic frequency (Hz)
    pub frequency: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Duty cycle (%)
    pub duty_cycle: f64,
    /// Treatment duration (s)
    pub duration: f64,
    /// Mechanical index target
    pub target_mi: f64,
    /// Microbubble size distribution (mean radius, std dev) in μm
    pub bubble_size: (f64, f64),
}

impl Default for BBBParameters {
    fn default() -> Self {
        Self {
            frequency: 1.0e6,        // 1 MHz
            prf: 1.0,                // 1 Hz
            duty_cycle: 10.0,        // 10%
            duration: 120.0,         // 2 minutes
            target_mi: 0.3,          // Low MI for BBB opening
            bubble_size: (1.5, 0.3), // 1.5 ± 0.3 μm
        }
    }
}

impl BBBOpening {
    /// Create new BBB opening simulation
    pub fn new(
        acoustic_pressure: Array3<f64>,
        microbubble_concentration: Array3<f64>,
        parameters: BBBParameters,
    ) -> Self {
        let dims = acoustic_pressure.dim();
        let permeability = PermeabilityEnhancement {
            permeability_factor: Array3::zeros(dims),
            opening_duration: Array3::zeros(dims),
            recovery_time: Array3::zeros(dims),
            microbubble_effect: Array3::zeros(dims),
        };

        Self {
            acoustic_pressure,
            microbubble_concentration,
            parameters,
            permeability,
        }
    }

    /// Simulate BBB opening process
    pub fn simulate_opening(&mut self) -> KwaversResult<()> {
        println!("Simulating BBB opening with parameters:");
        println!("  Frequency: {:.1} MHz", self.parameters.frequency / 1e6);
        println!("  MI target: {:.2}", self.parameters.target_mi);
        println!("  Duration: {:.1} s", self.parameters.duration);

        let (nx, ny, nz) = self.acoustic_pressure.dim();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let pressure = self.acoustic_pressure[[i, j, k]];
                    let bubble_conc = self.microbubble_concentration[[i, j, k]];

                    // Calculate local permeability enhancement
                    let enhancement =
                        self.calculate_permeability_enhancement(pressure, bubble_conc);
                    self.permeability.permeability_factor[[i, j, k]] = enhancement;

                    // Calculate opening duration
                    let duration = self.calculate_opening_duration(pressure, bubble_conc);
                    self.permeability.opening_duration[[i, j, k]] = duration;

                    // Calculate recovery time
                    let recovery = self.calculate_recovery_time(pressure, enhancement);
                    self.permeability.recovery_time[[i, j, k]] = recovery;

                    // Calculate microbubble effect
                    let effect = self.calculate_microbubble_effect(bubble_conc, pressure);
                    self.permeability.microbubble_effect[[i, j, k]] = effect;
                }
            }
        }

        Ok(())
    }

    /// Calculate permeability enhancement factor
    fn calculate_permeability_enhancement(&self, pressure: f64, bubble_conc: f64) -> f64 {
        // Enhanced permeability model based on microbubble oscillations
        // Reference: Choi et al. (2011) "Noninvasive and transient blood-brain barrier opening"

        let mi = self.calculate_mechanical_index(pressure);

        // Base enhancement from stable cavitation
        let base_enhancement = if mi > 0.1 && mi < 0.5 {
            // BBB opening window
            10.0 * (mi / 0.3).powf(1.5) // Empirical relation
        } else if mi >= 0.5 {
            // Risk of damage
            50.0 * (mi / 0.5).powf(2.0)
        } else {
            1.0 // No effect
        };

        // Microbubble concentration effect
        let bubble_factor = if bubble_conc > 0.0 {
            1.0 + 0.5 * (bubble_conc / 1e6).ln().max(0.0) // Logarithmic enhancement
        } else {
            1.0
        };

        base_enhancement * bubble_factor
    }

    /// Calculate BBB opening duration
    fn calculate_opening_duration(&self, pressure: f64, bubble_conc: f64) -> f64 {
        // Duration depends on acoustic exposure and microbubble concentration
        // Reference: O'Reilly & Hynynen (2012) "Blood-brain barrier"

        let mi = self.calculate_mechanical_index(pressure);
        let base_duration = self.parameters.duration; // Treatment duration

        // Opening duration scales with MI and bubble concentration
        let mi_factor = (mi / 0.3).powf(0.5);
        let bubble_factor = 1.0 + 0.2 * (bubble_conc / 1e8).min(5.0);

        base_duration * mi_factor * bubble_factor
    }

    /// Calculate BBB recovery time
    fn calculate_recovery_time(&self, pressure: f64, enhancement: f64) -> f64 {
        // Recovery time depends on the extent of opening
        // Reference: Baseri et al. (2010) "Multi-modal MRI"

        let mi = self.calculate_mechanical_index(pressure);

        // Base recovery time (hours)
        let base_recovery = if mi < 0.3 {
            4.0 // Hours for mild opening
        } else if mi < 0.5 {
            24.0 // Hours for moderate opening
        } else {
            72.0 // Hours for extensive opening
        };

        // Enhancement factor increases recovery time
        let enhancement_factor = (enhancement / 10.0).powf(0.3);

        base_recovery * enhancement_factor
    }

    /// Calculate microbubble oscillation effect
    fn calculate_microbubble_effect(&self, bubble_conc: f64, pressure: f64) -> f64 {
        // Effect of microbubble concentration on BBB opening efficiency
        // Reference: Tung et al. (2011) "Optimizating microbubble"

        if bubble_conc <= 0.0 {
            return 0.0;
        }

        let mi = self.calculate_mechanical_index(pressure);

        // Optimal concentration range: 1e7 - 1e8 bubbles/mL
        let optimal_conc = 5e7;
        let conc_ratio = bubble_conc / optimal_conc;

        let conc_factor = if conc_ratio < 0.1 {
            conc_ratio * 10.0 // Sub-optimal
        } else if conc_ratio < 10.0 {
            1.0 // Optimal range
        } else {
            10.0 / conc_ratio // Too concentrated
        };

        // MI dependence
        let mi_factor = if mi > 0.1 && mi < 0.6 {
            (mi / 0.3).powf(0.7)
        } else {
            0.1 // Outside therapeutic window
        };

        conc_factor * mi_factor
    }

    /// Calculate mechanical index
    fn calculate_mechanical_index(&self, pressure: f64) -> f64 {
        // MI = p_peak / sqrt(f) in MPa and MHz
        let p_mpa = pressure / 1e6; // Convert to MPa
        let f_mhz = self.parameters.frequency / 1e6; // Convert to MHz

        p_mpa / f_mhz.sqrt()
    }

    /// Get permeability enhancement results
    pub fn permeability(&self) -> &PermeabilityEnhancement {
        &self.permeability
    }

    /// Calculate optimal treatment parameters
    pub fn optimize_parameters(&self, target_region: &[(usize, usize, usize)]) -> BBBParameters {
        // Analyze current field to optimize parameters
        let mut max_pressure: f64 = 0.0;

        for &(i, j, k) in target_region {
            if i < self.acoustic_pressure.dim().0
                && j < self.acoustic_pressure.dim().1
                && k < self.acoustic_pressure.dim().2
            {
                max_pressure = max_pressure.max(self.acoustic_pressure[[i, j, k]]);
            }
        }

        // Optimize for target MI
        let current_mi = self.calculate_mechanical_index(max_pressure);
        let _pressure_scale = self.parameters.target_mi / current_mi.max(0.01);

        let mut optimized = self.parameters.clone();
        optimized.frequency = self.parameters.frequency; // Keep frequency
                                                         // Adjust other parameters based on optimization...

        optimized
    }

    /// Generate treatment protocol
    pub fn generate_protocol(&self) -> TreatmentProtocol {
        let target_mi = self.parameters.target_mi;
        let duration = self.parameters.duration;
        let frequency = self.parameters.frequency;

        // Calculate safe exposure time
        let max_safe_time = self.calculate_max_safe_time();

        TreatmentProtocol {
            frequency,
            target_mi,
            duration: duration.min(max_safe_time),
            prf: self.parameters.prf,
            duty_cycle: self.parameters.duty_cycle,
            microbubble_dose: self.calculate_microbubble_dose(),
            safety_checks: self.generate_safety_checks(),
        }
    }

    /// Calculate maximum safe exposure time
    fn calculate_max_safe_time(&self) -> f64 {
        // Based on thermal and mechanical limits
        // Simplified: assume 10 minutes max for BBB opening
        600.0 // 10 minutes
    }

    /// Calculate optimal microbubble dose
    fn calculate_microbubble_dose(&self) -> f64 {
        // Optimal dose: 1-5 μL/kg of 1-5% microbubble solution
        // Reference: Tung et al. (2011)
        3.0 // μL/kg
    }

    /// Generate safety checks for protocol
    fn generate_safety_checks(&self) -> Vec<String> {
        vec![
            "Verify microbubble concentration in target range (1e7-1e8/mL)".to_string(),
            "Monitor acoustic power to maintain MI < 0.5".to_string(),
            "Check for cavitation signals during treatment".to_string(),
            "Monitor subject for adverse reactions".to_string(),
            "Verify BBB opening with contrast-enhanced imaging".to_string(),
        ]
    }

    /// Validate treatment safety
    pub fn validate_safety(&self) -> SafetyValidation {
        let max_mi = self
            .acoustic_pressure
            .iter()
            .map(|&p| self.calculate_mechanical_index(p))
            .fold(0.0_f64, f64::max);

        let avg_enhancement = self.permeability.permeability_factor.iter().sum::<f64>()
            / self.permeability.permeability_factor.len() as f64;

        SafetyValidation {
            max_mechanical_index: max_mi,
            average_enhancement: avg_enhancement,
            is_safe: max_mi <= 0.6 && avg_enhancement <= 100.0,
            warnings: self.generate_warnings(max_mi, avg_enhancement),
        }
    }

    /// Generate safety warnings
    fn generate_warnings(&self, max_mi: f64, avg_enhancement: f64) -> Vec<String> {
        let mut warnings = Vec::new();

        if max_mi > 0.5 {
            warnings.push(format!("High MI ({:.2}) may cause tissue damage", max_mi));
        }

        if avg_enhancement > 50.0 {
            warnings.push(format!(
                "High permeability enhancement ({:.1}x) may indicate BBB disruption",
                avg_enhancement
            ));
        }

        if max_mi < 0.1 {
            warnings.push("Low MI may result in insufficient BBB opening".to_string());
        }

        warnings
    }
}

/// Treatment protocol for BBB opening
#[derive(Debug, Clone)]
pub struct TreatmentProtocol {
    pub frequency: f64,        // Hz
    pub target_mi: f64,        // MI
    pub duration: f64,         // seconds
    pub prf: f64,              // Hz
    pub duty_cycle: f64,       // %
    pub microbubble_dose: f64, // μL/kg
    pub safety_checks: Vec<String>,
}

/// Safety validation results
#[derive(Debug)]
pub struct SafetyValidation {
    pub max_mechanical_index: f64,
    pub average_enhancement: f64,
    pub is_safe: bool,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbb_opening_creation() {
        let pressure = Array3::from_elem((8, 8, 8), 1e5); // 100 kPa
        let bubbles = Array3::from_elem((8, 8, 8), 1e7); // 10^7 bubbles/mL
        let params = BBBParameters::default();

        let bbb = BBBOpening::new(pressure, bubbles, params);
        assert_eq!(bbb.acoustic_pressure.dim(), (8, 8, 8));
    }

    #[test]
    fn test_permeability_calculation() {
        let pressure = Array3::from_elem((4, 4, 4), 3e5); // 0.3 MPa
        let bubbles = Array3::from_elem((4, 4, 4), 5e7); // Optimal concentration
        let params = BBBParameters::default();

        let mut bbb = BBBOpening::new(pressure, bubbles, params);
        let result = bbb.simulate_opening();

        assert!(result.is_ok());
        assert!(bbb
            .permeability
            .permeability_factor
            .iter()
            .any(|&x| x > 1.0));
    }

    #[test]
    fn test_mechanical_index_calculation() {
        let pressure = Array3::from_elem((4, 4, 4), 1e5);
        let bubbles = Array3::from_elem((4, 4, 4), 1e7);
        let params = BBBParameters::default();

        let bbb = BBBOpening::new(pressure, bubbles, params);

        // MI for 100 kPa at 1 MHz should be ~0.1
        let mi = bbb.calculate_mechanical_index(1e5);
        assert!(mi > 0.05 && mi < 0.2);
    }

    #[test]
    fn test_safety_validation() {
        let pressure = Array3::from_elem((4, 4, 4), 3e5); // Safe pressure
        let bubbles = Array3::from_elem((4, 4, 4), 5e7);
        let params = BBBParameters::default();

        let mut bbb = BBBOpening::new(pressure, bubbles, params);
        bbb.simulate_opening().unwrap();

        let validation = bbb.validate_safety();
        assert!(validation.is_safe || validation.warnings.len() > 0);
    }

    #[test]
    fn test_treatment_protocol() {
        let pressure = Array3::from_elem((4, 4, 4), 1e5);
        let bubbles = Array3::from_elem((4, 4, 4), 1e7);
        let params = BBBParameters::default();

        let bbb = BBBOpening::new(pressure, bubbles, params);
        let protocol = bbb.generate_protocol();

        assert!(protocol.frequency > 0.0);
        assert!(protocol.duration > 0.0);
        assert!(!protocol.safety_checks.is_empty());
    }
}
