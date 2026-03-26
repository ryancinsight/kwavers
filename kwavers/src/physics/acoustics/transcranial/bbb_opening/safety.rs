use super::models::PermeabilityModels;
use super::simulator::BBBOpening;
use super::types::{SafetyValidation, TreatmentProtocol};

/// Safety and protocol generation for BBB Opening
impl BBBOpening {
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
        let models = PermeabilityModels::new(&self.parameters);
        
        let max_mi = self
            .acoustic_pressure
            .iter()
            .map(|&p| models.calculate_mechanical_index(p))
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
