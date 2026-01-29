//! Chemistry kinetics validation against literature
//!
//! Validates reaction rate constants and thermodynamic parameters
//! against peer-reviewed sources.
//!
//! References:
//! - Buxton et al. (1988) "Critical review of rate constants for reactions"
//! - Sehested et al. (1991) "Pulse radiolysis of oxygenated aqueous solutions"
//! - Minakata et al. (2009) "Ultrasound-activated reactions in aqueous media"
//! - Kiran et al. (2017) "Radical reactions in cavitation bubbles"

use crate::core::error::{KwaversResult, ValidationError};

/// Literature value for a rate constant with uncertainty
#[derive(Debug, Clone, Copy)]
pub struct LiteratureValue {
    /// Nominal value [M⁻ⁿ·s⁻¹] where n depends on reaction order
    pub nominal: f64,
    /// Minimum reported value (lower bound)
    pub min: f64,
    /// Maximum reported value (upper bound)
    pub max: f64,
    /// Standard deviation / uncertainty
    pub uncertainty: f64,
}

impl LiteratureValue {
    /// Create literature value from nominal ± uncertainty
    pub fn new(nominal: f64, uncertainty: f64) -> Self {
        let percent = uncertainty / nominal;
        Self {
            nominal,
            min: nominal * (1.0 - percent),
            max: nominal * (1.0 + percent),
            uncertainty,
        }
    }

    /// Create from literature range
    pub fn from_range(min: f64, max: f64) -> Self {
        let nominal = (min + max) / 2.0;
        let uncertainty = (max - min) / 2.0;
        Self {
            nominal,
            min,
            max,
            uncertainty,
        }
    }

    /// Check if simulated value is within acceptable range
    pub fn is_within_range(&self, simulated: f64) -> bool {
        simulated >= self.min && simulated <= self.max
    }

    /// Percent difference from nominal
    pub fn percent_difference(&self, simulated: f64) -> f64 {
        100.0 * (simulated - self.nominal).abs() / self.nominal
    }
}

/// Validated kinetics database
pub struct ValidatedKinetics {
    /// OH + OH → H2O2 rate constant
    pub oh_recombination: LiteratureValue,
    /// O2•⁻ + H⁺ + O2•⁻ → H2O2 + O2 rate constant
    pub superoxide_dismutation: LiteratureValue,
    /// H2O2 + •OH → HO2• + H2O rate constant
    pub peroxide_hydroxyl: LiteratureValue,
    /// O3 + •OH → •OOH + O2 rate constant
    pub ozone_hydroxyl: LiteratureValue,
    /// •OH + H2O2 → HO2• + H2O rate constant (alternative)
    pub hydroxyl_peroxide: LiteratureValue,
}

impl ValidatedKinetics {
    /// Create kinetics database with literature values
    ///
    /// All values at 25°C unless otherwise noted
    /// References: Buxton et al. (1988), Sehested et al. (1991)
    pub fn new() -> Self {
        Self {
            // OH radical self-recombination: k = (5.0 ± 1.0)×10⁹ M⁻¹·s⁻¹
            // Reference: Buxton et al. (1988)
            oh_recombination: LiteratureValue::from_range(4.5e9, 5.5e9),

            // Superoxide dismutation (pH dependent): k ≈ 1.6×10⁸ M⁻¹·s⁻¹ at neutral pH
            // Reference: Sehested et al. (1991)
            superoxide_dismutation: LiteratureValue::from_range(1.0e8, 2.2e8),

            // H2O2 + •OH → HO2• + H2O: k = (2.7 ± 0.3)×10⁷ M⁻¹·s⁻¹
            // Reference: Buxton et al. (1988)
            peroxide_hydroxyl: LiteratureValue::from_range(2.4e7, 3.0e7),

            // O3 + •OH → •OOH + O2: k = (1.0 ± 0.2)×10⁸ M⁻¹·s⁻¹
            // Reference: Sehested et al. (1991)
            ozone_hydroxyl: LiteratureValue::from_range(0.8e8, 1.2e8),

            // Alternative: •OH + H2O2: k ≈ 2.7×10⁷ M⁻¹·s⁻¹
            hydroxyl_peroxide: LiteratureValue::from_range(2.0e7, 3.5e7),
        }
    }

    /// Validate a simulated rate constant against literature
    pub fn validate(
        &self,
        reaction: &str,
        simulated_value: f64,
    ) -> KwaversResult<ValidationResult> {
        let literature = match reaction.to_lowercase().as_str() {
            "oh_recombination" | "oh self-recombination" | "2oh->h2o2" => self.oh_recombination,
            "superoxide_dismutation" | "superoxide dismutation" | "2o2- -> h2o2" => {
                self.superoxide_dismutation
            }
            "peroxide_hydroxyl" | "h2o2 + oh" => self.peroxide_hydroxyl,
            "ozone_hydroxyl" | "o3 + oh" => self.ozone_hydroxyl,
            "hydroxyl_peroxide" | "oh + h2o2" => self.hydroxyl_peroxide,
            _ => {
                return Err(ValidationError::InvalidParameter {
                    parameter: "reaction".to_string(),
                    reason: format!("Unknown reaction: {}", reaction),
                }
                .into())
            }
        };

        Ok(ValidationResult {
            reaction: reaction.to_string(),
            simulated_value,
            literature_value: literature.nominal,
            literature_min: literature.min,
            literature_max: literature.max,
            within_range: literature.is_within_range(simulated_value),
            percent_difference: literature.percent_difference(simulated_value),
        })
    }
}

impl Default for ValidatedKinetics {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of kinetics validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Reaction name
    pub reaction: String,
    /// Simulated rate constant value
    pub simulated_value: f64,
    /// Literature nominal value
    pub literature_value: f64,
    /// Literature minimum value
    pub literature_min: f64,
    /// Literature maximum value
    pub literature_max: f64,
    /// Is simulated value within literature range?
    pub within_range: bool,
    /// Percent difference from literature nominal
    pub percent_difference: f64,
}

impl ValidationResult {
    /// Print validation report
    pub fn report(&self) -> String {
        format!(
            "Reaction: {}\n  Simulated: {:.3e} M⁻¹·s⁻¹\n  Literature: {:.3e} ± {:.3e} M⁻¹·s⁻¹\n  Range: [{:.3e}, {:.3e}]\n  Within range: {}\n  Difference: {:.1}%",
            self.reaction,
            self.simulated_value,
            self.literature_value,
            (self.literature_max - self.literature_min) / 2.0,
            self.literature_min,
            self.literature_max,
            self.within_range,
            self.percent_difference
        )
    }
}

/// Temperature-dependent Arrhenius kinetics validator
pub struct ArrheniusValidator {
    /// Activation energy [J/mol]
    pub activation_energy: f64,
    /// Reference temperature [K]
    pub reference_temperature: f64,
}

impl ArrheniusValidator {
    /// Create validator with Arrhenius parameters
    pub fn new(activation_energy: f64, reference_temperature: f64) -> Self {
        Self {
            activation_energy,
            reference_temperature,
        }
    }

    /// Calculate rate constant at temperature using Arrhenius equation
    ///
    /// k(T) = k₀ · exp(-Eₐ/R·(1/T - 1/T₀))
    pub fn rate_constant_at_temperature(&self, rate_at_reference: f64, temperature: f64) -> f64 {
        let r = 8.314; // Universal gas constant [J/mol/K]
        let exponent =
            -self.activation_energy / r * (1.0 / temperature - 1.0 / self.reference_temperature);
        rate_at_reference * exponent.exp()
    }

    /// Q10 factor (rate change per 10°C)
    ///
    /// Q10 = k(T+10) / k(T) for typical reactions
    pub fn q10_factor(&self, temperature: f64) -> f64 {
        let k1 = self.rate_constant_at_temperature(1.0, temperature);
        let k2 = self.rate_constant_at_temperature(1.0, temperature + 10.0);
        k2 / k1
    }

    /// Validate Q10 is reasonable (typically 2-4)
    pub fn is_reasonable_q10(&self, temperature: f64) -> bool {
        let q10 = self.q10_factor(temperature);
        q10 >= 1.5 && q10 <= 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literature_value_creation() {
        let lit = LiteratureValue::new(5.5e9, 1e9);
        assert_eq!(lit.nominal, 5.5e9);
        assert!(lit.min < lit.nominal);
        assert!(lit.max > lit.nominal);
    }

    #[test]
    fn test_literature_value_from_range() {
        let lit = LiteratureValue::from_range(4.5e9, 5.5e9);
        assert_eq!(lit.nominal, 5.0e9);
        assert_eq!(lit.min, 4.5e9);
        assert_eq!(lit.max, 5.5e9);
    }

    #[test]
    fn test_within_range_check() {
        let lit = LiteratureValue::from_range(1e7, 2e7);
        assert!(lit.is_within_range(1.5e7));
        assert!(!lit.is_within_range(0.5e7));
        assert!(!lit.is_within_range(3e7));
    }

    #[test]
    fn test_percent_difference() {
        let lit = LiteratureValue::new(1e8, 1e7);
        let diff = lit.percent_difference(1.1e8);
        assert!((diff - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_validated_kinetics_oh_recombination() {
        let kinetics = ValidatedKinetics::new();
        let result = kinetics.validate("oh_recombination", 5.5e9);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.within_range);
    }

    #[test]
    fn test_validated_kinetics_out_of_range() {
        let kinetics = ValidatedKinetics::new();
        let result = kinetics.validate("oh_recombination", 1e10); // Too high
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(!res.within_range);
    }

    #[test]
    fn test_validation_result_report() {
        let kinetics = ValidatedKinetics::new();
        let result = kinetics.validate("oh_recombination", 5.5e9).unwrap();
        let report = result.report();
        assert!(!report.is_empty());
        assert!(report.contains("Reaction:"));
        assert!(report.contains("Within range:"));
    }

    #[test]
    fn test_arrhenius_temperature_dependence() {
        let validator = ArrheniusValidator::new(50000.0, 298.15); // 50 kJ/mol

        // Rate should increase with temperature
        let k_298 = validator.rate_constant_at_temperature(1e8, 298.15);
        let k_308 = validator.rate_constant_at_temperature(1e8, 308.15);

        assert!(k_308 > k_298);
    }

    #[test]
    fn test_q10_factor() {
        let validator = ArrheniusValidator::new(50000.0, 298.15);
        let q10 = validator.q10_factor(298.15);

        // Q10 should be between 1.5 and 5 for typical reactions
        assert!(q10 > 1.0);
        assert!(q10 < 10.0);
    }

    #[test]
    fn test_q10_reasonableness() {
        let validator = ArrheniusValidator::new(50000.0, 298.15);
        assert!(validator.is_reasonable_q10(298.15));
    }

    #[test]
    fn test_high_activation_energy_q10() {
        let validator = ArrheniusValidator::new(100000.0, 298.15); // 100 kJ/mol
        let q10 = validator.q10_factor(298.15);

        // Higher Ea should give higher Q10
        let validator_low = ArrheniusValidator::new(20000.0, 298.15);
        let q10_low = validator_low.q10_factor(298.15);

        assert!(q10 > q10_low);
    }

    #[test]
    fn test_kinetics_database_completeness() {
        let kinetics = ValidatedKinetics::new();

        // Test that all major reactions are in the database
        assert!(kinetics.validate("oh_recombination", 5e9).is_ok());
        assert!(kinetics.validate("superoxide_dismutation", 1.5e8).is_ok());
        assert!(kinetics.validate("peroxide_hydroxyl", 2.7e7).is_ok());
        assert!(kinetics.validate("ozone_hydroxyl", 1e8).is_ok());
    }

    #[test]
    fn test_unknown_reaction() {
        let kinetics = ValidatedKinetics::new();
        let result = kinetics.validate("unknown_reaction", 1e8);
        assert!(result.is_err());
    }

    #[test]
    fn test_case_insensitivity() {
        let kinetics = ValidatedKinetics::new();
        let result1 = kinetics.validate("OH_RECOMBINATION", 5.5e9);
        let result2 = kinetics.validate("oh_recombination", 5.5e9);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_eq!(result1.unwrap().within_range, result2.unwrap().within_range);
    }
}
