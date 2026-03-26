//! Validated kinetics database and results
//!
//! References:
//! - Buxton et al. (1988), Sehested et al. (1991)

use super::literature::LiteratureValue;
use crate::core::error::{KwaversResult, ValidationError};

/// Validated kinetics database
#[derive(Debug)]
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
    pub fn new() -> Self {
        Self {
            oh_recombination: LiteratureValue::from_range(4.5e9, 5.5e9),
            superoxide_dismutation: LiteratureValue::from_range(1.0e8, 2.2e8),
            peroxide_hydroxyl: LiteratureValue::from_range(2.4e7, 3.0e7),
            ozone_hydroxyl: LiteratureValue::from_range(0.8e8, 1.2e8),
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
