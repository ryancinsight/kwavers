use super::types::BBBParameters;

/// Permeability models for BBB opening
#[derive(Debug, Clone)]
pub struct PermeabilityModels<'a> {
    parameters: &'a BBBParameters,
}

impl<'a> PermeabilityModels<'a> {
    pub fn new(parameters: &'a BBBParameters) -> Self {
        Self { parameters }
    }

    /// Calculate permeability enhancement factor
    pub fn calculate_permeability_enhancement(&self, pressure: f64, bubble_conc: f64) -> f64 {
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
    pub fn calculate_opening_duration(&self, pressure: f64, bubble_conc: f64) -> f64 {
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
    pub fn calculate_recovery_time(&self, pressure: f64, enhancement: f64) -> f64 {
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
    pub fn calculate_microbubble_effect(&self, bubble_conc: f64, pressure: f64) -> f64 {
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
    pub fn calculate_mechanical_index(&self, pressure: f64) -> f64 {
        // MI = p_peak / sqrt(f) in MPa and MHz
        let p_mpa = pressure / 1e6; // Convert to MPa
        let f_mhz = self.parameters.frequency / 1e6; // Convert to MHz

        p_mpa / f_mhz.sqrt()
    }
}
