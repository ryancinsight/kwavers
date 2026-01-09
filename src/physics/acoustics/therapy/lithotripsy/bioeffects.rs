//! Bioeffects Modeling for Lithotripsy
//!
//! Assesses tissue damage from shock waves, cavitation, and thermal effects
//! during lithotripsy. Ensures safe treatment by monitoring damage thresholds.
//!
//! ## Key Damage Mechanisms
//!
//! 1. **Mechanical Damage**: Cavitation bubble collapse, microjet impacts
//! 2. **Acoustic Damage**: High-intensity ultrasound tissue heating
//! 3. **Shock Wave Damage**: Direct mechanical trauma from shock fronts
//! 4. **Vascular Damage**: Blood vessel rupture from cavitation
//!
//! ## Safety Metrics
//!
//! - Mechanical Index (MI): MI = p_r / √f < 1.9
//! - Thermal Index (TI): TI = W / (Wdeg + Wα) where W is power, Wdeg is degassed power
//! - Cavitation Dose: Cumulative cavitation activity
//!
//! ## References
//!
//! - AIUM (1992): "Bioeffects and Safety of Diagnostic Ultrasound"
//! - IEC 62359 (2010): "Ultrasonics - Field characterization - Test methods for the determination of thermal and mechanical indices"
//! - Sapozhnikov et al. (2007): "Tissue erosion using cavitation effects of shock waves"

use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Bioeffects assessment parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioeffectsParameters {
    /// Mechanical index threshold
    pub mi_threshold: f64,
    /// Thermal index threshold
    pub ti_threshold: f64,
    /// Cavitation dose threshold
    pub cavitation_threshold: f64,
    /// Tissue perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f64,
    /// Blood perfusion rate [1/s]
    pub blood_perfusion: f64,
    /// Tissue density [kg/m³]
    pub tissue_density: f64,
    /// Acoustic absorption coefficient [Np/m]
    pub absorption_coefficient: f64,
}

impl Default for BioeffectsParameters {
    fn default() -> Self {
        Self {
            mi_threshold: 1.9,           // FDA limit
            ti_threshold: 6.0,           // Soft tissue limit
            cavitation_threshold: 100.0, // Arbitrary units
            perfusion_rate: 0.001,       // 1/s
            thermal_conductivity: 0.5,   // W/(m·K)
            specific_heat: 3600.0,       // J/(kg·K)
            blood_perfusion: 0.01,       // 1/s
            tissue_density: 1050.0,      // kg/m³
            absorption_coefficient: 5.0, // Np/m at 1 MHz
        }
    }
}

/// Tissue damage assessment
#[derive(Debug)]
pub struct TissueDamageAssessment {
    /// Mechanical index field
    mechanical_index: Array3<f64>,
    /// Thermal index field
    thermal_index: Array3<f64>,
    /// Temperature rise field [°C]
    temperature_rise: Array3<f64>,
    /// Cavitation dose field
    cavitation_dose: Array3<f64>,
    /// Tissue damage probability field (0-1)
    damage_probability: Array3<f64>,
    /// Parameters
    params: BioeffectsParameters,
}

impl TissueDamageAssessment {
    /// Create new tissue damage assessment
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize), params: BioeffectsParameters) -> Self {
        Self {
            mechanical_index: Array3::zeros(grid_shape),
            thermal_index: Array3::zeros(grid_shape),
            temperature_rise: Array3::zeros(grid_shape),
            cavitation_dose: Array3::zeros(grid_shape),
            damage_probability: Array3::zeros(grid_shape),
            params,
        }
    }

    /// Assess bioeffects from acoustic fields
    ///
    /// # Arguments
    /// * `pressure_field` - Acoustic pressure field [Pa]
    /// * `intensity_field` - Acoustic intensity field [W/m²]
    /// * `cavitation_field` - Cavitation activity field
    /// * `frequency` - Acoustic frequency [Hz]
    /// * `exposure_time` - Total exposure time [s]
    pub fn assess_bioeffects(
        &mut self,
        pressure_field: &Array3<f64>,
        intensity_field: &Array3<f64>,
        cavitation_field: &Array3<f64>,
        frequency: f64,
        exposure_time: f64,
    ) {
        // Calculate mechanical index
        self.calculate_mechanical_index(pressure_field, frequency);

        // Calculate thermal effects
        self.calculate_thermal_effects(intensity_field, exposure_time);

        // Assess cavitation effects
        self.assess_cavitation_effects(cavitation_field, exposure_time);

        // Calculate overall damage probability
        self.calculate_damage_probability();
    }

    /// Calculate mechanical index: MI = p_r / √f
    ///
    /// Where:
    /// - p_r is peak rarefactional pressure [MPa]
    /// - f is frequency [MHz]
    fn calculate_mechanical_index(&mut self, pressure_field: &Array3<f64>, frequency: f64) {
        let f_mhz = frequency / 1e6; // Convert to MHz

        let (nx, ny, nz) = pressure_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pressure_pa = pressure_field[[i, j, k]];
                    let pressure_mpa = pressure_pa / 1e6; // Convert to MPa

                    // MI uses the more negative (rarefactional) pressure
                    let p_r = if pressure_mpa < 0.0 {
                        -pressure_mpa // Take absolute value for rarefaction
                    } else {
                        0.0 // No rarefaction
                    };

                    let mi = p_r / f_mhz.sqrt();
                    self.mechanical_index[[i, j, k]] = mi;
                }
            }
        }
    }

    /// Calculate thermal effects and thermal index
    fn calculate_thermal_effects(&mut self, intensity_field: &Array3<f64>, _exposure_time: f64) {
        // Simplified thermal model using bioheat equation
        // dT/dt = (α I) / (ρ c) - perfusion_rate * (T - T_blood)

        let _t_blood = 37.0; // Blood temperature (°C)
        let _ambient_temp = 37.0; // Initial tissue temperature (°C)

        let (nx, ny, nz) = intensity_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let intensity = intensity_field[[i, j, k]];

                    // Heating rate: dT/dt = (α I) / (ρ c_p)
                    let heating_rate = (self.params.absorption_coefficient * intensity)
                        / (self.params.tissue_density * self.params.specific_heat);

                    // Steady-state temperature rise (simplified)
                    let temp_rise = heating_rate / self.params.perfusion_rate;

                    self.temperature_rise[[i, j, k]] = temp_rise;

                    // Thermal Index: TI = T_rise / 1°C (soft tissue)
                    // Actually: TI = W / (W_degassed + W_attenuation)
                    // Simplified version:
                    self.thermal_index[[i, j, k]] = temp_rise;
                }
            }
        }
    }

    /// Assess cavitation effects and calculate cavitation dose
    fn assess_cavitation_effects(&mut self, cavitation_field: &Array3<f64>, exposure_time: f64) {
        // Cavitation dose accumulates over time
        let dose_rate = 1.0; // Arbitrary units per second

        let (nx, ny, nz) = cavitation_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let cavitation_activity = cavitation_field[[i, j, k]];

                    // Dose increases with cavitation activity
                    let dose_increment = cavitation_activity * dose_rate * exposure_time;
                    self.cavitation_dose[[i, j, k]] += dose_increment;
                }
            }
        }
    }

    /// Calculate overall tissue damage probability
    fn calculate_damage_probability(&mut self) {
        let (nx, ny, nz) = self.damage_probability.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mi = self.mechanical_index[[i, j, k]];
                    let ti = self.thermal_index[[i, j, k]];
                    let cavitation = self.cavitation_dose[[i, j, k]];

                    // Damage probability based on exceeding thresholds
                    let mi_damage = if mi > self.params.mi_threshold {
                        0.8
                    } else {
                        0.0
                    };
                    let ti_damage = if ti > self.params.ti_threshold {
                        0.6
                    } else {
                        0.0
                    };
                    let cav_damage = if cavitation > self.params.cavitation_threshold {
                        0.4
                    } else {
                        0.0
                    };

                    // Combined damage (simplified model)
                    let combined_damage =
                        1.0 - (1.0 - mi_damage) * (1.0 - ti_damage) * (1.0 - cav_damage);
                    self.damage_probability[[i, j, k]] = combined_damage;
                }
            }
        }
    }

    /// Check if treatment is within safety limits
    #[must_use]
    pub fn check_safety_limits(&self) -> SafetyAssessment {
        let max_mi = self.mechanical_index.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_ti = self.thermal_index.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_cavitation = self.cavitation_dose.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_damage_prob = self
            .damage_probability
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));

        let mi_safe = max_mi <= self.params.mi_threshold;
        let ti_safe = max_ti <= self.params.ti_threshold;
        let cavitation_safe = max_cavitation <= self.params.cavitation_threshold;
        let overall_safe = mi_safe && ti_safe && cavitation_safe;

        SafetyAssessment {
            overall_safe,
            max_mechanical_index: max_mi,
            max_thermal_index: max_ti,
            max_cavitation_dose: max_cavitation,
            max_damage_probability: max_damage_prob,
            violations: vec![
                if !mi_safe {
                    Some("Mechanical Index".to_string())
                } else {
                    None
                },
                if !ti_safe {
                    Some("Thermal Index".to_string())
                } else {
                    None
                },
                if !cavitation_safe {
                    Some("Cavitation Dose".to_string())
                } else {
                    None
                },
            ]
            .into_iter()
            .flatten()
            .collect(),
        }
    }

    /// Get mechanical index field
    #[must_use]
    pub fn mechanical_index(&self) -> &Array3<f64> {
        &self.mechanical_index
    }

    /// Get thermal index field
    #[must_use]
    pub fn thermal_index(&self) -> &Array3<f64> {
        &self.thermal_index
    }

    /// Get temperature rise field
    #[must_use]
    pub fn temperature_rise(&self) -> &Array3<f64> {
        &self.temperature_rise
    }

    /// Get cavitation dose field
    #[must_use]
    pub fn cavitation_dose(&self) -> &Array3<f64> {
        &self.cavitation_dose
    }

    /// Get damage probability field
    #[must_use]
    pub fn damage_probability(&self) -> &Array3<f64> {
        &self.damage_probability
    }
}

/// Safety assessment result
#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    /// Overall safety status
    pub overall_safe: bool,
    /// Maximum mechanical index observed
    pub max_mechanical_index: f64,
    /// Maximum thermal index observed
    pub max_thermal_index: f64,
    /// Maximum cavitation dose observed
    pub max_cavitation_dose: f64,
    /// Maximum damage probability
    pub max_damage_probability: f64,
    /// List of safety violations
    pub violations: Vec<String>,
}

/// Bioeffects model combining all assessment components
#[derive(Debug)]
pub struct BioeffectsModel {
    /// Tissue damage assessment
    assessment: TissueDamageAssessment,
    /// Safety monitoring history
    safety_history: Vec<SafetyAssessment>,
    /// Treatment exposure time [s]
    total_exposure_time: f64,
}

impl BioeffectsModel {
    /// Create new bioeffects model
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize), params: BioeffectsParameters) -> Self {
        let assessment = TissueDamageAssessment::new(grid_shape, params);

        Self {
            assessment,
            safety_history: Vec::new(),
            total_exposure_time: 0.0,
        }
    }

    /// Update bioeffects assessment
    pub fn update_assessment(
        &mut self,
        pressure_field: &Array3<f64>,
        intensity_field: &Array3<f64>,
        cavitation_field: &Array3<f64>,
        frequency: f64,
        time_step: f64,
    ) {
        self.total_exposure_time += time_step;

        self.assessment.assess_bioeffects(
            pressure_field,
            intensity_field,
            cavitation_field,
            frequency,
            self.total_exposure_time,
        );
    }

    /// Check current safety status
    #[must_use]
    pub fn check_safety(&mut self) -> &SafetyAssessment {
        let assessment = self.assessment.check_safety_limits();
        self.safety_history.push(assessment);
        self.safety_history.last().unwrap()
    }

    /// Get safety history
    #[must_use]
    pub fn safety_history(&self) -> &[SafetyAssessment] {
        &self.safety_history
    }

    /// Get current assessment
    #[must_use]
    pub fn current_assessment(&self) -> &TissueDamageAssessment {
        &self.assessment
    }

    /// Reset assessment for new treatment
    pub fn reset(&mut self) {
        self.total_exposure_time = 0.0;
        self.safety_history.clear();
        // Note: assessment fields are reset in assess_bioeffects if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mechanical_index_calculation() {
        let params = BioeffectsParameters::default();
        let grid_shape = (5, 5, 5);

        let mut assessment = TissueDamageAssessment::new(grid_shape, params);

        // Create pressure field with -10 MPa rarefactional pressure
        let pressure_field = Array3::from_elem(grid_shape, -10e6); // -10 MPa
        let intensity_field = Array3::zeros(grid_shape);
        let cavitation_field = Array3::zeros(grid_shape);

        assessment.assess_bioeffects(
            &pressure_field,
            &intensity_field,
            &cavitation_field,
            1e6,
            1.0,
        );

        // MI = p_r / √f = 10 / √1 = 10 (should be high)
        let mi = assessment.mechanical_index()[[2, 2, 2]];
        assert!(mi > 9.0 && mi < 11.0);
    }

    #[test]
    fn test_safety_assessment() {
        let params = BioeffectsParameters::default();
        let grid_shape = (3, 3, 3);

        let mut assessment = TissueDamageAssessment::new(grid_shape, params);

        // Safe levels
        let pressure_field = Array3::from_elem(grid_shape, -1e6); // -1 MPa (safe)
        let intensity_field = Array3::zeros(grid_shape);
        let cavitation_field = Array3::zeros(grid_shape);

        assessment.assess_bioeffects(
            &pressure_field,
            &intensity_field,
            &cavitation_field,
            1e6,
            1.0,
        );

        let safety = assessment.check_safety_limits();
        assert!(safety.overall_safe);
        assert!(safety.violations.is_empty());
    }

    #[test]
    fn test_bioeffects_model() {
        let params = BioeffectsParameters::default();
        let grid_shape = (3, 3, 3);

        let mut model = BioeffectsModel::new(grid_shape, params);

        let pressure_field = Array3::from_elem(grid_shape, -1.5e6); // 1.5 MPa - below MI threshold
        let intensity_field = Array3::from_elem(grid_shape, 1000.0); // 1000 W/m²
        let cavitation_field = Array3::zeros(grid_shape);

        model.update_assessment(
            &pressure_field,
            &intensity_field,
            &cavitation_field,
            1e6,
            1.0,
        );

        let safety = model.check_safety();
        assert!(safety.overall_safe);

        // Should have one entry in history
        assert_eq!(model.safety_history().len(), 1);
    }
}
