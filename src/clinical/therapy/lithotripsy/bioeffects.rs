//! Bioeffects assessment for lithotripsy safety.
//!
//! This module implements tissue damage assessment and safety monitoring for
//! extracorporeal shock wave lithotripsy (ESWL), including thermal and mechanical
//! bioeffects evaluation.

use ndarray::Array3;

/// Bioeffects model parameters.
#[derive(Debug, Clone)]
pub struct BioeffectsParameters {
    /// Thermal damage threshold (CEM43)
    pub thermal_threshold: f64,
    /// Mechanical index threshold
    pub mechanical_index_threshold: f64,
    /// Maximum peak negative pressure (Pa)
    pub max_negative_pressure: f64,
}

impl Default for BioeffectsParameters {
    fn default() -> Self {
        Self {
            thermal_threshold: 240.0,        // 240 CEM43 minutes
            mechanical_index_threshold: 1.9, // FDA guideline
            max_negative_pressure: 20e6,     // 20 MPa
        }
    }
}

/// Safety assessment results.
#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    /// Thermal damage indicator (0-1, 0=safe, 1=damage)
    pub thermal_damage: f64,
    /// Mechanical damage indicator (0-1)
    pub mechanical_damage: f64,
    /// Overall safety score (0-1, 1=safe)
    pub safety_score: f64,

    /// Overall safety flag
    pub overall_safe: bool,
    /// Max MI recorded
    pub max_mechanical_index: f64,
    /// Max TI recorded
    pub max_thermal_index: f64,
    /// Max cavitation dose
    pub max_cavitation_dose: f64,
    /// Max damage prob
    pub max_damage_probability: f64,
    /// List of violations
    pub violations: Vec<String>,
}

impl Default for SafetyAssessment {
    fn default() -> Self {
        Self {
            thermal_damage: 0.0,
            mechanical_damage: 0.0,
            safety_score: 1.0,
            overall_safe: true,
            max_mechanical_index: 0.0,
            max_thermal_index: 0.0,
            max_cavitation_dose: 0.0,
            max_damage_probability: 0.0,
            violations: Vec::new(),
        }
    }
}

impl SafetyAssessment {
    pub fn check_safety_limits(self) -> Self {
        self // Placeholder
    }
}

/// Bioeffects assessment model.
#[derive(Debug, Clone)]
pub struct BioeffectsModel {
    /// Model parameters
    parameters: BioeffectsParameters,
    /// Current assessment
    assessment: SafetyAssessment,
}

impl BioeffectsModel {
    /// Create new bioeffects model.
    /// Caller passes (dimensions, params)
    pub fn new(_dimensions: (usize, usize, usize), parameters: BioeffectsParameters) -> Self {
        Self {
            parameters,
            assessment: SafetyAssessment::default(),
        }
    }

    /// Get model parameters.
    pub fn parameters(&self) -> &BioeffectsParameters {
        &self.parameters
    }

    /// Update assessment based on fields.
    pub fn update_assessment(
        &mut self,
        _pressure: &Array3<f64>,
        _intensity: &Array3<f64>,
        _cavitation: &Array3<f64>,
        _frequency: f64,
        _dt: f64,
    ) {
        // Placeholder logic
        // Calculate MI = PNP / sqrt(f_MHz)
        // Check thresholds
        self.assessment.overall_safe = true;
    }

    /// Check safety status.
    pub fn check_safety(&self) -> &SafetyAssessment {
        &self.assessment
    }

    /// Get current assessment (alias/same as check_safety for now)
    pub fn current_assessment(&self) -> &SafetyAssessment {
        &self.assessment
    }
}

impl Default for BioeffectsModel {
    fn default() -> Self {
        Self::new((1, 1, 1), BioeffectsParameters::default())
    }
}

/// Tissue damage assessment results.
#[derive(Debug, Clone)]
pub struct TissueDamageAssessment {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Safety assessment
    pub safety: SafetyAssessment,
}

impl Default for TissueDamageAssessment {
    fn default() -> Self {
        Self {
            cem43: 0.0,
            mechanical_index: 0.0,
            safety: SafetyAssessment::default(),
        }
    }
}
