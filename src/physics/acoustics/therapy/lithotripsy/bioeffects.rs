//! Bioeffects assessment for lithotripsy safety.
//!
//! This module implements tissue damage assessment and safety monitoring for
//! extracorporeal shock wave lithotripsy (ESWL), including thermal and mechanical
//! bioeffects evaluation.

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

/// Bioeffects assessment model.
#[derive(Debug, Clone)]
pub struct BioeffectsModel {
    /// Model parameters
    parameters: BioeffectsParameters,
}

impl BioeffectsModel {
    /// Create new bioeffects model.
    pub fn new(parameters: BioeffectsParameters) -> Self {
        Self { parameters }
    }

    /// Get model parameters.
    pub fn parameters(&self) -> &BioeffectsParameters {
        &self.parameters
    }
}

impl Default for BioeffectsModel {
    fn default() -> Self {
        Self::new(BioeffectsParameters::default())
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
}

impl Default for SafetyAssessment {
    fn default() -> Self {
        Self {
            thermal_damage: 0.0,
            mechanical_damage: 0.0,
            safety_score: 1.0,
        }
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
