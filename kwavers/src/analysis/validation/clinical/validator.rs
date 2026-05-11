//! `ClinicalValidator` struct and construction.
//!
//! SRP: changes when the initial requirements registry changes.

use super::types::{ClinicalCategory, ClinicalRequirements, ClinicalStandard};
use std::collections::HashMap;

/// Clinical validation framework
#[allow(missing_debug_implementations)]
pub struct ClinicalValidator {
    pub(super) requirements: HashMap<(ClinicalStandard, ClinicalCategory), ClinicalRequirements>,
}

impl ClinicalValidator {
    #[must_use] 
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for ClinicalValidator {
    fn default() -> Self {
        let mut requirements = HashMap::new();

        let bmode_reqs = ClinicalRequirements {
            minimum_metrics: [
                ("contrast_resolution".to_owned(), 30.0),
                ("dynamic_range".to_owned(), 60.0),
                ("snr".to_owned(), 20.0),
            ]
            .into(),
            maximum_errors: [
                ("distance_error".to_owned(), 5.0),
                ("area_error".to_owned(), 10.0),
                ("axial_resolution".to_owned(), 0.5),
                ("lateral_resolution".to_owned(), 1.0),
            ]
            .into(),
            safety_thresholds: [
                ("mechanical_index".to_owned(), 1.9),
                ("thermal_index".to_owned(), 6.0),
            ]
            .into(),
            standard: ClinicalStandard::FDA510k,
            category: ClinicalCategory::BMode,
        };
        requirements.insert(
            (ClinicalStandard::FDA510k, ClinicalCategory::BMode),
            bmode_reqs,
        );

        let safety_reqs = ClinicalRequirements {
            minimum_metrics: HashMap::new(),
            maximum_errors: HashMap::new(),
            safety_thresholds: [
                ("mechanical_index_max".to_owned(), 1.9),
                ("thermal_index_soft_max".to_owned(), 6.0),
                ("thermal_index_bone_max".to_owned(), 1.0),
                ("thermal_index_cranial_max".to_owned(), 1.0),
                ("spta_max".to_owned(), 720.0),
                ("sppa_max".to_owned(), 190.0),
            ]
            .into(),
            standard: ClinicalStandard::IEC60601_2_37,
            category: ClinicalCategory::Safety,
        };
        requirements.insert(
            (ClinicalStandard::IEC60601_2_37, ClinicalCategory::Safety),
            safety_reqs,
        );

        Self { requirements }
    }
}
