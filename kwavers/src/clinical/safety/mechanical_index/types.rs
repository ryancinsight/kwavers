//! Types for mechanical index safety calculation.

use crate::core::constants::medical::{
    MI_CAVITATION_BOWEL, MI_CAVITATION_BRAIN, MI_CAVITATION_FETAL, MI_CAVITATION_LUNG,
    MI_CAVITATION_OPHTHALMIC, MI_CAVITATION_SOFT_TISSUE,
    MI_LIMIT_BOWEL, MI_LIMIT_BRAIN, MI_LIMIT_FETAL, MI_LIMIT_LUNG, MI_LIMIT_OPHTHALMIC,
    MI_LIMIT_SOFT_TISSUE,
};

/// Tissue types with different MI safety thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MechanicalIndexTissueType {
    /// General soft tissue (MI < 1.9)
    SoftTissue,
    /// Ophthalmic tissue (MI < 0.23)
    Ophthalmic,
    /// Lung tissue with gas bodies (MI < 0.7)
    Lung,
    /// Bowel tissue with gas bodies (MI < 0.7)
    Bowel,
    /// Fetal tissue (MI < 1.0 recommended)
    Fetal,
    /// Brain tissue (MI < 1.5)
    Brain,
}

impl MechanicalIndexTissueType {
    /// Get FDA/WFUMB recommended MI limit for this tissue type
    #[must_use]
    pub fn safety_limit(&self) -> f64 {
        match self {
            Self::SoftTissue => MI_LIMIT_SOFT_TISSUE,
            Self::Ophthalmic => MI_LIMIT_OPHTHALMIC,
            Self::Lung => MI_LIMIT_LUNG,
            Self::Bowel => MI_LIMIT_BOWEL,
            Self::Fetal => MI_LIMIT_FETAL,
            Self::Brain => MI_LIMIT_BRAIN,
        }
    }

    /// Get cavitation threshold estimate for this tissue
    #[must_use]
    pub fn cavitation_threshold(&self) -> f64 {
        match self {
            Self::SoftTissue => MI_CAVITATION_SOFT_TISSUE,
            Self::Ophthalmic => MI_CAVITATION_OPHTHALMIC,
            Self::Lung => MI_CAVITATION_LUNG,
            Self::Bowel => MI_CAVITATION_BOWEL,
            Self::Fetal => MI_CAVITATION_FETAL,
            Self::Brain => MI_CAVITATION_BRAIN,
        }
    }
}
/// Mechanical Index calculation result
#[derive(Debug, Clone)]
pub struct MechanicalIndexResult {
    /// Calculated MI value
    pub mi: f64,
    /// Peak rarefactional pressure (MPa)
    pub peak_rarefactional_pressure_mpa: f64,
    /// Center frequency (MHz)
    pub center_frequency_mhz: f64,
    /// Safety assessment
    pub safety_status: MechanicalIndexSafetyStatus,
    /// Distance from transducer where MI is calculated (cm)
    pub focal_distance_cm: f64,
    /// Tissue-specific safety limit
    pub safety_limit: f64,
}

/// Safety status based on MI value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MechanicalIndexSafetyStatus {
    /// MI well below safety limits
    Safe,
    /// MI approaching safety limits (>80% of limit)
    Caution,
    /// MI exceeds recommended limits
    Unsafe,
    /// MI at cavitation threshold
    CavitationRisk,
}

impl MechanicalIndexResult {
    /// Check if MI is within safety limits
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self.safety_status, MechanicalIndexSafetyStatus::Safe)
    }

    /// Get safety margin as percentage below limit
    #[must_use]
    pub fn safety_margin_percent(&self) -> f64 {
        ((self.safety_limit - self.mi) / self.safety_limit) * 100.0
    }

    /// Format result for display
    #[must_use]
    pub fn format_report(&self) -> String {
        format!(
            "Mechanical Index Report\n\
             ========================\n\
             MI Value: {:.3}\n\
             Peak Rarefactional Pressure: {:.3} MPa\n\
             Center Frequency: {:.2} MHz\n\
             Focal Distance: {:.1} cm\n\
             Safety Limit: {:.2}\n\
             Safety Status: {:?}\n\
             Safety Margin: {:.1}%\n",
            self.mi,
            self.peak_rarefactional_pressure_mpa,
            self.center_frequency_mhz,
            self.focal_distance_cm,
            self.safety_limit,
            self.safety_status,
            self.safety_margin_percent()
        )
    }
}
